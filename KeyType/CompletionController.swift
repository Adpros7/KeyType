//
//  CompletionController.swift
//  KeyType
//
//  Orchestrates the live completion pipeline (M6): focused-field snapshots → prompt → constrained
//  generation → candidate filtering → inline ghost-text overlay, plus Tab/Shift+Tab acceptance via
//  TextInsertion. Generation runs off the main actor (the runtime is an actor) and is cancelled by
//  the next keystroke. See ADR-016.
//

import AppCompatibility
import AppKit
import AutocompleteCore
import CompletionUI
import ConstrainedGeneration
import Foundation
import LlamaModelRuntime
import MacContextCapture
import ModelRuntime
import Observation
import Prompting
import TextInsertion
import TokenProfiles
import os

@MainActor
@Observable
final class CompletionController {
    enum LoadState: Equatable {
        case idle
        case loading
        case ready
        case unavailable(String)
    }

    private let tracker: AccessibilityContextTracker
    private let compatibilityStore: AppCompatibilityStore
    private let presenter: InlineGhostTextPresenter
    private let placementResolver: OverlayPlacementResolver
    private let inserter: PasteboardCompletionInserter
    private let filter: DefaultCandidateFilter
    private let predictionLog = PredictionLog()
    private let log = Logger(subsystem: "com.pattonium.KeyType", category: "completion")

    private var engine: ConstrainedGenerationEngine?
    private var generationTask: Task<Void, Never>?
    private var debounceTask: Task<Void, Never>?
    private var listenerToken: UUID?
    /// Content signature of the last snapshot we acted on, so re-emitted snapshots whose text is
    /// unchanged (caret-geometry repolls) don't tear down and rebuild the overlay — that churn is
    /// what makes the ghost text flash.
    private var lastContextKey: String?

    private(set) var loadState: LoadState = .idle
    private(set) var isRunning = false

    /// The completion currently shown as ghost text and its originating context (used by the Tab
    /// acceptance controller). Nil when nothing is displayed.
    private(set) var visibleCandidate: CompletionCandidate?
    private var visibleContext: TextFieldContext?

    var completionsEnabled = true {
        didSet {
            if !completionsEnabled { reset() }
        }
    }

    init(
        tracker: AccessibilityContextTracker,
        compatibilityStore: AppCompatibilityStore = KeyTypeModuleGraph.makeCompatibilityStore()
    ) {
        self.tracker = tracker
        self.compatibilityStore = compatibilityStore
        self.presenter = InlineGhostTextPresenter()
        self.placementResolver = OverlayPlacementResolver(compatibilityStore: compatibilityStore)
        self.inserter = PasteboardCompletionInserter(
            planner: InsertionPlanner(compatibilityStore: compatibilityStore)
        )
        // The live typo defence is the in-beam guard (ADR-015); the output filter's typo net stays
        // inert here to avoid double spell-checking. All other taxonomy reasons are enforced.
        self.filter = DefaultCandidateFilter(compatibilityStore: compatibilityStore)
    }

    // MARK: - Lifecycle

    /// Load the model + profile + engine once, off the main actor. Safe to call repeatedly.
    func loadIfNeeded() {
        guard loadState == .idle else { return }
        loadState = .loading
        Task {
            do {
                let engine = try await Self.buildEngine(compatibilityStore: compatibilityStore)
                self.engine = engine
                self.loadState = .ready
                self.log.info("Completion engine ready")
            } catch {
                self.loadState = .unavailable("\(error)")
                self.log.error("Completion engine unavailable: \(error, privacy: .public)")
            }
        }
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true
        loadIfNeeded()
        listenerToken = tracker.addListener { [weak self] snapshot in
            self?.handle(snapshot)
        }
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false
        if let listenerToken {
            tracker.removeListener(listenerToken)
        }
        listenerToken = nil
        reset()
    }

    // MARK: - Pipeline

    private func handle(_ snapshot: FocusedFieldSnapshot?) {
        // Conditions under which there can be no suggestion: tear everything down and reset.
        guard completionsEnabled, loadState == .ready, let engine else { reset(); return }
        guard let snapshot, let caretRect = snapshot.caretRect, !caretRect.isEmpty else { reset(); return }

        let context = snapshot.context
        let policy = compatibilityStore.policy(for: context.target)
        guard policy.isCompletionEnabled,
              policy.allowsMidLineCompletion || context.afterCursor.isEmpty
        else { reset(); return }

        // No usable prefix → don't generate (Cotypist's `emptyPrompt` gate). A base model given an
        // empty before-cursor just continues the prompt scaffolding (e.g. echoes section headers),
        // so there is nothing worth showing until the user has typed something at the caret.
        guard !context.beforeCursor.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { reset(); return }

        // Skip identical re-emits — keep whatever is on screen, no flicker.
        let key = context.beforeCursor + "\u{1}" + context.afterCursor + "\u{1}" + context.target.bundleIdentifier
        guard key != lastContextKey else { return }
        lastContextKey = key

        guard let placement = placementResolver.placement(for: context) else { reset(); return }

        // Resolve the field font now, on the main actor, before suspending into generation.
        let font = FieldFontResolver.currentFont()
        let promptResult = KeyTypeModuleGraph.makePrompt(for: context, compatibilityStore: compatibilityStore)
        let request = CompletionRequest(
            context: context,
            prompt: promptResult.prompt,
            mode: .prose,
            maxCompletionTokens: 4,
            maxDisplayWidth: 60
        )

        // Debounce: coalesce rapid keystrokes, and DON'T hide the current ghost up front — we
        // transition directly old → new (or → hidden) when generation finishes, so typing updates
        // the suggestion in place instead of blinking it out and back in.
        generationTask?.cancel()
        debounceTask?.cancel()
        debounceTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 90_000_000)
            guard !Task.isCancelled, let self else { return }
            self.generationTask = Task { [weak self] in
                guard let self else { return }
                do {
                    let candidates = try await engine.completions(for: request)
                    try Task.checkCancellation()
                    self.present(candidates, request: request, placement: placement, font: font)
                } catch is CancellationError {
                    // Superseded by a newer keystroke — leave the current ghost as-is.
                } catch {
                    self.log.error("Generation failed: \(error, privacy: .public)")
                }
            }
        }
    }

    private func present(
        _ candidates: [CompletionCandidate],
        request: CompletionRequest,
        placement: OverlayPlacement,
        font: NSFont?
    ) {
        let ctx = PredictionLog.contextTail(request.context.beforeCursor)
        let ranked = candidates.prefix(5)
            .map { "\"\(PredictionLog.escape($0.text))\"" }
            .joined(separator: " | ")

        guard let best = candidates.first else {
            predictionLog.append("PREDICT ctx=\"\(ctx)\" → SUPPRESS(noCandidate)")
            clearCompletion()
            return
        }
        if let reason = filter.suppressionReason(for: best, request: request) {
            log.debug("Suppressed: \(String(describing: reason), privacy: .public)")
            predictionLog.append("PREDICT ctx=\"\(ctx)\" [\(ranked)] → SUPPRESS(\(reason))")
            clearCompletion()
            return
        }
        predictionLog.append("PREDICT ctx=\"\(ctx)\" [\(ranked)] → SHOWN \"\(PredictionLog.escape(best.text))\"")
        visibleCandidate = best
        visibleContext = request.context
        presenter.show(candidate: best, placement: placement, font: font)
    }

    private func clearCompletion() {
        presenter.hide()
        visibleCandidate = nil
        visibleContext = nil
    }

    /// Hide any ghost text and forget the last context, so the next snapshot is treated as new.
    private func reset() {
        debounceTask?.cancel()
        generationTask?.cancel()
        lastContextKey = nil
        clearCompletion()
    }

    // MARK: - Acceptance (driven by the Tab hotkey)

    /// True when there is a visible completion the user is allowed to accept with Tab.
    var canAcceptCompletion: Bool {
        guard visibleCandidate != nil, let context = visibleContext else { return false }
        return compatibilityStore.policy(for: context.target).allowsTabAcceptance
    }

    /// Tab: insert the next word of the suggestion. The induced text change regenerates a fresh
    /// completion from the new cursor position.
    func acceptNextWord() {
        guard canAcceptCompletion, let candidate = visibleCandidate, let context = visibleContext else { return }
        let (head, _) = NextWordSplitter.split(candidate.text)
        predictionLog.append(
            "ACCEPT(word) \"\(PredictionLog.escape(head))\" of \"\(PredictionLog.escape(candidate.text))\""
        )
        insert(text: head, context: context)
    }

    /// Shift+Tab: insert the whole suggestion.
    func acceptFullCompletion() {
        guard canAcceptCompletion, let candidate = visibleCandidate, let context = visibleContext else { return }
        predictionLog.append("ACCEPT(full) \"\(PredictionLog.escape(candidate.text))\"")
        insert(text: candidate.text, context: context)
    }

    private func insert(text: String, context: TextFieldContext) {
        guard !text.isEmpty else { return }
        let plan = inserter.planInsertion(candidate: CompletionCandidate(text: text), context: context)
        // Drop the dedupe key so the post-insertion snapshot always regenerates a fresh suggestion.
        lastContextKey = nil
        clearCompletion()
        Task {
            do {
                try await inserter.insert(plan: plan)
            } catch {
                log.error("Insertion failed: \(error, privacy: .public)")
            }
        }
    }

    // MARK: - Engine construction

    /// Builds the runtime + profile + engine. Marked `nonisolated` so the heavy ~0.3 s model load
    /// runs off the main actor (the call site `await`s it from a `Task`) rather than hitching the
    /// UI. Inlined here — rather than via `KeyTypeModuleGraph`, whose helpers are main-actor
    /// isolated by default — so every step stays off main.
    nonisolated private static func buildEngine(
        compatibilityStore: AppCompatibilityStore
    ) async throws -> ConstrainedGenerationEngine {
        let family = "qwen3-v151936"
        guard ModelContainer.defaultModelExists() else {
            throw CompletionLoadError.modelMissing(ModelContainer.defaultModelFilename)
        }
        let runtime = try LlamaModelRuntime(modelURL: try ModelContainer.modelURL())
        let profile = try MmapAutocompleteProfile.open(
            at: try ModelContainer.profileURL(family: family),
            tokenizerVocabSize: runtime.metadata.vocabularySize,
            tokenizerBytes: { try runtime.tokenizer.rawBytes(for: $0) },
            expectedModelFamily: family
        )
        return ConstrainedGenerationEngine(
            runtime: runtime,
            profile: profile,
            compatibilityStore: compatibilityStore,
            wordRecognizer: SystemWordRecognizer()
        )
    }

    enum CompletionLoadError: Error, CustomStringConvertible {
        case modelMissing(String)

        var description: String {
            switch self {
            case let .modelMissing(name):
                return "Model file '\(name)' not found in Application Support/KeyType/Models"
            }
        }
    }
}
