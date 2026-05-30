import AppCompatibility
import AutocompleteCore
import Foundation
import ModelRuntime
import TokenProfiles

/// Real constrained, multi-branch decoder (M5, see ADR-010).
///
/// The engine drives the **existing** linear `LocalModelRuntime` protocol — to score a branch
/// it re-`prepare`s `basePrompt + branchTokens` and reads the next-token logits, relying on KV
/// prefix reuse to keep that cheap. Search is a deterministic best-first beam ordered by
/// cumulative log-probability; `temperature` / `topK` / `topP` shape the per-step candidate
/// pool (no RNG). Admissibility (required prefix + byte/trie constraints) and token policy
/// (exclusions, bias, stop behaviour, display width) come from the `AutocompleteProfile`, so the
/// engine works identically against the in-memory and the memory-mapped ACPF profile.
public final class ConstrainedGenerationEngine: CompletionGenerating {
    private let runtime: LocalModelRuntime
    private let profile: AutocompleteProfile
    private let compatibilityStore: AppCompatibilityStore
    private let configuration: DecodingConfiguration
    private let wordRecognizer: WordRecognizing?

    public init(
        runtime: LocalModelRuntime,
        profile: AutocompleteProfile,
        compatibilityStore: AppCompatibilityStore = AppCompatibilityStore(),
        configuration: DecodingConfiguration = DecodingConfiguration(),
        wordRecognizer: WordRecognizing? = nil
    ) {
        self.runtime = runtime
        self.profile = profile
        self.compatibilityStore = compatibilityStore
        self.configuration = configuration
        self.wordRecognizer = wordRecognizer
    }

    public func completions(for request: CompletionRequest) async throws -> [CompletionCandidate] {
        let policy = compatibilityStore.policy(for: request.context.target)
        guard policy.isCompletionEnabled else { return [] }
        guard policy.allowsMidLineCompletion || request.context.afterCursor.isEmpty else { return [] }

        let basePrompt = try runtime.tokenizer.tokenize(request.prompt)
        // A short tail of the prompt so sentence-boundary disambiguation can see context that
        // precedes the generated text (e.g. an abbreviation the prompt ends on).
        let promptTail = String(request.prompt.suffix(32))

        // Drops branches whose current word completes into a misspelling, mid-search, so the beam
        // keeps exploring correctly-spelled continuations instead (see ADR-015 / CurrentWordTypoGuard).
        let typoGuard = CurrentWordTypoGuard(recognizer: wordRecognizer, request: request)

        var live = [GenerationBranch(requiredPrefix: request.requiredPrefixBytes)]
        var finalized: [GenerationBranch] = []
        let maxDepth = max(0, request.maxCompletionTokens)

        depthLoop: for _ in 0..<maxDepth {
            try Task.checkCancellation()
            if live.isEmpty { break }

            var nextLive: [GenerationBranch] = []
            for branch in live {
                try Task.checkCancellation()

                try await runtime.prepare(promptTokens: basePrompt + branch.tokenIDs)
                let logits = try await runtime.logitsForNextToken()
                guard !logits.isEmpty else {
                    finalizeIfValid(branch, into: &finalized)
                    continue
                }

                let result = TokenSampler.rank(
                    logits: logits,
                    mode: request.mode,
                    profile: profile,
                    configuration: configuration,
                    isAdmissible: { profile.tokenAllowed($0, afterRequiredPrefix: branch.remainingPrefix) }
                )

                // The model's single most likely continuation being a terminator is the
                // signal to stop this branch and keep what we have so far.
                if let top = result.argmaxTokenID, isHardStop(top) {
                    finalizeIfValid(branch, into: &finalized)
                    continue
                }
                if result.tokens.isEmpty {
                    finalizeIfValid(branch, into: &finalized)
                    continue
                }

                for token in result.tokens {
                    let id = token.tokenID
                    if isHardStop(id) { continue } // never displayed; argmax handles "stop here"

                    let tokenBytes = profileBytes(for: id)
                    let outcome = branch.extending(
                        withToken: id,
                        bytes: tokenBytes,
                        logProbability: token.logProbability,
                        maxDisplayWidth: request.maxDisplayWidth
                    )

                    switch outcome {
                    case .inadmissiblePrefix, .invalidUTF8, .overWidth:
                        continue // drop this extension
                    case let .extended(child):
                        // If this token just closed the word the user is completing and that word is
                        // a misspelling, drop the branch now — never finalise it and never spend
                        // further beam budget continuing from the wrong spelling.
                        if await typoGuard.shouldDrop(parentText: branch.text, childText: child.text) {
                            continue
                        }
                        switch profile.stopBehavior(for: id) {
                        case .stopAndDisplay:
                            // The sentence-end flag is context-free; only stop on a *real*
                            // boundary. A false one ("1.", "Mr.", "e.g.") keeps generating so we
                            // don't truncate a numbered list / abbreviation mid-thought.
                            if SentenceBoundary.isTerminal(promptTail + child.text) {
                                finalizeIfValid(child, into: &finalized)
                            } else {
                                nextLive.append(child)
                            }
                        case .stopAndSuppress:
                            continue
                        case .continueGeneration:
                            nextLive.append(child)
                        }
                    }
                }
            }

            live = prune(nextLive)
        }

        // Branches still alive at the depth cap are valid candidates too.
        for branch in live {
            finalizeIfValid(branch, into: &finalized)
        }

        return makeCandidates(from: finalized, mode: request.mode)
    }

    // MARK: - Helpers

    private func isHardStop(_ id: TokenID) -> Bool {
        if let eos = runtime.metadata.eosTokenID, id == eos { return true }
        if let eot = runtime.metadata.eotTokenID, id == eot { return true }
        return profile.stopBehavior(for: id) == .stopAndSuppress
    }

    private func profileBytes(for id: TokenID) -> [UInt8] {
        if let bytes = profile.record(for: id)?.bytes { return bytes }
        return (try? runtime.tokenizer.rawBytes(for: id)) ?? []
    }

    private func finalizeIfValid(_ branch: GenerationBranch, into finalized: inout [GenerationBranch]) {
        if branch.isCompleteAndValid {
            finalized.append(branch)
        }
    }

    /// Keep the highest-scoring branches within the relative-cutoff margin and beam width.
    private func prune(_ branches: [GenerationBranch]) -> [GenerationBranch] {
        guard !branches.isEmpty else { return [] }
        var sorted = branches.sorted { $0.score > $1.score }
        let best = sorted[0].score
        sorted = sorted.filter { best - $0.score <= configuration.relativeCutoff }
        if configuration.branchWidth > 0 && sorted.count > configuration.branchWidth {
            sorted.removeLast(sorted.count - configuration.branchWidth)
        }
        return sorted
    }

    /// Dedupe by emitted text (best score wins), rank, and cap to `maxCandidates`.
    private func makeCandidates(from branches: [GenerationBranch], mode: CompletionMode) -> [CompletionCandidate] {
        var bestByText: [String: GenerationBranch] = [:]
        for branch in branches {
            if let existing = bestByText[branch.text], existing.score >= branch.score { continue }
            bestByText[branch.text] = branch
        }

        let ordered = bestByText.values.sorted { lhs, rhs in
            lhs.score != rhs.score ? lhs.score > rhs.score : lhs.text < rhs.text
        }

        return ordered.prefix(max(0, configuration.maxCandidates)).map { branch in
            CompletionCandidate(
                text: branch.text,
                tokenIDs: branch.tokenIDs,
                logProbability: Double(branch.score),
                displayWidth: branch.displayWidth,
                mode: mode
            )
        }
    }
}
