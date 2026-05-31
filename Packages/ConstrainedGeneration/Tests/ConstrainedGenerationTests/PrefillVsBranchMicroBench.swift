import AutocompleteCore
@testable import ConstrainedGeneration
import LlamaModelRuntime
import ModelRuntime
import Prompting
import TokenProfiles
import XCTest

// NOTE: temporary latency-investigation harness (ADR-043 follow-up). Skip-gated on the GGUF.

/// Temporary micro-benchmark (latency investigation). Separates the *one-time prefill* cost from
/// the *per-branch restore+decode* cost so we can attribute the ~87 ms warm completion latency to
/// either (a) processing the prompt once, or (b) the 12 anchored restore/decode calls the beam
/// makes. Run:
///   swift test --package-path Packages/ConstrainedGeneration --filter PrefillVsBranchMicroBench -c release
final class PrefillVsBranchMicroBench: XCTestCase {
    private static let family = "qwen3-v151936"

    private func load() throws -> LlamaModelRuntime {
        try XCTSkipUnless(ModelContainer.defaultModelExists(), "GGUF missing; skipping")
        return try LlamaModelRuntime(modelURL: try ModelContainer.modelURL(), contextLength: 2048, enableKVFork: true)
    }

    private func openProfile(_ runtime: LlamaModelRuntime) throws -> MmapAutocompleteProfile {
        let profileURL = try ModelContainer.profileURL(family: Self.family)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: profileURL.path), "profile missing")
        return try MmapAutocompleteProfile.open(
            at: profileURL,
            tokenizerVocabSize: runtime.metadata.vocabularySize,
            tokenizerBytes: { try runtime.tokenizer.rawBytes(for: $0) },
            expectedModelFamily: Self.family
        )
    }

    /// Min-of-`runs` wall-clock seconds of an async op (min rejects scheduler/thermal noise).
    private func minSeconds(_ runs: Int, _ block: () async throws -> Void) async rethrows -> Double {
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<runs {
            let start = DispatchTime.now()
            try await block()
            best = min(best, Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000)
        }
        return best
    }

    /// Ordinary least-squares slope/intercept for y = intercept + slope·x over a few sample points.
    private func linearFit(_ xs: [Double], _ ys: [Double]) -> (intercept: Double, slope: Double) {
        let n = Double(xs.count)
        let sx = xs.reduce(0, +), sy = ys.reduce(0, +)
        let sxx = zip(xs, xs).reduce(0) { $0 + $1.0 * $1.1 }
        let sxy = zip(xs, ys).reduce(0) { $0 + $1.0 * $1.1 }
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        return ((sy - slope * sx) / n, slope)
    }

    /// Detailed component profile (ADR-043 follow-up). Times directly-measurable primitives and
    /// linear-fits the ones that scale (prefill length, branch depth, batch width) so we can split
    /// the per-`llama_decode` *fixed floor* (full-model weight stream) from the per-token *forward*
    /// compute, the *restore* cost, and CPU-side readback/sampling — then reconcile against a real
    /// completion. Run:
    ///   swift test --package-path Packages/ConstrainedGeneration --filter testDetailedComponentProfile -c release
    func testDetailedComponentProfile() async throws {
        let runtime = try load()
        let profile = try openProfile(runtime)
        let config = DecodingConfiguration(maxCandidates: 5, enableFillInMiddle: true)

        // A long token source so we can take prefixes of arbitrary length.
        let longText = String(repeating: "The quick brown fox jumps over the lazy dog near the riverbank, and then ", count: 24)
        let longTokens = try runtime.tokenizer.tokenize(longText)
        func prefix(_ n: Int) -> [TokenID] { Array(longTokens.prefix(n)) }

        let anchor = prefix(128)
        let t0 = anchor.last ?? 0
        let t1 = anchor.dropLast().last ?? 0
        func suffix(_ k: Int) -> [TokenID] { (0..<k).map { $0 % 2 == 0 ? t0 : t1 } }

        // Warm kernels.
        await runtime.resetKVCache()
        _ = try await runtime.anchoredLogits(anchor: anchor, suffix: [])

        // ---- A. Cold prefill scaling: clear + decode N tokens, logits on last only ----
        let prefillNs = [16, 32, 64, 128, 256]
        var prefillMs: [Double] = []
        for n in prefillNs {
            let p = prefix(n)
            let s = try await minSeconds(5) {
                await runtime.resetKVCache()
                _ = try await runtime.anchoredLogits(anchor: p, suffix: [])
            }
            prefillMs.append(s * 1000)
        }
        let prefillFit = linearFit(prefillNs.map(Double.init), prefillMs)

        // ---- B. Branch depth scaling: resident anchor, restore + decode K-token suffix ----
        _ = try await runtime.anchoredLogits(anchor: anchor, suffix: [])  // make anchor resident
        let branchKs = [1, 2, 3, 4, 6, 8]
        var branchMs: [Double] = []
        for k in branchKs {
            let suf = suffix(k)
            let s = try await minSeconds(8) { _ = try await runtime.anchoredLogits(anchor: anchor, suffix: suf) }
            branchMs.append(s * 1000)
        }
        let branchFit = linearFit(branchKs.map(Double.init), branchMs)

        // ---- C. Batched width amortization: W branches (suffix len 1) in ONE decode ----
        let widths = [1, 2, 3, 4]
        var widthMs: [Double] = []
        for w in widths {
            let sufs = Array(repeating: suffix(1), count: w)
            let s = try await minSeconds(8) { _ = try await runtime.anchoredLogitsBatch(anchor: anchor, suffixes: sufs) }
            widthMs.append(s * 1000)
        }
        let widthFit = linearFit(widths.map(Double.init), widthMs)

        // ---- D. CPU-side readback + materialization (logitsForNextToken) ----
        _ = try await runtime.anchoredLogits(anchor: anchor, suffix: suffix(1))
        let readbackMs = try await minSeconds(20) { _ = try await runtime.logitsForNextToken() } * 1000

        // ---- E. CPU-side sampler (TokenSampler.rank over the real vocab) ----
        let logits = try await runtime.anchoredLogits(anchor: anchor, suffix: suffix(1))
        var samplerMs = Double.greatestFiniteMagnitude
        for _ in 0..<20 {
            let start = DispatchTime.now()
            _ = TokenSampler.rank(logits: logits, mode: .prose, profile: profile, configuration: config, isAdmissible: { _ in true })
            samplerMs = min(samplerMs, Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
        }

        // ---- F. Model size cross-check for the per-decode weight-stream floor ----
        let modelURL = try ModelContainer.modelURL()
        let modelAttrs = try? FileManager.default.attributesOfItem(atPath: modelURL.path)
        let modelBytes = (modelAttrs?[.size] as? NSNumber)?.intValue ?? 0
        let modelGiB = Double(modelBytes) / 1_073_741_824.0
        // Effective BW implied if the per-decode floor were pure weight streaming.
        let floorMs = branchFit.intercept
        let impliedBWGiBs = floorMs > 0 ? modelGiB / (floorMs / 1000.0) : 0

        // ---- Derived components ----
        let perDecodeFloor = branchFit.intercept           // restore + dispatch + full-model stream (incl. LM head)
        let restoreCost = branchFit.intercept - prefillFit.intercept  // branch floor includes a restore; prefill does not
        let perTokenForwardSmall = branchFit.slope         // small-batch sequential forward / token
        let perTokenForwardParallel = prefillFit.slope     // parallel prefill forward / token
        let perBranchMarginal = widthFit.slope             // restore + forward + extra LM-head output per added branch

        print("\n================ KeyType detailed component profile (ADR-043) ================")
        print(String(format: "  model: %@  (%.2f GiB on disk)", modelURL.lastPathComponent, modelGiB))
        print("  -- raw measurements (min-of-N, warm, release) --")
        print("  A) cold prefill (ms) by tokens \(prefillNs): \(prefillMs.map { String(format: "%.1f", $0) })")
        print(String(format: "     fit: %.2f ms + %.3f ms/token   (intercept = per-decode floor; slope = parallel fwd/token)", prefillFit.intercept, prefillFit.slope))
        print("  B) branch restore+decode (ms) by depth \(branchKs): \(branchMs.map { String(format: "%.1f", $0) })")
        print(String(format: "     fit: %.2f ms + %.3f ms/token   (intercept = floor+restore; slope = small-batch fwd/token)", branchFit.intercept, branchFit.slope))
        print("  C) batched width (ms) by branches \(widths): \(widthMs.map { String(format: "%.1f", $0) })")
        print(String(format: "     fit: %.2f ms + %.3f ms/branch  (slope = per-added-branch restore+fwd+LM-row)", widthFit.intercept, widthFit.slope))
        print(String(format: "  D) logitsForNextToken readback+materialize : %.3f ms", readbackMs))
        print(String(format: "  E) TokenSampler.rank over full vocab        : %.3f ms", samplerMs))
        print("  -- derived primitives --")
        print(String(format: "     per-decode FIXED floor (weight stream+dispatch+LM head) : %.2f ms", perDecodeFloor))
        print(String(format: "       └─ implied effective bandwidth if pure weight stream : %.0f GiB/s (model %.2f GiB)", impliedBWGiBs, modelGiB))
        print(String(format: "     snapshot restore (per branch seed)                       : %.2f ms", restoreCost))
        print(String(format: "     forward / token  — parallel(prefill) %.3f | small-batch %.3f ms", perTokenForwardParallel, perTokenForwardSmall))
        print(String(format: "     per-added-branch marginal in a batched decode            : %.2f ms", perBranchMarginal))

        // ---- Reconcile against a real depth-4 width-4 completion ----
        let target = AppTarget(bundleIdentifier: "com.apple.TextEdit", appName: "TextEdit", windowTitle: "Untitled")
        let before = "I am writing to let you know that the meeting scheduled for tomorrow "
        let ctx = TextFieldContext(beforeCursor: before, afterCursor: "", target: target, detectedLanguage: "en")
        let promptText = PromptBuilder().buildPrompt(context: ctx).prompt
        let promptTokens = try runtime.tokenizer.tokenize(promptText).count
        let request = CompletionRequest(context: ctx, prompt: promptText, mode: .prose, maxCompletionTokens: 4, maxDisplayWidth: 60)
        await runtime.resetKVCache()
        _ = try await ConstrainedGenerationEngine(runtime: runtime, profile: profile, configuration: config).completions(for: request)
        let real = try await minSeconds(3) {
            await runtime.resetKVCache()
            _ = try await ConstrainedGenerationEngine(runtime: runtime, profile: profile, configuration: config).completions(for: request)
        } * 1000

        // Model: prefill(P) + 3 batched levels (≈ floor + width·(restore+fwd of full suffix)) + sampling.
        let modeledPrefill = prefillFit.intercept + prefillFit.slope * Double(promptTokens)
        let modeledLevels = (1...3).reduce(0.0) { acc, d in acc + perDecodeFloor + 4.0 * (restoreCost + perTokenForwardSmall * Double(d)) }
        let modeledSampling = 13.0 * (readbackMs + samplerMs)
        let modeled = modeledPrefill + modeledLevels + modeledSampling

        print("  -- reconciliation: real depth-4 width-4 completion --")
        print(String(format: "     prompt tokens %d", promptTokens))
        print(String(format: "     measured completion        : %.1f ms", real))
        print(String(format: "     modeled from primitives    : %.1f ms", modeled))
        print(String(format: "       ├─ prefill (1×)          : %.1f ms (%.0f%%)", modeledPrefill, modeledPrefill / modeled * 100))
        print(String(format: "       ├─ 3 batched levels      : %.1f ms (%.0f%%)", modeledLevels, modeledLevels / modeled * 100))
        print(String(format: "       └─ readback+sampling     : %.1f ms (%.0f%%)", modeledSampling, modeledSampling / modeled * 100))
        print("=============================================================================\n")

        await runtime.shutdown()
    }

    /// Sweeps `maxSequences` (n_seq_max) to show that latency is flat across it once it covers the
    /// beam's `branchWidth` (4) — extra sequence slots only reserve recurrent buffers, they are not
    /// a matmul batch dimension, so there is no power-of-two effect. Confirms 4 is the tight optimum.
    func testMaxSequencesSweep() async throws {
        try XCTSkipUnless(ModelContainer.defaultModelExists(), "GGUF missing; skipping")
        let profileURL = try ModelContainer.profileURL(family: Self.family)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: profileURL.path), "profile missing")

        let target = AppTarget(bundleIdentifier: "com.apple.TextEdit", appName: "TextEdit", windowTitle: "Untitled")
        let before = "I am writing to let you know that the meeting scheduled for tomorrow "
        let context = TextFieldContext(beforeCursor: before, afterCursor: "", target: target, detectedLanguage: "en")
        let promptText = PromptBuilder().buildPrompt(context: context).prompt
        let request = CompletionRequest(
            context: context, prompt: promptText, mode: .prose, maxCompletionTokens: 4, maxDisplayWidth: 60
        )
        let config = DecodingConfiguration(maxCandidates: 5, enableFillInMiddle: true) // branchWidth 4

        print("\n================ maxSequences (n_seq_max) sweep — branchWidth=4 ================")
        for maxSeq in [1, 2, 3, 4, 5, 8] {
            let runtime = try LlamaModelRuntime(
                modelURL: try ModelContainer.modelURL(), contextLength: 2048, enableKVFork: true, maxSequences: maxSeq
            )
            let profile = try MmapAutocompleteProfile.open(
                at: profileURL,
                tokenizerVocabSize: runtime.metadata.vocabularySize,
                tokenizerBytes: { try runtime.tokenizer.rawBytes(for: $0) },
                expectedModelFamily: Self.family
            )
            func once() async throws -> (Double, [String]) {
                await runtime.resetKVCache()
                let engine = ConstrainedGenerationEngine(runtime: runtime, profile: profile, configuration: config)
                let start = DispatchTime.now()
                let cands = try await engine.completions(for: request)
                let ms = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
                return (ms, cands.map(\.text))
            }
            _ = try await once() // warm
            var best = Double.greatestFiniteMagnitude
            var cands: [String] = []
            for _ in 0..<3 { let (ms, c) = try await once(); best = min(best, ms); cands = c } // min of 3 (cold each)
            print(String(format: "  maxSequences=%d : %6.1f ms   top=%@", maxSeq, best, cands.first.map { "\"\($0)\"" } ?? "—"))
            await runtime.shutdown()
        }
        print("==============================================================================\n")
    }

    private func seconds(_ block: () async throws -> Void) async rethrows -> Double {
        let start = DispatchTime.now()
        try await block()
        return Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    }

    func testPrefillVsBranchCost() async throws {
        let runtime = try load()
        let target = AppTarget(bundleIdentifier: "com.apple.TextEdit", appName: "TextEdit", windowTitle: "Untitled")
        let before = "I am writing to let you know that the meeting scheduled for tomorrow "
        let prompt = PromptBuilder().buildPrompt(
            context: TextFieldContext(beforeCursor: before, afterCursor: "", target: target, detectedLanguage: "en")
        ).prompt
        let anchor = try runtime.tokenizer.tokenize(prompt)

        // A couple of plausible continuation tokens to feed as branch suffixes.
        let t1 = anchor.last ?? 0
        let t2 = anchor.dropLast().last ?? 0

        // Warm: hot kernels + first prefill.
        await runtime.resetKVCache()
        _ = try await runtime.anchoredLogits(anchor: anchor, suffix: [])

        // 1) Cold full prefill (clear, decode the whole anchor, snapshot, read logits).
        var prefill = 0.0
        let prefillRuns = 5
        for _ in 0..<prefillRuns {
            await runtime.resetKVCache()
            prefill += try await seconds { _ = try await runtime.anchoredLogits(anchor: anchor, suffix: []) }
        }
        prefill /= Double(prefillRuns)

        // Ensure the anchor snapshot is resident for the per-branch measurements below.
        _ = try await runtime.anchoredLogits(anchor: anchor, suffix: [])

        // 2) Cached root (empty suffix): no decode, cached anchor-end logits.
        var root = 0.0
        let rootRuns = 20
        for _ in 0..<rootRuns {
            root += try await seconds { _ = try await runtime.anchoredLogits(anchor: anchor, suffix: []) }
        }
        root /= Double(rootRuns)

        // 3) Per-branch: restore anchor snapshot + decode a 1-token suffix + read logits.
        var branch1 = 0.0
        let branchRuns = 20
        for i in 0..<branchRuns {
            let tok = (i % 2 == 0) ? t1 : t2
            branch1 += try await seconds { _ = try await runtime.anchoredLogits(anchor: anchor, suffix: [tok]) }
        }
        branch1 /= Double(branchRuns)

        // 4) Per-branch with a 3-token suffix (deeper beam level): restore + decode 3 tokens.
        var branch3 = 0.0
        for i in 0..<branchRuns {
            let tok = (i % 2 == 0) ? t1 : t2
            branch3 += try await seconds { _ = try await runtime.anchoredLogits(anchor: anchor, suffix: [tok, t2, t1]) }
        }
        branch3 /= Double(branchRuns)

        // 5) Pure greedy append: decode 1 token with NO restore (the cost a single-branch / greedy
        //    step pays). Isolates llama_decode launch+compute from the snapshot-restore overhead.
        await runtime.resetKVCache()
        _ = try await runtime.anchoredLogits(anchor: anchor, suffix: [])
        var greedy = 0.0
        let greedyRuns = 20
        for i in 0..<greedyRuns {
            let tok = (i % 2 == 0) ? t1 : t2
            greedy += try await seconds {
                try await runtime.decodeNext(tokenID: tok)
                _ = try await runtime.logitsForNextToken()
            }
        }
        greedy /= Double(greedyRuns)

        // Model a depth-4 width-4 beam: 1 prefill + 1 cached root + 4×(1-tok) + 4×(2-tok) + 4×(3-tok).
        // Approximate the 2-tok cost as the midpoint of branch1 and branch3.
        let branch2 = (branch1 + branch3) / 2
        let modeled = prefill + root + 4 * branch1 + 4 * branch2 + 4 * branch3

        print("\n================ prefill vs per-branch micro-bench ================")
        print(String(format: "  anchor tokens                 : %d", anchor.count))
        print(String(format: "  1) cold full prefill          : %7.2f ms", prefill * 1000))
        print(String(format: "  2) cached root (empty suffix) : %7.2f ms", root * 1000))
        print(String(format: "  3) restore + decode 1 token   : %7.2f ms", branch1 * 1000))
        print(String(format: "  4) restore + decode 3 tokens  : %7.2f ms", branch3 * 1000))
        print(String(format: "  5) greedy append 1 tok (no restore): %7.2f ms", greedy * 1000))
        print(String(format: "     → restore overhead alone   : %7.2f ms (branch1 minus greedy)", (branch1 - greedy) * 1000))
        let marginalPerToken: Double = (branch3 - branch1) / 2
        let restoreOverhead: Double = branch1 - marginalPerToken
        let branchShare: Double = 4 * branch1 + 4 * branch2 + 4 * branch3
        print(String(format: "     → marginal cost / token     : %7.2f ms (decode-bound part)", marginalPerToken * 1000))
        print(String(format: "     → restore + fixed overhead  : %7.2f ms (branch1 minus 1 token)", restoreOverhead * 1000))
        print(String(format: "  modeled depth4xwidth4 total   : %7.2f ms", modeled * 1000))
        print(String(format: "     prefill share              : %5.1f%%", prefill / modeled * 100))
        print(String(format: "     12 branch expansions       : %5.1f%%", branchShare / modeled * 100))
        print("==================================================================\n")

        await runtime.shutdown()
    }
}
