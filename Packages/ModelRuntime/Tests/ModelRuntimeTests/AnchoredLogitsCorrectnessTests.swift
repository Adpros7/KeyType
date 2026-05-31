import AutocompleteCore
import LlamaModelRuntime
import ModelRuntime
import XCTest

/// Gates the KV-fork optimization (ADR-018): `anchoredLogits` must produce the SAME next-token
/// distribution as a from-scratch decode of `anchor + suffix`. If `llama_memory_seq_cp` were
/// incorrect on this model's hybrid memory, the forked logits would diverge here and we'd fall back
/// to snapshot/restore before shipping. On-device only (skips without the GGUF).
final class AnchoredLogitsCorrectnessTests: XCTestCase {
    private func makeRuntime(enableKVFork: Bool) throws -> LlamaModelRuntime {
        try XCTSkipUnless(
            ModelContainer.defaultModelExists(),
            "Model file not present at \(ModelContainer.defaultModelFilename); skipping"
        )
        return try LlamaModelRuntime(
            modelURL: try ModelContainer.modelURL(),
            contextLength: 2048,
            reuseThreshold: 8,
            enableKVFork: enableKVFork
        )
    }

    /// Ground truth: clear + full decode of `anchor + suffix`, returning the raw next-token logits.
    private func groundTruthLogits(_ runtime: LlamaModelRuntime, anchor: [TokenID], suffix: [TokenID]) async throws -> [TokenLogit] {
        await runtime.resetKVCache()
        try await runtime.prepare(promptTokens: anchor + suffix)
        return try await runtime.logitsForNextToken()
    }

    private func topK(_ logits: [TokenLogit], _ k: Int) -> [TokenID] {
        logits.sorted { $0.logit > $1.logit }.prefix(k).map(\.tokenID)
    }

    private func argmax(_ logits: [TokenLogit]) -> TokenID? {
        logits.max { $0.logit < $1.logit }?.tokenID
    }

    /// The project's documented correctness envelope for KV-reuse / batched decode on this hybrid
    /// recurrent model (ADR-012/018/043): the **argmax is identical** and the **top-k set is
    /// identical**. Only the order of near-tied tokens at ranks 3+ may shuffle (≤~0.12 logit drift
    /// from the parallel/split recurrent path), which never changes the displayed (top) candidate.
    private func assertSameDistribution(
        _ got: [TokenLogit], _ expected: [TokenLogit], k: Int = 5,
        _ message: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(argmax(got), argmax(expected), "argmax diverged: \(message)", file: file, line: line)
        XCTAssertEqual(
            Set(topK(got, k)), Set(topK(expected, k)),
            "top-\(k) set diverged: \(message)", file: file, line: line
        )
    }

    func testForkedLogitsMatchFullDecode() async throws {
        let runtime = try makeRuntime(enableKVFork: true)
        let tok = runtime.tokenizer
        let anchorText = "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is"
        let anchor = try tok.tokenize(anchorText)

        let suffixes: [[TokenID]] = [
            [],                                   // root branch (empty suffix)
            try tok.tokenize(" Mad"),
            try tok.tokenize(" the"),
            try tok.tokenize(" a beautiful")
        ]

        for suffix in suffixes {
            // Ground truth uses a *separate* runtime so it can't accidentally benefit from resident
            // state left by the fork path.
            let truthRuntime = try makeRuntime(enableKVFork: true)
            let expected = try await groundTruthLogits(truthRuntime, anchor: anchor, suffix: suffix)
            let forked = try await runtime.anchoredLogits(anchor: anchor, suffix: suffix)
            assertSameDistribution(
                forked, expected,
                "forked snapshot/restore vs full decode for suffix \(suffix)"
            )
        }
    }

    /// The fork path must keep the anchor resident so repeated branches (and cross-keystroke
    /// appends) reuse it: after a fresh anchor decode, each forked branch should decode only its own
    /// suffix, and an anchor that extends the resident one should decode only the appended tokens.
    func testAnchorResidencyDecodesOnlySuffixAndTypedDelta() async throws {
        let runtime = try makeRuntime(enableKVFork: true)
        let tok = runtime.tokenizer
        let anchor = try tok.tokenize("I am writing to let you know that the meeting tomorrow")
        await runtime.resetKVCache()

        // First branch establishes the anchor (full decode) ...
        _ = try await runtime.anchoredLogits(anchor: anchor, suffix: try tok.tokenize(" is"))
        let afterAnchor = await runtime.lastPrepareDecodedCount
        XCTAssertEqual(afterAnchor, anchor.count, "first call should decode the whole anchor once")

        // ... subsequent branches with the SAME anchor must not re-decode it (ensureResident = 0).
        _ = try await runtime.anchoredLogits(anchor: anchor, suffix: try tok.tokenize(" has"))
        let afterSameAnchor = await runtime.lastPrepareDecodedCount
        XCTAssertEqual(afterSameAnchor, 0, "same anchor must stay resident across branches")

        // Cross-keystroke: anchor grows by the newly typed tokens; only those are decoded.
        let typed = try tok.tokenize(" at")
        let grown = anchor + typed
        _ = try await runtime.anchoredLogits(anchor: grown, suffix: [])
        let afterGrowth = await runtime.lastPrepareDecodedCount
        XCTAssertEqual(afterGrowth, typed.count, "only the typed delta should be decoded")
    }

    /// Gates the batched beam-frontier expansion (ADR-043): `anchoredLogitsBatch` must produce the
    /// SAME next-token distribution for each branch as scoring that branch on its own with
    /// `anchoredLogits`. If multi-sequence seeding or batched decode diverged on this model's hybrid
    /// memory, the top-k would differ here and we'd fall back to the per-branch path.
    func testBatchedFrontierMatchesPerBranch() async throws {
        let runtime = try makeRuntime(enableKVFork: true)
        let tok = runtime.tokenizer
        let anchorText = "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is"
        let anchor = try tok.tokenize(anchorText)

        let suffixes: [[TokenID]] = [
            [],                              // root branch (cached anchor-end logits)
            try tok.tokenize(" Mad"),
            try tok.tokenize(" the largest"),
            try tok.tokenize(" a"),
            try tok.tokenize(" Barcelona and")
        ]

        // Per-branch ground truth from the single-branch path (itself gated against full decode).
        var perBranch: [[TokenLogit]] = []
        for suffix in suffixes {
            perBranch.append(try await runtime.anchoredLogits(anchor: anchor, suffix: suffix))
        }

        // One batched call must reproduce every branch's distribution, in input order.
        let batched = try await runtime.anchoredLogitsBatch(anchor: anchor, suffixes: suffixes)
        XCTAssertEqual(batched.count, suffixes.count)
        for (i, logits) in batched.enumerated() {
            assertSameDistribution(logits, perBranch[i], "batched branch \(i) vs per-branch for suffix \(suffixes[i])")
        }
    }

    /// A frontier wider than `n_seq_max` must still be correct: the runtime chunks it into multiple
    /// batched decodes, and every branch's logits must match the per-branch path regardless of which
    /// chunk it landed in. `maxSequences: 2` forces chunking with a 5-branch frontier.
    func testBatchedFrontierChunksBeyondSeqMax() async throws {
        try XCTSkipUnless(ModelContainer.defaultModelExists(), "Model file not present; skipping")
        let runtime = try LlamaModelRuntime(
            modelURL: try ModelContainer.modelURL(), contextLength: 2048, enableKVFork: true, maxSequences: 2
        )
        let tok = runtime.tokenizer
        let anchor = try tok.tokenize("I am writing to let you know that the meeting tomorrow")
        let suffixes: [[TokenID]] = [
            try tok.tokenize(" is"), try tok.tokenize(" has"), try tok.tokenize(" will"),
            try tok.tokenize(" at"), try tok.tokenize(" might be")
        ]

        var perBranch: [[TokenLogit]] = []
        for suffix in suffixes {
            perBranch.append(try await runtime.anchoredLogits(anchor: anchor, suffix: suffix))
        }
        let batched = try await runtime.anchoredLogitsBatch(anchor: anchor, suffixes: suffixes)
        for (i, logits) in batched.enumerated() {
            assertSameDistribution(logits, perBranch[i], "chunked batch branch \(i)")
        }
    }

    /// Disabling the flag falls back to the default full-decode path and must still be correct.
    func testDisabledForkMatchesFullDecode() async throws {
        let runtime = try makeRuntime(enableKVFork: false)
        let tok = runtime.tokenizer
        let anchor = try tok.tokenize("Once upon a time there was a")
        let suffix = try tok.tokenize(" small")

        let truthRuntime = try makeRuntime(enableKVFork: false)
        let expected = try await groundTruthLogits(truthRuntime, anchor: anchor, suffix: suffix)
        let got = try await runtime.anchoredLogits(anchor: anchor, suffix: suffix)
        assertSameDistribution(got, expected, "fork-disabled vs full decode")
    }
}
