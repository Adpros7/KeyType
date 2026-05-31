import AutocompleteCore
import ConstrainedGeneration
import ModelRuntime
import TokenProfiles
import XCTest

/// Deterministic coverage for the engine's fill-in-the-middle prompt assembly and its safe
/// fallback to base continuation. No model required — a recording runtime captures the exact token
/// sequence the engine would decode.

/// Tokenizer that encodes the three FIM markers as single dedicated ids and everything else as its
/// UTF-8 bytes (so token order is trivially checkable).
private struct FIMStubTokenizer: ModelTokenizing {
    static let pre: TokenID = 9001
    static let suf: TokenID = 9002
    static let mid: TokenID = 9003

    func tokenize(_ text: String) throws -> [TokenID] { text.utf8.map { TokenID($0) } }
    func detokenize(_ tokenIDs: [TokenID]) throws -> String {
        String(decoding: tokenIDs.compactMap { UInt8(exactly: $0) }, as: UTF8.self)
    }
    func rawBytes(for tokenID: TokenID) throws -> [UInt8] { UInt8(exactly: tokenID).map { [$0] } ?? [] }
    func tokenizeAllowingSpecial(_ text: String) throws -> [TokenID] {
        switch text {
        case "<|fim_prefix|>": return [Self.pre]
        case "<|fim_suffix|>": return [Self.suf]
        case "<|fim_middle|>": return [Self.mid]
        default: return try tokenize(text)
        }
    }
}

/// Records the first prompt the engine prepares, then returns no logits so generation ends at once.
private final class RecordingRuntime: LocalModelRuntime {
    let metadata: ModelMetadata
    let tokenizer: ModelTokenizing
    private(set) var preparedTokens: [TokenID] = []
    private var recorded = false

    init(tokenizer: ModelTokenizing) {
        self.tokenizer = tokenizer
        self.metadata = ModelMetadata(
            identifier: "recording",
            family: "stub",
            vocabularySize: 70_000,
            contextLength: 4096,
            eosTokenID: nil
        )
    }

    func prepare(promptTokens: [TokenID]) async throws {
        if !recorded { preparedTokens = promptTokens; recorded = true }
    }
    func logitsForNextToken() async throws -> [TokenLogit] { [] }
    func decodeNext(tokenID: TokenID) async throws {}
    func resetKVCache() async {}
}

final class FillInMiddleAssemblyTests: XCTestCase {
    private static let target = AppTarget(bundleIdentifier: "com.test.app", appName: "Test")

    private func prepared(
        tokenizer: ModelTokenizing,
        beforeCursor: String,
        afterCursor: String,
        prompt: String = "SCAFFOLD",
        enableFIM: Bool
    ) async throws -> [TokenID] {
        let runtime = RecordingRuntime(tokenizer: tokenizer)
        let engine = ConstrainedGenerationEngine(
            runtime: runtime,
            profile: InMemoryAutocompleteProfile(vocabularySize: 70_000, records: []),
            configuration: DecodingConfiguration(maxCandidates: 5, enableFillInMiddle: enableFIM)
        )
        _ = try await engine.completions(for: CompletionRequest(
            context: TextFieldContext(beforeCursor: beforeCursor, afterCursor: afterCursor, target: Self.target),
            prompt: prompt,
            mode: .prose,
            maxCompletionTokens: 1,
            maxDisplayWidth: 80
        ))
        return runtime.preparedTokens
    }

    func testAssemblesPrefixSuffixMiddleAndTrimsPrefixTrailingSpace() async throws {
        let tokens = try await prepared(
            tokenizer: FIMStubTokenizer(),
            beforeCursor: "ab ",      // trailing space trimmed → "ab" → [97, 98]
            afterCursor: "cd",        // → [99, 100]
            enableFIM: true
        )
        XCTAssertEqual(
            tokens,
            [FIMStubTokenizer.pre, 97, 98, FIMStubTokenizer.suf, 99, 100, FIMStubTokenizer.mid]
        )
    }

    func testFallsBackToBasePromptWhenDisabled() async throws {
        let tokens = try await prepared(
            tokenizer: FIMStubTokenizer(),
            beforeCursor: "ab ",
            afterCursor: "cd",
            enableFIM: false
        )
        XCTAssertEqual(tokens, Array("SCAFFOLD".utf8).map { TokenID($0) })
    }

    func testFallsBackToBasePromptWhenNoSuffix() async throws {
        let tokens = try await prepared(
            tokenizer: FIMStubTokenizer(),
            beforeCursor: "ab",
            afterCursor: "",
            enableFIM: true
        )
        XCTAssertEqual(tokens, Array("SCAFFOLD".utf8).map { TokenID($0) })
    }

    func testFallsBackWhenModelLacksSingleTokenMarkers() async throws {
        // UTF8FallbackTokenizer has no special handling, so the markers tokenize to many bytes →
        // the engine must not feed angle-bracket text and instead use base continuation.
        let tokens = try await prepared(
            tokenizer: UTF8FallbackTokenizer(),
            beforeCursor: "ab ",
            afterCursor: "cd",
            enableFIM: true
        )
        XCTAssertEqual(tokens, Array("SCAFFOLD".utf8).map { TokenID($0) })
    }
}
