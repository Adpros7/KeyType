//
//  KeyTypeTests.swift
//  KeyTypeTests
//
//  Created by John Bean on 5/29/26.
//

import AutocompleteCore
import Testing
@testable import KeyType

struct KeyTypeTests {
    private static let target = AppTarget(bundleIdentifier: "com.test.app", appName: "Test")

    @Test func adaptiveDebounceUsesFastPathAfterResponsiveGeneration() {
        #expect(CompletionController.adaptiveDebounceNanoseconds(lastGenerationLatencyMs: 35) == 35_000_000)
    }

    @Test func adaptiveDebounceKeepsConservativeDelayAfterSlowGeneration() {
        #expect(CompletionController.adaptiveDebounceNanoseconds(lastGenerationLatencyMs: 180) == 90_000_000)
    }

    @Test func adaptiveDebounceStartsAtModerateDelayBeforeTelemetry() {
        #expect(CompletionController.adaptiveDebounceNanoseconds(lastGenerationLatencyMs: nil) == 50_000_000)
    }

    @Test func typeThroughAdvanceConsumesTypedPrefixBeforeAXSnapshot() {
        let anchor = TextFieldContext(beforeCursor: "with a 20-core Media", target: Self.target)
        let advanced = CompletionController.typeThroughAdvance(
            anchorText: "Tek designed GPU, and a 10",
            anchorContext: anchor,
            liveContext: anchor,
            typedCharacters: "T"
        )

        #expect(advanced?.context.beforeCursor == "with a 20-core MediaT")
        #expect(advanced?.remainingText == "ek designed GPU, and a 10")
    }

    @Test func typeThroughAdvanceStacksWithoutWaitingForAXSnapshot() {
        let anchor = TextFieldContext(beforeCursor: "with a 20-core Media", target: Self.target)
        let first = CompletionController.typeThroughAdvance(
            anchorText: "Tek designed GPU",
            anchorContext: anchor,
            liveContext: anchor,
            typedCharacters: "T"
        )
        let second = first.flatMap {
            CompletionController.typeThroughAdvance(
                anchorText: "Tek designed GPU",
                anchorContext: anchor,
                liveContext: $0.context,
                typedCharacters: "e"
            )
        }

        #expect(second?.context.beforeCursor == "with a 20-core MediaTe")
        #expect(second?.remainingText == "k designed GPU")
    }

    @Test func typeThroughAdvanceRejectsDivergentTypedText() {
        let anchor = TextFieldContext(beforeCursor: "with a 20-core Media", target: Self.target)
        let advanced = CompletionController.typeThroughAdvance(
            anchorText: "Tek designed GPU",
            anchorContext: anchor,
            liveContext: anchor,
            typedCharacters: "X"
        )

        #expect(advanced == nil)
    }

}
