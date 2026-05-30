import AutocompleteCore
import XCTest

final class NextWordSplitterTests: XCTestCase {
    func testEmpty() {
        let (head, rest) = NextWordSplitter.split("")
        XCTAssertEqual(head, "")
        XCTAssertEqual(rest, "")
    }

    func testSingleWordAcceptsWholesale() {
        let (head, rest) = NextWordSplitter.split("tomorrow")
        XCTAssertEqual(head, "tomorrow")
        XCTAssertEqual(rest, "")
    }

    func testMidWordCompletionWithoutLeadingSpace() {
        let (head, rest) = NextWordSplitter.split("orrow to talk")
        XCTAssertEqual(head, "orrow ")
        XCTAssertEqual(rest, "to talk")
    }

    func testLeadingWhitespaceTravelsWithFirstWord() {
        let (head, rest) = NextWordSplitter.split(" world today")
        XCTAssertEqual(head, " world ")
        XCTAssertEqual(rest, "today")
    }

    func testRepeatedSplitWalksTheSuggestion() {
        var remaining = " quick brown fox"
        var accepted = ""
        while !remaining.isEmpty {
            let (head, rest) = NextWordSplitter.split(remaining)
            accepted += head
            remaining = rest
        }
        XCTAssertEqual(accepted, " quick brown fox")
    }

    func testChineseIsSegmentedNotTakenWholesale() {
        // ICU segments Chinese into words, so the first Tab should not swallow the whole string.
        let (head, rest) = NextWordSplitter.split("今天天气很好")
        XCTAssertFalse(head.isEmpty)
        XCTAssertFalse(rest.isEmpty)
        XCTAssertEqual(head + rest, "今天天气很好")
    }
}
