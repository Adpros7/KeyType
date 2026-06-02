//
//  CaretGeometryQualityTests.swift
//  MacContextCaptureTests
//

import AppKit
import XCTest
@testable import MacContextCapture

final class CaretGeometryQualityTests: XCTestCase {
    func testQualityOrdering() {
        XCTAssertLessThan(AXCaretGeometryQuality.estimated, AXCaretGeometryQuality.derived)
        XCTAssertLessThan(AXCaretGeometryQuality.derived, AXCaretGeometryQuality.exact)
    }

    func testQualityLabels() {
        XCTAssertEqual(AXCaretGeometryQuality.exact.label, "exact")
        XCTAssertEqual(AXCaretGeometryQuality.derived.label, "derived")
        XCTAssertEqual(AXCaretGeometryQuality.estimated.label, "estimated")
    }

    func testGeometryStrategyNamesTheNonInvasivePath() {
        XCTAssertEqual(AXCaretGeometryStrategy.full, .full)
        XCTAssertEqual(AXCaretGeometryStrategy.primary, .primary)
        XCTAssertEqual(AXCaretGeometryStrategy.nonInvasive, .nonInvasive)
    }

    func testNativeMultilineTextUsesPrimaryGeometryForAlignment() {
        XCTAssertEqual(FocusedFieldReader.caretGeometryStrategy(
            isWebField: false,
            role: kAXTextAreaRole as String,
            subrole: nil
        ), .primary)
        XCTAssertEqual(FocusedFieldReader.caretGeometryStrategy(
            isWebField: false,
            role: "AXDocument",
            subrole: nil
        ), .primary)
    }

    func testNativeSingleLineTextUsesNonInvasiveGeometry() {
        XCTAssertEqual(FocusedFieldReader.caretGeometryStrategy(
            isWebField: false,
            role: kAXTextFieldRole as String,
            subrole: nil
        ), .nonInvasive)
    }

    func testWebFieldsKeepFullGeometry() {
        XCTAssertEqual(FocusedFieldReader.caretGeometryStrategy(
            isWebField: true,
            role: kAXTextAreaRole as String,
            subrole: nil
        ), .full)
    }

    func testNativeNonTextFocusedElementsDoNotTriggerDescendantSearch() {
        XCTAssertFalse(FocusedFieldReader.shouldSearchDescendantTextElement(
            rootIsUsable: false,
            rootIsWebContainer: false,
            preferDescendantTextElement: false
        ))
    }

    func testKnownWebBackedFocusedElementsCanSearchForEditableDescendants() {
        XCTAssertTrue(FocusedFieldReader.shouldSearchDescendantTextElement(
            rootIsUsable: false,
            rootIsWebContainer: false,
            preferDescendantTextElement: true
        ))
        XCTAssertTrue(FocusedFieldReader.shouldSearchDescendantTextElement(
            rootIsUsable: true,
            rootIsWebContainer: true,
            preferDescendantTextElement: true
        ))
    }

    func testFieldSizedBoundsAreNotTrustedAsCaretRects() {
        let field = CGRect(x: 80, y: 100, width: 900, height: 120)
        let bogusCaret = field

        XCTAssertTrue(AXCaretGeometryResolver.rectLooksLikeTextContainer(bogusCaret, anchor: field))
    }

    func testLineSizedBoundsAreTrustedAsCaretRects() {
        let field = CGRect(x: 80, y: 100, width: 900, height: 120)
        let lineCaret = CGRect(x: 220, y: 158, width: 2, height: 20)

        XCTAssertFalse(AXCaretGeometryResolver.rectLooksLikeTextContainer(lineCaret, anchor: field))
    }

    func testEstimatedCaretLayoutAccountsForSoftWrappedLines() {
        let font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        let characterWidth = ("a" as NSString).size(withAttributes: [.font: font]).width
        let text = "aaaa aaaa aaaa"

        let layout = AXCaretGeometryResolver.estimatedSoftWrappedCaretLayout(
            in: text,
            selection: NSRange(location: (text as NSString).length, length: 0),
            availableWidth: characterWidth * 10.5,
            font: font,
            widthBias: 1
        )

        XCTAssertEqual(layout.lineIndex, 1)
        XCTAssertEqual(layout.xOffset, characterWidth * 4, accuracy: 0.5)
    }
}
