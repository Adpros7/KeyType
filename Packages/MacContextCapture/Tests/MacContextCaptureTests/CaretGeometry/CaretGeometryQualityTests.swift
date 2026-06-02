//
//  CaretGeometryQualityTests.swift
//  MacContextCaptureTests
//

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
}
