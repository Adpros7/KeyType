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
