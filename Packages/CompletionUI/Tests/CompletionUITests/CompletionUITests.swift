import AppCompatibility
import AppKit
import AutocompleteCore
import CoreGraphics
import XCTest
@testable import CompletionUI

final class CompletionUITests: XCTestCase {
    private static let target = AppTarget(bundleIdentifier: "com.test.app", appName: "Test")

    private func context(cursorRect: CGRect?, isRTL: Bool = false) -> TextFieldContext {
        TextFieldContext(
            beforeCursor: "hello",
            geometry: TextFieldGeometry(cursorRect: cursorRect, isAtEndOfLine: true, isRightToLeft: isRTL),
            target: Self.target
        )
    }

    // MARK: - Placement resolver

    func testPlacementNilWhenNoCaretRect() {
        let resolver = OverlayPlacementResolver()
        XCTAssertNil(resolver.placement(for: context(cursorRect: nil)))
    }

    func testPlacementCarriesGeometryAndPolicy() {
        let resolver = OverlayPlacementResolver(compatibilityStore: AppCompatibilityStore(overrides: [
            TargetOverride(bundleIdentifier: Self.target.bundleIdentifier, verticalAlignmentOffset: 3)
        ]))
        let rect = CGRect(x: 10, y: 20, width: 1, height: 16)
        let placement = resolver.placement(for: context(cursorRect: rect, isRTL: true))
        XCTAssertEqual(placement?.cursorRect, rect)
        XCTAssertEqual(placement?.isRightToLeft, true)
        XCTAssertEqual(placement?.verticalOffset, 3)
    }

    // MARK: - Noop presenter visible state

    func testNoopPresenterTracksVisibleCandidate() {
        let presenter = NoopCompletionOverlayPresenter()
        XCTAssertNil(presenter.visibleCandidate)

        let candidate = CompletionCandidate(text: " world")
        presenter.show(candidate: candidate, placement: OverlayPlacement(cursorRect: .zero))
        XCTAssertEqual(presenter.visibleCandidate, candidate)

        presenter.hide()
        XCTAssertNil(presenter.visibleCandidate)
    }

    // MARK: - Font resolution

    @MainActor
    func testResolveFontUsesFieldFontScaledByFactor() {
        let field = NSFont.systemFont(ofSize: 12)
        let placement = OverlayPlacement(cursorRect: CGRect(x: 0, y: 0, width: 1, height: 20), fontSizeAdjustmentFactor: 2)
        let resolved = InlineGhostTextPresenter.resolveFont(field, placement: placement)
        XCTAssertEqual(resolved.pointSize, 24, accuracy: 0.01)
    }

    @MainActor
    func testResolveFontFallsBackToCaretHeight() {
        let placement = OverlayPlacement(cursorRect: CGRect(x: 0, y: 0, width: 1, height: 20))
        let resolved = InlineGhostTextPresenter.resolveFont(nil, placement: placement)
        // Estimated from caret height (20 * 0.72 ≈ 14.4), clamped into [8, 48].
        XCTAssertGreaterThanOrEqual(resolved.pointSize, 8)
        XCTAssertLessThanOrEqual(resolved.pointSize, 48)
    }
}
