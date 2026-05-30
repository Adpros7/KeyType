//
//  GhostTextOverlayWindow.swift
//  CompletionUI
//
//  The real inline ghost-text overlay (M6). Reuses the proven Red Dot `NSPanel` recipe (the same
//  borderless, non-activating, all-spaces, click-through panel as `CaretDebugOverlayWindow`, see
//  ADR-004 / ADR-006), but hosts dimmed completion text sized to the measured string and pinned to
//  the caret so it reads as a continuation of what the user typed. See ADR-016.
//

import AppKit
import AutocompleteCore
import CoreGraphics
import SwiftUI

@MainActor
public final class GhostTextOverlayWindow {
    private lazy var window: NSPanel = makeWindow()
    private let hosting = NSHostingView(rootView: GhostTextView(text: ""))

    public nonisolated init() {}

    /// Show `text` in `font`, positioned inline at the caret described by `placement`.
    ///
    /// Coordinates are AppKit (bottom-left origin, points). LTR ghost text starts at the caret's
    /// right edge and extends rightward; RTL text ends at the caret's left edge and extends
    /// leftward. The vertical extent matches the caret rect (so the text sits on the same line),
    /// shifted by `placement.verticalOffset`.
    public func show(text: String, font: NSFont, placement: OverlayPlacement) {
        guard !text.isEmpty else { hide(); return }

        hosting.rootView = GhostTextView(text: text, font: font, isRightToLeft: placement.isRightToLeft)

        let caret = placement.cursorRect
        let measuredWidth = ceil((text as NSString).size(withAttributes: [.font: font]).width) + 2
        let height = max(caret.height, ceil(font.ascender - font.descender))

        let x: CGFloat = placement.isRightToLeft
            ? caret.minX - measuredWidth
            : caret.maxX
        let y = caret.minY + (caret.height - height) / 2 - CGFloat(placement.verticalOffset)

        window.setFrame(
            CGRect(x: x, y: y, width: measuredWidth, height: height),
            display: true
        )

        if !window.isVisible {
            window.orderFrontRegardless()
        }
    }

    public func hide() {
        window.orderOut(nil)
    }

    public var isVisible: Bool { window.isVisible }

    private func makeWindow() -> NSPanel {
        let panel = NSPanel(
            contentRect: CGRect(origin: .zero, size: CGSize(width: 1, height: 1)),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        panel.backgroundColor = .clear
        panel.contentView = hosting
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
        panel.hasShadow = false
        panel.hidesOnDeactivate = false
        panel.ignoresMouseEvents = true
        panel.isFloatingPanel = true
        panel.level = .screenSaver
        panel.isOpaque = false
        panel.isReleasedWhenClosed = false

        return panel
    }
}

/// Real `CompletionOverlayPresenting` backed by `GhostTextOverlayWindow`. The app resolves a
/// placement (via `OverlayPlacementResolver`) and the field font, then calls `show`; this presenter
/// owns the borderless panel and keeps it pinned to the caret.
@MainActor
public final class InlineGhostTextPresenter: CompletionOverlayPresenting {
    private let window: GhostTextOverlayWindow
    public private(set) var visibleCandidate: CompletionCandidate?

    public nonisolated init(window: GhostTextOverlayWindow = GhostTextOverlayWindow()) {
        self.window = window
    }

    public func show(candidate: CompletionCandidate, placement: OverlayPlacement, font: NSFont?) {
        let resolved = Self.resolveFont(font, placement: placement)
        window.show(text: candidate.text, font: resolved, placement: placement)
        visibleCandidate = candidate
    }

    public func hide() {
        window.hide()
        visibleCandidate = nil
    }

    public var isVisible: Bool { window.isVisible }

    /// Use the field's font when known (scaled by the per-app adjustment factor); otherwise fall
    /// back to a system font sized from the caret height — a decent proxy for the line's font size.
    static func resolveFont(_ font: NSFont?, placement: OverlayPlacement) -> NSFont {
        let factor = CGFloat(placement.fontSizeAdjustmentFactor)
        if let font {
            let size = max(1, font.pointSize * factor)
            return NSFont(descriptor: font.fontDescriptor, size: size) ?? font
        }
        let estimated = placement.cursorRect.height > 0
            ? placement.cursorRect.height * 0.72
            : NSFont.systemFontSize
        return .systemFont(ofSize: max(8, min(48, estimated * factor)))
    }
}
