//
//  FieldFontResolver.swift
//  KeyType
//
//  Best-effort read of the focused text field's font via Accessibility, so the ghost-text overlay
//  can match the field (M6 / ADR-016). Returns nil whenever the app doesn't surface a font through
//  AX; the overlay presenter then falls back to a system font sized from the caret height.
//

import AppKit
import ApplicationServices

@MainActor
enum FieldFontResolver {
    // AX attributed strings describe their font with these keys (HIServices `AXConstants`), not the
    // AppKit `NSFont` attribute — the value is a dictionary, not an NSFont. We read both forms.
    private static let axFontAttribute = NSAttributedString.Key("AXFont")
    private static let axFontNameKey = "AXFontName" // PostScript name, usable by NSFont(name:size:)
    private static let axFontSizeKey = "AXFontSize"

    /// The font around the insertion point of the system-wide focused element, or nil if it can't
    /// be determined. Reads `AXAttributedStringForRange` over a 1-character probe at the caret.
    static func currentFont() -> NSFont? {
        let systemWide = AXUIElementCreateSystemWide()

        var focused: AnyObject?
        guard AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
              let element = focused, CFGetTypeID(element) == AXUIElementGetTypeID()
        else { return nil }
        let field = element as! AXUIElement

        guard var probe = probeRange(for: field) else { return nil }

        guard let axRange = AXValueCreate(.cfRange, &probe) else { return nil }
        var attributed: AnyObject?
        guard AXUIElementCopyParameterizedAttributeValue(
            field,
            kAXAttributedStringForRangeParameterizedAttribute as CFString,
            axRange,
            &attributed
        ) == .success,
            let string = attributed as? NSAttributedString,
            string.length > 0
        else { return nil }

        // Some apps expose a real NSFont; most expose the AX font dictionary.
        if let nsFont = string.attribute(.font, at: 0, effectiveRange: nil) as? NSFont {
            return nsFont
        }
        if let info = string.attribute(axFontAttribute, at: 0, effectiveRange: nil) as? [String: Any] {
            return font(fromAXFontInfo: info)
        }
        return nil
    }

    /// Build an `NSFont` from the AX font dictionary (`{AXFontName, AXFontSize, …}`).
    private static func font(fromAXFontInfo info: [String: Any]) -> NSFont? {
        let size = (info[axFontSizeKey] as? CGFloat)
            ?? (info[axFontSizeKey] as? Double).map { CGFloat($0) }
            ?? NSFont.systemFontSize
        if let name = info[axFontNameKey] as? String, let font = NSFont(name: name, size: size) {
            return font
        }
        return NSFont.systemFont(ofSize: size)
    }

    /// A length-1 range at (or just before) the caret. AX text APIs return nothing for a zero-length
    /// range, so when the caret is collapsed we probe the preceding character.
    private static func probeRange(for field: AXUIElement) -> CFRange? {
        var rangeValue: AnyObject?
        guard AXUIElementCopyAttributeValue(field, kAXSelectedTextRangeAttribute as CFString, &rangeValue) == .success,
              let value = rangeValue, CFGetTypeID(value) == AXValueGetTypeID()
        else { return nil }

        var selected = CFRange()
        guard AXValueGetValue(value as! AXValue, .cfRange, &selected) else { return nil }

        if selected.length > 0 {
            return CFRange(location: selected.location, length: 1)
        }
        if selected.location > 0 {
            return CFRange(location: selected.location - 1, length: 1)
        }
        return CFRange(location: 0, length: 1)
    }
}
