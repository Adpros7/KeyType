//
//  CompletionAcceptanceController.swift
//  KeyType
//
//  Global Tab / Shift+Tab acceptance hotkey (M6 / ADR-016). A session-level CGEvent tap consumes
//  Tab only while a completion is visible and the app's CompletionPolicy allows Tab acceptance;
//  otherwise every Tab passes straight through so native Tab behaviour is untouched.
//
//  Tab accepts the next word of the suggestion; Shift+Tab accepts the whole suggestion.
//

import AppKit
import CoreGraphics
import os

@MainActor
final class CompletionAcceptanceController {
    /// macOS virtual key code for Tab.
    private static let tabKeyCode: Int64 = 48

    weak var completionController: CompletionController?

    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private let log = Logger(subsystem: "com.pattonium.KeyType", category: "acceptance")

    private(set) var isRunning = false

    func start() {
        guard !isRunning else { return }

        let mask: CGEventMask = (1 << CGEventType.keyDown.rawValue)
        let refcon = Unmanaged.passUnretained(self).toOpaque()

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: mask,
            callback: { _, type, event, refcon in
                guard let refcon else { return Unmanaged.passUnretained(event) }
                let controller = Unmanaged<CompletionAcceptanceController>.fromOpaque(refcon).takeUnretainedValue()
                return MainActor.assumeIsolated {
                    controller.process(type: type, event: event)
                }
            },
            userInfo: refcon
        ) else {
            log.error("Failed to create Tab event tap (Accessibility not granted?)")
            return
        }

        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)

        eventTap = tap
        runLoopSource = source
        isRunning = true
        log.debug("Tab acceptance tap installed")
    }

    func stop() {
        guard isRunning else { return }
        if let eventTap {
            CGEvent.tapEnable(tap: eventTap, enable: false)
        }
        if let runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
        }
        eventTap = nil
        runLoopSource = nil
        isRunning = false
    }

    /// Decide whether to consume (return nil) or pass through (return the event). Runs on the main
    /// run loop, so `MainActor.assumeIsolated` at the call site is valid.
    private func process(type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        // The system disables a tap that times out or is interrupted; re-enable and pass through.
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            if let eventTap { CGEvent.tapEnable(tap: eventTap, enable: true) }
            return Unmanaged.passUnretained(event)
        }

        guard type == .keyDown else { return Unmanaged.passUnretained(event) }
        guard event.getIntegerValueField(.keyboardEventKeycode) == Self.tabKeyCode else {
            return Unmanaged.passUnretained(event)
        }

        let flags = event.flags
        // Leave command/control/option-Tab (app switching, etc.) to the system.
        if flags.contains(.maskCommand) || flags.contains(.maskControl) || flags.contains(.maskAlternate) {
            return Unmanaged.passUnretained(event)
        }

        guard let controller = completionController, controller.canAcceptCompletion else {
            return Unmanaged.passUnretained(event) // nothing to accept → native Tab
        }

        if flags.contains(.maskShift) {
            controller.acceptFullCompletion()
        } else {
            controller.acceptNextWord()
        }
        return nil // consume — Tab accepted the completion instead of inserting a tab
    }
}
