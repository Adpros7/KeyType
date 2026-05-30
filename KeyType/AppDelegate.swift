//
//  AppDelegate.swift
//  KeyType
//
//  Created by Codex on 5/29/26.
//

import AppKit
import MacContextCapture
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    static let onboardingWindowID = "onboarding"
    private static let hasCompletedOnboardingDefaultsKey = "KeyType.hasCompletedOnboarding"

    let permissions = PermissionsManager()
    // One AX tracker feeds both the (debug) context capture and the live completion pipeline.
    private let tracker: AccessibilityContextTracker
    let contextCapture: ContextCaptureController
    let completion: CompletionController
    private let acceptance = CompletionAcceptanceController()
    private var permissionSyncTimer: Timer?
    /// Set once the user has confirmed quitting and the async model teardown is under way, so the
    /// confirmation alert isn't shown twice and `applicationShouldTerminate` doesn't re-prompt.
    private var isTerminating = false

    override init() {
        let tracker = AccessibilityContextTracker()
        self.tracker = tracker
        self.contextCapture = ContextCaptureController(tracker: tracker)
        self.completion = CompletionController(tracker: tracker)
        super.init()
        acceptance.completionController = completion
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Background / agent app: no dock icon. LSUIElement in Info.plist already suppresses the
        // dock icon; making the activation policy explicit guards against alternate launch paths.
        NSApp.setActivationPolicy(.accessory)

        permissions.startMonitoring()
        syncContextCaptureWithPermission()
        startObservingPermissionChanges()

        if shouldShowOnboardingOnLaunch {
            // The SwiftUI scene observes this and calls `openWindow(id:)` for us.
            requestOpenOnboarding()
        }
    }

    /// Start/stop the context tracker so it only runs when AX is actually granted. We poll the
    /// `PermissionsManager` (which itself polls AX status at 1 Hz) once per second; this is a
    /// background, low-frequency check — the tracker itself reacts to AX notifications.
    private func startObservingPermissionChanges() {
        let timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.syncContextCaptureWithPermission()
            }
        }
        timer.tolerance = 0.5
        RunLoop.main.add(timer, forMode: .common)
        permissionSyncTimer = timer
    }

    private func syncContextCaptureWithPermission() {
        if permissions.accessibility.isGranted {
            contextCapture.start()
            completion.start()
            acceptance.start()
        } else {
            contextCapture.stop()
            completion.stop()
            acceptance.stop()
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // Keep running as a menu-bar agent even after the onboarding window is dismissed.
        false
    }

    /// Gate every quit path (menu item, ⌘Q) behind a confirmation, then tear the model down before
    /// exiting. The teardown is mandatory, not just polite: llama.cpp's ggml-metal backend aborts in
    /// its process-exit C++ destructors unless the llama context/model were freed first (the GPU
    /// residency-set assert in the crash report). We free them asynchronously, then let termination
    /// proceed via `reply(toApplicationShouldTerminate:)`. See ADR-021.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if isTerminating { return .terminateNow }

        // The agent has no dock icon, so bring the alert to the front explicitly.
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = "Quit KeyType?"
        alert.informativeText = "KeyType will stop suggesting completions until you open it again."
        alert.alertStyle = .warning
        // First button is the default (highlighted, triggered by Return) and sits on the right.
        alert.addButton(withTitle: "Quit")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else {
            return .terminateCancel
        }

        isTerminating = true
        Task { @MainActor in
            await completion.shutdown()
            NSApp.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    func requestOpenOnboarding() {
        NSApp.activate(ignoringOtherApps: true)
        NotificationCenter.default.post(name: .keyTypeShouldOpenOnboarding, object: nil)
    }

    func markOnboardingCompleted() {
        UserDefaults.standard.set(true, forKey: Self.hasCompletedOnboardingDefaultsKey)
    }

    private var shouldShowOnboardingOnLaunch: Bool {
        let defaults = UserDefaults.standard
        let completed = defaults.bool(forKey: Self.hasCompletedOnboardingDefaultsKey)
        // Always show on first run, or whenever Accessibility hasn't been granted yet.
        return !completed || !permissions.accessibility.isGranted
    }
}

extension Notification.Name {
    static let keyTypeShouldOpenOnboarding = Notification.Name("KeyType.shouldOpenOnboarding")
}
