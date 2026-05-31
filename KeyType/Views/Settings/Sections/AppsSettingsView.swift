//
//  AppsSettingsView.swift
//  KeyType
//
//  The "Apps" Settings pane: per-app completion toggles for the currently running apps. Split out of
//  SettingsView so each sidebar category lives in its own file.
//

import AppKit
import SwiftUI

struct AppsSettingsView: View {
    @Bindable var settings: SettingsStore

    @State private var runningApps: [RunningApp] = []

    var body: some View {
        Form {
            Section("Per-app completions") {
                if runningApps.isEmpty {
                    Text("No apps detected.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                ForEach(runningApps) { app in
                    Toggle(app.name, isOn: Binding(
                        get: { settings.isAppEnabled(app.bundleIdentifier) },
                        set: { settings.setApp(app.bundleIdentifier, enabled: $0) }
                    ))
                }
                Button("Refresh app list") { runningApps = Self.loadRunningApps() }
                    .font(.footnote)
            }
        }
        .formStyle(.grouped)
        .task { runningApps = Self.loadRunningApps() }
    }

    private static func loadRunningApps() -> [RunningApp] {
        NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular }
            .compactMap { (app: NSRunningApplication) -> RunningApp? in
                guard let bundleID = app.bundleIdentifier else { return nil }
                return RunningApp(bundleIdentifier: bundleID, name: app.localizedName ?? bundleID)
            }
            .reduce(into: [RunningApp]()) { acc, app in
                if !acc.contains(where: { $0.bundleIdentifier == app.bundleIdentifier }) {
                    acc.append(app)
                }
            }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    struct RunningApp: Identifiable {
        let bundleIdentifier: String
        let name: String
        var id: String { bundleIdentifier }
    }
}
