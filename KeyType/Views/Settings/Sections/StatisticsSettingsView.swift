//
//  StatisticsSettingsView.swift
//  KeyType
//
//  The "Statistics" Settings pane: read-only local telemetry (acceptance/suppression/latency). Split
//  out of SettingsView so each sidebar category lives in its own file.
//

import Personalization
import SwiftUI

struct StatisticsSettingsView: View {
    let telemetry: CompletionTelemetryStore

    @State private var snapshot: TelemetrySnapshot = TelemetrySnapshot()

    var body: some View {
        Form {
            Section("Local stats") {
                statRow("Acceptance rate", percent(snapshot.acceptanceRate),
                        detail: "\(snapshot.acceptedCount) accepted / \(snapshot.shownCount) shown")
                statRow("Suppression rate", percent(snapshot.suppressionRate),
                        detail: "\(snapshot.suppressedCount) of \(snapshot.generatedCount) generated")
                statRow("Latency (p50 / p95)",
                        "\(Int(snapshot.latencyMillisP50)) / \(Int(snapshot.latencyMillisP95)) ms",
                        detail: "\(snapshot.latencySampleCount) samples")
                Button("Refresh stats") { snapshot = telemetry.snapshot() }
                    .font(.footnote)
            }
        }
        .formStyle(.grouped)
        .task { snapshot = telemetry.snapshot() }
    }

    private func statRow(_ title: String, _ value: String, detail: String) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                Text(detail).font(.footnote).foregroundStyle(.secondary)
            }
            Spacer()
            Text(value).font(.body.monospacedDigit()).foregroundStyle(.secondary)
        }
    }

    private func percent(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100)
    }
}
