//
//  DeveloperSettingsView.swift
//  KeyType
//
//  The "Developer" Settings pane: diagnostics that aren't part of the everyday flow. Currently the
//  caret debug overlay (moved here from the menu bar), which draws a marker at the detected caret to
//  verify context capture. Gated on Accessibility, like the capture pipeline it visualizes.
//

import SwiftUI

struct DeveloperSettingsView: View {
    @Bindable var contextCapture: ContextCaptureController
    let permissions: PermissionsManager

    var body: some View {
        Form {
            Section {
                Toggle(isOn: $contextCapture.debugOverlayEnabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Show caret debug overlay")
                        Text("Draws a marker at the detected caret position to verify context capture.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
                .disabled(!permissions.accessibility.isGranted)
            } header: {
                Text("Diagnostics")
            } footer: {
                if !permissions.accessibility.isGranted {
                    Text("Requires Accessibility access.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }
}
