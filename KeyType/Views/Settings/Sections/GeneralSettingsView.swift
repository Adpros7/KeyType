//
//  GeneralSettingsView.swift
//  KeyType
//
//  The "General" Settings pane: startup, completion length, and global writing instructions. Split
//  out of SettingsView so each sidebar category lives in its own file.
//

import LaunchAtLogin
import SwiftUI

struct GeneralSettingsView: View {
    @Bindable var settings: SettingsStore

    var body: some View {
        Form {
            Section("Startup") {
                LaunchAtLogin.Toggle()
                Text("Start KeyType automatically when you log in to your Mac.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Section("Completion length") {
                Picker("Length", selection: $settings.completionLength) {
                    ForEach(CompletionLength.allCases) { length in
                        Text(length.title).tag(length)
                    }
                }
                .pickerStyle(.segmented)
                Text("Shorter completions are more conservative; longer ones suggest more at once.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Section("Writing instructions") {
                TextEditor(text: $settings.customInstructions)
                    .font(.body)
                    .frame(minHeight: 88)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.secondary.opacity(0.2))
                    )

                HStack {
                    Text("Added to every local prompt before app-specific instructions.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Clear") {
                        settings.customInstructions = ""
                    }
                    .disabled(settings.customInstructions.isEmpty)
                }
            }
        }
        .formStyle(.grouped)
    }
}
