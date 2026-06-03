//
//  ModelSetupCoordinator.swift
//  KeyType
//

import Foundation
import ModelManagement
import ModelProfileGeneration
import ModelRuntime
import Observation

/// App-local orchestrator for turning a catalog entry or imported GGUF into a ready model.
///
/// `ModelManagement` deliberately keeps downloads and profile generation decoupled. The app layer
/// owns the product-level state: a model is "Ready" only once both the GGUF and its ACPF token
/// profile are present and valid.
@MainActor
@Observable
final class ModelSetupCoordinator {
    enum SetupState: Equatable {
        case idle
        case downloading(progress: Double?)
        case paused(progress: Double?)
        case preparingProfile
        case ready
        case failed(String)
    }

    enum ImportState: Equatable {
        case idle
        case preparing(filename: String)
    }

    let downloads: ModelDownloadManager
    let catalog: [DownloadableRuntimeModel]

    private(set) var profileStates: [String: SetupState] = [:]
    private(set) var importState: ImportState = .idle

    var onModelReady: ((String) -> Void)?
    var onImportFailure: ((String) -> Void)?

    private var profileTasks: [String: Task<Void, Never>] = [:]

    init() {
        self.downloads = ModelDownloadManager()
        self.catalog = self.downloads.catalog
        self.downloads.onGGUFInstalled = { [weak self] model in
            self?.startProfileGeneration(for: model.filename)
        }
        refresh()
    }

    init(downloads: ModelDownloadManager) {
        self.downloads = downloads
        self.catalog = downloads.catalog
        self.downloads.onGGUFInstalled = { [weak self] model in
            self?.startProfileGeneration(for: model.filename)
        }
        refresh()
    }

    func refresh() {
        downloads.refreshStates()
        for model in catalog where isFullyInstalled(model) {
            profileStates[model.filename] = .ready
        }
    }

    func state(for model: DownloadableRuntimeModel) -> SetupState {
        if let profileState = profileStates[model.filename] {
            return profileState
        }

        switch downloads.state(for: model) {
        case .idle:
            return .idle
        case .downloading(let progress):
            return .downloading(progress: progress)
        case .paused(let progress):
            return .paused(progress: progress)
        case .downloaded:
            return isFullyInstalled(model) ? .ready : .idle
        case .failed(let message):
            return .failed(message)
        }
    }

    func isFullyInstalled(_ model: DownloadableRuntimeModel) -> Bool {
        downloads.isInstalled(filename: model.filename) && profileExists(for: model.filename)
    }

    func beginSetup(for model: DownloadableRuntimeModel) {
        profileStates[model.filename] = nil
        if downloads.isInstalled(filename: model.filename) {
            startProfileGeneration(for: model.filename)
        } else {
            downloads.download(model)
        }
    }

    func pause(_ model: DownloadableRuntimeModel) {
        downloads.pause(filename: model.filename)
    }

    func resume(_ model: DownloadableRuntimeModel) {
        downloads.resume(model)
    }

    func cancel(_ model: DownloadableRuntimeModel) {
        profileTasks[model.filename]?.cancel()
        profileTasks[model.filename] = nil
        profileStates[model.filename] = nil
        downloads.cancel(filename: model.filename)
        refresh()
    }

    func importModel(from sourceURL: URL) {
        let filename = sourceURL.lastPathComponent
        importState = .preparing(filename: filename)

        Task { [weak self] in
            guard let self else { return }
            do {
                try await downloads.installLocalModel(from: sourceURL)
                try await generateProfile(for: filename)
                importState = .idle
                onModelReady?(filename)
                refresh()
            } catch {
                importState = .idle
                onImportFailure?(Self.userFacingMessage(for: error))
            }
        }
    }

    private func startProfileGeneration(for filename: String) {
        guard profileTasks[filename] == nil else { return }
        profileStates[filename] = .preparingProfile

        let task = Task { [weak self] in
            guard let self else { return }
            do {
                try await generateProfile(for: filename)
                profileStates[filename] = .ready
                onModelReady?(filename)
            } catch is CancellationError {
                profileStates[filename] = nil
            } catch {
                profileStates[filename] = .failed(Self.userFacingMessage(for: error))
            }
            profileTasks[filename] = nil
        }
        profileTasks[filename] = task
    }

    private func generateProfile(for filename: String) async throws {
        _ = try await ProfileGenerator.generateProfileIfNeeded(forModelFilename: filename)
    }

    private func profileExists(for filename: String) -> Bool {
        let family = RuntimeModelCatalog.model(forFilename: filename)?.tokenizerFamily
        guard let family else { return false }
        guard let url = try? ModelContainer.profileURL(family: family) else { return false }
        return FileManager.default.fileExists(atPath: url.path)
    }

    private static func userFacingMessage(for error: Error) -> String {
        return error.localizedDescription
    }
}
