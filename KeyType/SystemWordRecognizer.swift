//
//  SystemWordRecognizer.swift
//  KeyType
//
//  `WordRecognizing` backed by the macOS system dictionary (`NSSpellChecker`). Feeds the
//  constrained decoder's current-word typo guard (see ADR-015).
//

import AppKit
import AutocompleteCore

/// Recognises words against the system dictionary via `NSSpellChecker`.
///
/// Deliberately conservative so the typo guard never produces a false positive: any word the
/// checker cannot evaluate — an empty string, or a language with no installed dictionary — is
/// reported as *recognised*. `NSSpellChecker` is main-thread affine, so each lookup hops to the
/// main actor; the guard only calls this when a word closes, which is rare relative to per-token
/// decode work.
struct SystemWordRecognizer: WordRecognizing {
    func recognizes(_ word: String, language: String?) async -> Bool {
        guard !word.isEmpty else { return true }
        return await MainActor.run {
            let checker = NSSpellChecker.shared
            let resolved = Self.resolveLanguage(language, checker: checker)
            checker.automaticallyIdentifiesLanguages = (resolved == nil)
            let misspelled = checker.checkSpelling(
                of: word,
                startingAt: 0,
                language: resolved,
                wrap: false,
                inSpellDocumentWithTag: 0,
                wordCount: nil
            )
            // No misspelled range found → the word is recognised.
            return misspelled.location == NSNotFound
        }
    }

    /// Map a detected-language tag (BCP-47 `"en-US"` or NSSpellChecker `"en_US"`) onto an installed
    /// dictionary, falling back to the base language and then to `nil` (auto-detect) so we never
    /// force a checker into a language it can't handle.
    private static func resolveLanguage(_ requested: String?, checker: NSSpellChecker) -> String? {
        guard let requested, !requested.isEmpty else { return nil }
        let normalized = requested.replacingOccurrences(of: "-", with: "_")
        let available = checker.availableLanguages
        if available.contains(normalized) { return normalized }
        let base = String(normalized.prefix { $0 != "_" })
        if available.contains(base) { return base }
        return nil
    }
}
