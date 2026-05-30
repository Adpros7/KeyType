import Foundation

/// Splits a completion string into the "next word" to accept (Tab) and the remainder, using ICU
/// word boundaries so it works across scripts — space-delimited (Latin/Cyrillic/…) *and*
/// space-less ones the system segments (CJK, Thai). See ADR-016.
///
/// "Next word" is everything from the start of the string up to where the *second* word begins, so
/// leading whitespace and the word's own trailing whitespace/punctuation travel with it (accepting
/// `" world today"` inserts `" world "` and leaves `"today"`; accepting `"orrow to talk"` inserts
/// `"orrow "` and leaves `"to talk"`). A string with one or zero words accepts wholesale.
public enum NextWordSplitter {
    /// `head` is the slice to insert on a single Tab; `rest` is what remains for the next Tab.
    public static func split(_ text: String) -> (head: String, rest: String) {
        guard !text.isEmpty else { return ("", "") }

        var wordStarts: [String.Index] = []
        text.enumerateSubstrings(in: text.startIndex..<text.endIndex, options: [.byWords]) { _, range, _, stop in
            wordStarts.append(range.lowerBound)
            if wordStarts.count >= 2 { stop = true }
        }

        guard wordStarts.count >= 2 else { return (text, "") }
        let secondStart = wordStarts[1]
        return (String(text[..<secondStart]), String(text[secondStart...]))
    }
}
