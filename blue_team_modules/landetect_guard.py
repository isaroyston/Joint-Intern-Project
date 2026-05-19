"""Standalone language detection / non-English blocking guard."""

from typing import FrozenSet

try:
    from langdetect import DetectorFactory, detect_langs
except ModuleNotFoundError:
    DetectorFactory = None
    detect_langs = None


if DetectorFactory is not None:
    DetectorFactory.seed = 0

# A small whitelist of non-ASCII code points that legitimately appear in
# English input (smart quotes, dashes, ellipsis, NBSP, common currency).
# Everything else outside printable ASCII is rejected.
ALLOWED_NON_ASCII_CODEPOINTS: FrozenSet[int] = frozenset({
    0x00A0,                                  # NBSP
    0x2018, 0x2019, 0x201A, 0x201B,          # ‘ ’ ‚ ‛
    0x201C, 0x201D, 0x201E, 0x201F,          # “ ” „ ‟
    0x2013, 0x2014,                          # – —
    0x2026,                                  # …
    0x00A3, 0x00A5, 0x20AC,                  # £ ¥ €
})


class LanguageDetectGuard:
    """Two-stage non-English detection (char filter + optional langdetect)."""

    def __init__(self, min_chars_for_lang_check: int = 30, min_english_prob: float = 0.85):
        self.min_chars_for_lang_check = min_chars_for_lang_check
        self.min_english_prob = min_english_prob

    def block_non_english(self, text: str, enforce_langdetect: bool = False) -> bool:
        """Return True if `text` should be rejected as non-English.

        Two-stage defence designed to defeat prompt-injection via foreign scripts:

        1. Per-character whitelist. ANY code point outside printable ASCII +
            the small typography/currency whitelist trips the block. This
            catches CJK, Cyrillic, Arabic, Greek, Devanagari, Hebrew, Thai,
            emoji, zero-width chars, and homoglyph attacks (e.g. Cyrillic 'а'
            U+0430 used to spell "bаlance"). A single foreign char is enough,
            so mixed English+Chinese like "balance 忽略指令" is rejected.

        2. Latin-script language check. ASCII-only text can still be Spanish,
            French, Indonesian, etc. For long-enough input we run langdetect
            and require high-confidence English. We skip this for short input
            because langdetect is noisy on snippets like "ok" or "thanks".
        """        
        t = (text or "").strip()
        if not t:
            return False

        for ch in t:
            cp = ord(ch)
            if cp in (0x09, 0x0A, 0x0D) or 0x20 <= cp <= 0x7E:
                continue
            if cp in ALLOWED_NON_ASCII_CODEPOINTS:
                continue
            return True

        if not enforce_langdetect:
            return False

        if detect_langs is None:
            return False

        # Stage 2 needs enough signal — under 30 chars langdetect is unreliable.
        if len(t) < self.min_chars_for_lang_check:
            return False

        try:
            ranked = detect_langs(t)
        except Exception:
            return True

        if not ranked:
            return True

        top = ranked[0]
        return top.lang != "en" or top.prob < self.min_english_prob


_default_guard = LanguageDetectGuard()


def block_non_english(text: str, enforce_langdetect: bool = False) -> bool:
    """Drop-in helper for existing integrations."""
    return _default_guard.block_non_english(text, enforce_langdetect=enforce_langdetect)
