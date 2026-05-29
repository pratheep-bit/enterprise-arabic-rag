"""
arabic_processor.py — Arabic Text Processing Pipeline

Handles Unicode normalization, Kashida removal, diacritics management,
whitespace cleanup, and RTL marker sanitization for Arabic text.
"""

import re
import unicodedata
from typing import Optional


# ============================================================
# Arabic Unicode Character Ranges & Constants
# ============================================================

# Kashida (Tatweel) — decorative elongation character
KASHIDA = "\u0640"

# Arabic diacritical marks (Tashkeel / Harakat)
ARABIC_DIACRITICS = re.compile(
    "["
    "\u0610-\u061A"  # Arabic signs (e.g., small high ligatures)
    "\u064B-\u065F"  # Arabic Fathatan through Wavy Hamza Below
    "\u0670"         # Arabic Letter Superscript Alef
    "\u06D6-\u06DC"  # Arabic small high ligatures for Quran
    "\u06DF-\u06E4"  # Arabic small high signs
    "\u06E7-\u06E8"  # Arabic small high yeh/noon
    "\u06EA-\u06ED"  # Arabic small low/high signs
    "]+"
)

# RTL/LTR directional markers
DIRECTIONAL_MARKERS = re.compile(
    "["
    "\u200E"  # LTR mark
    "\u200F"  # RTL mark
    "\u200B"  # Zero-width space
    "\u200C"  # Zero-width non-joiner
    "\u200D"  # Zero-width joiner
    "\u202A"  # LTR embedding
    "\u202B"  # RTL embedding
    "\u202C"  # Pop directional formatting
    "\u202D"  # LTR override
    "\u202E"  # RTL override
    "\u2066"  # LTR isolate
    "\u2067"  # RTL isolate
    "\u2068"  # First strong isolate
    "\u2069"  # Pop directional isolate
    "\uFEFF"  # BOM / zero-width no-break space
    "]+"
)

# Common Arabic character normalizations (Alef variants → base Alef)
ALEF_VARIANTS = {
    "\u0622": "\u0627",  # Alef with Madda → Alef
    "\u0623": "\u0627",  # Alef with Hamza Above → Alef
    "\u0625": "\u0627",  # Alef with Hamza Below → Alef
    "\u0671": "\u0627",  # Alef Wasla → Alef
}

# Teh Marbuta → Heh normalization (optional, context-dependent)
TEH_MARBUTA = "\u0629"
HEH = "\u0647"


class ArabicProcessor:
    """
    Production-grade Arabic text processor for RAG pipelines.
    
    Supports:
    - Unicode NFC normalization
    - Kashida (Tatweel) removal
    - Diacritics stripping (optional)
    - Alef normalization
    - RTL marker cleanup
    - Whitespace normalization
    """

    def __init__(
        self,
        remove_diacritics: bool = True,
        remove_kashida: bool = True,
        normalize_alef: bool = True,
        normalize_teh_marbuta: bool = False,
        remove_directional_markers: bool = True,
    ):
        """
        Initialize the Arabic text processor.

        Args:
            remove_diacritics: Strip Arabic diacritical marks (Tashkeel).
            remove_kashida: Remove Kashida/Tatweel characters.
            normalize_alef: Normalize Alef variants to base Alef.
            normalize_teh_marbuta: Convert Teh Marbuta to Heh (use with caution).
            remove_directional_markers: Strip RTL/LTR Unicode markers.
        """
        self.remove_diacritics = remove_diacritics
        self.remove_kashida = remove_kashida
        self.normalize_alef = normalize_alef
        self.normalize_teh_marbuta = normalize_teh_marbuta
        self.remove_directional_markers = remove_directional_markers

    def process(self, text: Optional[str]) -> str:
        """
        Apply the full Arabic text processing pipeline.

        Args:
            text: Raw Arabic text to process.

        Returns:
            Cleaned and normalized Arabic text.
        """
        if not text:
            return ""

        # Step 1: Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        # Step 2: Remove directional markers
        if self.remove_directional_markers:
            text = DIRECTIONAL_MARKERS.sub("", text)

        # Step 3: Remove Kashida
        if self.remove_kashida:
            text = text.replace(KASHIDA, "")

        # Step 4: Remove diacritics (Tashkeel)
        if self.remove_diacritics:
            text = ARABIC_DIACRITICS.sub("", text)

        # Step 5: Normalize Alef variants
        if self.normalize_alef:
            for variant, base in ALEF_VARIANTS.items():
                text = text.replace(variant, base)

        # Step 6: Normalize Teh Marbuta (optional)
        if self.normalize_teh_marbuta:
            text = text.replace(TEH_MARBUTA, HEH)

        # Step 7: Normalize whitespace
        text = self._normalize_whitespace(text)

        return text.strip()

    def process_for_embedding(self, text: Optional[str]) -> str:
        """
        Process text specifically for embedding generation.
        Applies all normalizations to maximize semantic matching.

        Args:
            text: Raw text to process for embedding.

        Returns:
            Normalized text optimized for embedding similarity.
        """
        # For embeddings, we always strip diacritics and normalize aggressively
        original_settings = (self.remove_diacritics, self.normalize_alef)
        self.remove_diacritics = True
        self.normalize_alef = True

        result = self.process(text)

        # Restore original settings
        self.remove_diacritics, self.normalize_alef = original_settings
        return result

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace: collapse multiple spaces/tabs into single space,
        preserve paragraph breaks (double newlines), normalize single newlines.
        """
        # Normalize different types of whitespace characters
        text = text.replace("\t", " ")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Preserve paragraph breaks (2+ newlines → double newline)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces into one (within lines)
        text = re.sub(r"[^\S\n]+", " ", text)

        # Remove spaces at start/end of lines
        text = re.sub(r" *\n *", "\n", text)

        return text

    @staticmethod
    def is_arabic(text: str) -> bool:
        """
        Check if text contains significant Arabic content.

        Args:
            text: Text to check.

        Returns:
            True if the text contains Arabic characters.
        """
        if not text:
            return False
        arabic_char_count = sum(
            1 for char in text if "\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F"
        )
        return arabic_char_count > len(text) * 0.1  # At least 10% Arabic characters

    @staticmethod
    def get_text_direction(text: str) -> str:
        """
        Determine the primary text direction.

        Returns:
            'rtl' if primarily Arabic, 'ltr' otherwise.
        """
        if ArabicProcessor.is_arabic(text):
            return "rtl"
        return "ltr"


# ============================================================
# Module-level convenience function
# ============================================================

# Default processor instance (singleton)
_default_processor = ArabicProcessor()


def normalize_arabic(text: Optional[str]) -> str:
    """
    Convenience function: normalize Arabic text with default settings.

    Args:
        text: Raw Arabic text.

    Returns:
        Normalized Arabic text.
    """
    return _default_processor.process(text)


def normalize_for_embedding(text: Optional[str]) -> str:
    """
    Convenience function: normalize text for embedding generation.

    Args:
        text: Raw text to normalize for embeddings.

    Returns:
        Text normalized for embedding similarity.
    """
    return _default_processor.process_for_embedding(text)
