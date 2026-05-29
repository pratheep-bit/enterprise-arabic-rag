"""
dialect_mapper.py — Gulf Arabic Dialect Normalization

Maps Gulf Arabic dialect terms to Modern Standard Arabic (MSA) equivalents,
providing query expansion for improved semantic retrieval.
"""

import re
from typing import Optional


# ============================================================
# Gulf Dialect → MSA Mapping Table
# ============================================================
# This mapping covers common Gulf Arabic colloquialisms used
# in UAE, Saudi Arabia, Kuwait, Bahrain, Qatar, and Oman.

DIALECT_TO_MSA_MAP: dict[str, str] = {
    # ==========================================================
    # Gulf Arabic (UAE, Saudi, Kuwait, Bahrain, Qatar, Oman)
    # ==========================================================

    # --- Question words ---
    "شلون": "كيف",           # How (Gulf) → How (MSA)
    "شلونك": "كيف حالك",    # How are you
    "ويش": "ماذا",           # What (Gulf) → What (MSA)
    "إيش": "ماذا",           # What (variant)
    "شنو": "ماذا",           # What (Iraqi/Gulf)
    "ليش": "لماذا",          # Why (Gulf) → Why (MSA)
    "ليه": "لماذا",          # Why (variant)
    "وين": "أين",            # Where (Gulf) → Where (MSA)
    "متى": "متى",            # When (same in both)
    "منو": "من",             # Who (Gulf) → Who (MSA)
    "شقد": "كم",             # How much/many (Gulf)
    "جم": "كم",              # How much/many (variant)

    # --- Verbs ---
    "أبي": "أريد",            # I want (Gulf) → I want (MSA)
    "أبغى": "أريد",          # I want (variant)
    "أبا": "أريد",           # I want (variant)
    "يبي": "يريد",           # He wants
    "تبي": "تريد",           # She/You want
    "نبي": "نريد",           # We want
    "يبغى": "يريد",          # He wants (variant)

    # --- Time ---
    "الحين": "الآن",          # Now (Gulf) → Now (MSA)
    "دحين": "الآن",          # Now (variant)
    "هالحين": "الآن",        # Now (variant)
    "أول": "سابقاً",         # Previously / before
    "بعدين": "لاحقاً",      # Later / afterwards

    # --- Adjectives/Adverbs ---
    "وايد": "كثير",          # A lot / very (Gulf)
    "حيل": "كثيراً",         # Very much (Gulf)
    "مره": "جداً",           # Very (Gulf)
    "زين": "جيد",            # Good (Gulf)
    "حلو": "جميل",          # Nice/beautiful (Gulf)
    "خوش": "جيد",            # Good (Gulf/Iraqi)
    "مو": "ليس",             # Not (Gulf)
    "مب": "ليس",             # Not (Emirati variant)

    # --- Nouns ---
    "يهال": "أطفال",         # Children (Gulf)
    "عيال": "أطفال",         # Children (Gulf variant)
    "ربع": "أصدقاء",        # Friends / group (Gulf)
    "سيدا": "مباشرة",       # Directly (Gulf)
    "خل": "دع",              # Let (Gulf)
    "خلك": "كن",             # Be (Gulf)

    # --- Affirmative/Negative ---
    "إي": "نعم",             # Yes (Gulf)
    "إيه": "نعم",            # Yes (variant)
    "هيه": "نعم",            # Yeah (Gulf)

    # ==========================================================
    # Egyptian Arabic
    # ==========================================================

    # --- Question words ---
    "ازاي": "كيف",           # How (Egyptian)
    "إزاي": "كيف",           # How (variant spelling)
    "فين": "أين",            # Where (Egyptian)
    "إمتى": "متى",           # When (Egyptian)

    # --- Verbs ---
    "عايز": "أريد",          # I want (Egyptian, masculine)
    "عايزة": "أريد",         # I want (Egyptian, feminine)
    "عاوز": "أريد",          # I want (Egyptian variant)
    "مش عايز": "لا أريد",   # I don't want (Egyptian)

    # --- Time ---
    "دلوقتي": "الآن",       # Now (Egyptian)
    "إمبارح": "أمس",        # Yesterday (Egyptian)
    "بكره": "غداً",          # Tomorrow (Egyptian)

    # --- Adjectives/Adverbs ---
    "كده": "هكذا",           # Like this (Egyptian)
    "أوي": "جداً",           # Very (Egyptian)
    "خالص": "تماماً",        # Completely (Egyptian)
    "حاجة": "شيء",          # Thing (Egyptian)
    "مش": "ليس",             # Not (Egyptian)
    "كويس": "جيد",           # Good (Egyptian)

    # --- Common phrases ---
    "يعني": "أي",            # I mean / that is (Egyptian, pan-Arab)
    "طيب": "حسناً",          # OK (Egyptian)

    # ==========================================================
    # Levantine Arabic (Syria, Lebanon, Jordan, Palestine)
    # ==========================================================

    # --- Question words ---
    "شو": "ماذا",            # What (Levantine)
    "كيفك": "كيف حالك",     # How are you (Levantine)

    # --- Verbs ---
    "بدي": "أريد",           # I want (Levantine)
    "بدك": "تريد",           # You want (Levantine)
    "بدو": "يريد",           # He wants (Levantine)
    "بدها": "تريد",          # She wants (Levantine)
    "بدنا": "نريد",          # We want (Levantine)

    # --- Time ---
    "هلق": "الآن",           # Now (Levantine)
    "هلأ": "الآن",           # Now (variant)
    "بكرا": "غداً",          # Tomorrow (Levantine)
    "مبارح": "أمس",          # Yesterday (Levantine)

    # --- Adjectives/Adverbs ---
    "هون": "هنا",            # Here (Levantine)
    "هونيك": "هناك",         # There (Levantine)
    "كتير": "كثير",          # A lot (Levantine)
    "منيح": "جيد",           # Good (Levantine)
    "هيك": "هكذا",           # Like this (Levantine)
}

# Compile a regex pattern matching all dialect terms
# Sort by length (longest first) to avoid partial matches
_DIALECT_TERMS_SORTED = sorted(DIALECT_TO_MSA_MAP.keys(), key=len, reverse=True)
_DIALECT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in _DIALECT_TERMS_SORTED) + r")\b",
    re.UNICODE,
)


class DialectMapper:
    """
    Maps Arabic dialect terms to Modern Standard Arabic (MSA).

    Supports Gulf, Egyptian, and Levantine dialects.
    Provides two main functionalities:
    1. Direct replacement of dialect terms with MSA equivalents
    2. Query expansion (original + normalized) for better retrieval
    """

    def __init__(self, custom_mappings: Optional[dict[str, str]] = None):
        """
        Initialize the dialect mapper.

        Args:
            custom_mappings: Additional dialect→MSA mappings to merge with defaults.
        """
        self.mapping = DIALECT_TO_MSA_MAP.copy()
        if custom_mappings:
            self.mapping.update(custom_mappings)

        # Rebuild regex with updated mappings
        terms_sorted = sorted(self.mapping.keys(), key=len, reverse=True)
        self.pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in terms_sorted) + r")\b",
            re.UNICODE,
        )

    def normalize_to_msa(self, text: str) -> str:
        """
        Replace dialect terms with MSA equivalents.

        Args:
            text: Arabic text potentially containing dialect terms.

        Returns:
            Text with dialect terms replaced by MSA equivalents.
        """
        if not text:
            return ""

        def _replace_match(match: re.Match) -> str:
            return self.mapping.get(match.group(0), match.group(0))

        return self.pattern.sub(_replace_match, text)

    def expand_query(self, query: str) -> list[str]:
        """
        Expand a query by generating both the original and MSA-normalized version.
        This improves retrieval by searching with multiple representations.

        Args:
            query: User's question (potentially in Gulf dialect).

        Returns:
            List of query variants: [original, msa_normalized]
            If no dialect terms found, returns just [original].
        """
        if not query:
            return []

        normalized = self.normalize_to_msa(query)

        # Only return both if normalization actually changed something
        if normalized != query:
            return [query, normalized]
        return [query]

    def has_dialect_terms(self, text: str) -> bool:
        """
        Check if text contains any known dialect terms (Gulf, Egyptian, Levantine).

        Args:
            text: Text to check.

        Returns:
            True if dialect terms are found.
        """
        if not text:
            return False
        return bool(self.pattern.search(text))

    def get_dialect_terms_found(self, text: str) -> list[dict[str, str]]:
        """
        Extract all dialect terms found in the text with their MSA equivalents.

        Args:
            text: Text to scan.

        Returns:
            List of dicts: [{"dialect": term, "msa": equivalent}, ...]
        """
        if not text:
            return []

        found = []
        seen = set()
        for match in self.pattern.finditer(text):
            term = match.group(0)
            if term not in seen:
                seen.add(term)
                found.append(
                    {
                        "dialect": term,
                        "msa": self.mapping[term],
                    }
                )
        return found


# ============================================================
# Module-level convenience instances
# ============================================================

_default_mapper = DialectMapper()


def normalize_dialect(text: str) -> str:
    """Convenience: normalize Arabic dialect (Gulf/Egyptian/Levantine) to MSA."""
    return _default_mapper.normalize_to_msa(text)


def expand_dialect_query(query: str) -> list[str]:
    """Convenience: expand query with MSA normalization."""
    return _default_mapper.expand_query(query)
