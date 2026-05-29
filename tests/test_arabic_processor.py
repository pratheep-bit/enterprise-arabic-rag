"""Tests for ArabicProcessor — Unicode normalization, Kashida, diacritics, etc."""

from app.utils.arabic_processor import ArabicProcessor, normalize_for_embedding


processor = ArabicProcessor()


class TestArabicProcessorBasics:
    """Basic input handling."""

    def test_empty_string(self):
        assert processor.process("") == ""

    def test_none_returns_empty(self):
        assert processor.process(None) == ""

    def test_whitespace_only(self):
        result = processor.process("   \n\t  ")
        assert result.strip() == ""

    def test_english_only_passthrough(self):
        result = processor.process("Hello World")
        assert "Hello" in result


class TestKashidaRemoval:
    """Kashida (Tatweel U+0640) removal."""

    def test_removes_kashida(self):
        result = processor.process("مـــرحـــبـــا")
        assert "\u0640" not in result

    def test_no_kashida_unchanged(self):
        text = "مرحبا"
        result = processor.process(text)
        assert "مرحبا" in result


class TestDiacriticsStripping:
    """Arabic diacritical marks removal."""

    def test_strips_fatha(self):
        result = processor.process("كَتَبَ")
        assert "\u064E" not in result  # Fatha

    def test_strips_kasra(self):
        result = processor.process("بِسْمِ")
        assert "\u0650" not in result  # Kasra

    def test_strips_damma(self):
        result = processor.process("كُتُبٌ")
        assert "\u064F" not in result  # Damma

    def test_strips_shadda(self):
        result = processor.process("شَدَّة")
        assert "\u0651" not in result  # Shadda

    def test_strips_tanwin(self):
        result = processor.process("كتاباً")
        assert "\u064B" not in result  # Fathatan

    def test_pure_diacritics_returns_empty(self):
        """A string that is only diacritics should produce empty/whitespace."""
        result = processor.process("\u064E\u064F\u0650\u0651\u0652")
        assert result.strip() == ""


class TestAlefNormalization:
    """Alef variant normalization (أ/إ/آ → ا)."""

    def test_alef_hamza_above(self):
        result = processor.process("أحمد")
        assert "احمد" in result

    def test_alef_hamza_below(self):
        result = processor.process("إسلام")
        assert "اسلام" in result

    def test_alef_madda(self):
        result = processor.process("آمنة")
        assert "امنه" in result or "امنة" in result


class TestRTLMarkers:
    """RTL/LTR Unicode marker removal."""

    def test_removes_rtl_mark(self):
        result = processor.process("مرحبا\u200F")
        assert "\u200F" not in result

    def test_removes_ltr_mark(self):
        result = processor.process("\u200Eمرحبا")
        assert "\u200E" not in result

    def test_removes_bom(self):
        result = processor.process("\uFEFFمرحبا")
        assert "\uFEFF" not in result


class TestMixedContent:
    """Mixed Arabic-English text."""

    def test_mixed_arabic_english(self):
        result = processor.process("مرحبا Hello مرحبا")
        assert "Hello" in result
        assert "\u0640" not in result

    def test_numbers_preserved(self):
        result = processor.process("العدد 42 والعدد 100")
        assert "42" in result
        assert "100" in result


class TestNormalizeForEmbedding:
    """Test the convenience function used before embedding."""

    def test_normalize_for_embedding_strips_diacritics(self):
        result = normalize_for_embedding("كَتَبَ")
        assert "\u064E" not in result

    def test_normalize_for_embedding_empty(self):
        result = normalize_for_embedding("")
        assert result == ""


class TestIsArabic:
    """Test Arabic detection."""

    def test_arabic_text(self):
        assert processor.is_arabic("مرحبا بالعالم")

    def test_english_text(self):
        assert not processor.is_arabic("Hello World")

    def test_mixed_majority_arabic(self):
        assert processor.is_arabic("مرحبا بالعالم Hello")
