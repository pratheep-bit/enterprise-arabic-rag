"""Tests for DialectMapper — Gulf, Egyptian, Levantine dialect handling."""

from app.utils.dialect_mapper import DialectMapper


mapper = DialectMapper()


class TestGulfDialect:
    """Gulf Arabic term detection and expansion."""

    def test_detects_shloun(self):
        assert mapper.has_dialect_terms("شلون الحال")

    def test_detects_weish(self):
        assert mapper.has_dialect_terms("ويش تبي")

    def test_detects_abi(self):
        assert mapper.has_dialect_terms("أبي أعرف")

    def test_detects_alhin(self):
        assert mapper.has_dialect_terms("الحين وين")

    def test_detects_shnu(self):
        assert mapper.has_dialect_terms("شنو هذا")

    def test_detects_hayil(self):
        assert mapper.has_dialect_terms("حيل زين")

    def test_normalizes_shloun_to_kayf(self):
        result = mapper.normalize_to_msa("شلون الحال")
        assert "كيف" in result

    def test_normalizes_weish_to_matha(self):
        result = mapper.normalize_to_msa("ويش تبي")
        assert "ماذا" in result

    def test_expand_query_with_dialect(self):
        variants = mapper.expand_query("شلون الأمور")
        assert len(variants) == 2
        assert variants[0] == "شلون الأمور"
        assert "كيف" in variants[1]


class TestEgyptianDialect:
    """Egyptian Arabic term detection."""

    def test_detects_ezay(self):
        assert mapper.has_dialect_terms("ازاي الحال")

    def test_detects_fein(self):
        assert mapper.has_dialect_terms("فين المكان")

    def test_detects_ayez(self):
        assert mapper.has_dialect_terms("عايز أعرف")

    def test_detects_dilwaqti(self):
        assert mapper.has_dialect_terms("دلوقتي مش وقته")

    def test_detects_kwayyes(self):
        assert mapper.has_dialect_terms("كويس جداً")

    def test_normalizes_ezay_to_kayf(self):
        result = mapper.normalize_to_msa("ازاي الحال")
        assert "كيف" in result

    def test_normalizes_ayez_to_ureed(self):
        result = mapper.normalize_to_msa("عايز أعرف")
        assert "أريد" in result


class TestLevantineDialect:
    """Levantine Arabic term detection."""

    def test_detects_shou(self):
        assert mapper.has_dialect_terms("شو بدك")

    def test_detects_biddi(self):
        assert mapper.has_dialect_terms("بدي أروح")

    def test_detects_halla(self):
        assert mapper.has_dialect_terms("هلق لازم")

    def test_detects_hon(self):
        assert mapper.has_dialect_terms("هون المحل")

    def test_detects_mneeh(self):
        assert mapper.has_dialect_terms("منيح كتير")

    def test_normalizes_biddi_to_ureed(self):
        result = mapper.normalize_to_msa("بدي أروح")
        assert "أريد" in result

    def test_normalizes_shou_to_matha(self):
        result = mapper.normalize_to_msa("شو القصة")
        assert "ماذا" in result


class TestMSAPassthrough:
    """MSA text should pass through unchanged."""

    def test_msa_no_dialect(self):
        text = "الأهداف الرئيسية للمشروع المذكور"
        assert not mapper.has_dialect_terms(text)

    def test_expand_query_msa_only(self):
        text = "الأهداف الرئيسية للمشروع"
        variants = mapper.expand_query(text)
        assert len(variants) == 1
        assert variants[0] == text


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_string(self):
        assert not mapper.has_dialect_terms("")
        assert mapper.normalize_to_msa("") == ""
        assert mapper.expand_query("") == []

    def test_get_dialect_terms_found_returns_dicts(self):
        found = mapper.get_dialect_terms_found("شلون الحال ويش تبي")
        assert len(found) >= 2
        assert "dialect" in found[0]
        assert "msa" in found[0]

    def test_get_dialect_terms_found_empty(self):
        found = mapper.get_dialect_terms_found("")
        assert found == []

    def test_custom_mappings(self):
        custom = DialectMapper(custom_mappings={"تست": "اختبار"})
        assert custom.has_dialect_terms("تست")
        result = custom.normalize_to_msa("تست")
        assert "اختبار" in result
