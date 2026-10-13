import asyncio
from app.api import routes
from app.models.schemas import TextAnalysisRequest


def test_analyze_arabic_content_route():
    req = TextAnalysisRequest(text="شلون الحال دابا؟ هَذَا اختبار رائع!")
    response = asyncio.run(routes.analyze_arabic_content(req))

    assert response.total_chars > 0
    assert response.word_count > 0
    assert response.is_primarily_arabic is True
    assert response.diacritic_count > 0
    # Dialect detection should detect شلون (Gulf) and دابا (Maghrebi)
    terms = [d["dialect"] for d in response.detected_dialects]
    assert "شلون" in terms
    assert "دابا" in terms


def test_analyze_empty_arabic_content():
    req = TextAnalysisRequest(text="Hello world")
    response = asyncio.run(routes.analyze_arabic_content(req))
    assert response.is_primarily_arabic is False
    assert response.word_count == 0
    assert response.detected_dialects == []
