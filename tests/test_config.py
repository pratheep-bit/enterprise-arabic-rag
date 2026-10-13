from app.core.config import Settings, get_settings


def test_settings_default_values():
    settings = Settings(_env_file=None)
    assert settings.api_port == 8000
    assert settings.api_host == "0.0.0.0"
    assert "8501" in settings.api_cors_origins
    assert settings.min_similarity_threshold == 0.15
    assert settings.enable_query_cache is True
    assert len(settings.get_cors_origins_list()) >= 1


def test_settings_singleton_caching():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_cors_origins_list_parsing():
    settings = Settings(API_CORS_ORIGINS="http://example.com, https://app.example.com")
    origins = settings.get_cors_origins_list()
    assert origins == ["http://example.com", "https://app.example.com"]
