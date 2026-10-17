"""
Unit tests for src.search.content_extractor — URL safety, blocking,
ExtractedContent dataclass, and text cleaning helpers.

No network — we only exercise the URL filtering and dataclass logic.
The extract() / extract_many() paths require the network and are
covered by the integration test suite.
"""

from __future__ import annotations

import pytest
from src.search.content_extractor import (
    BLOCKED_DOMAINS,
    BLOCKED_EXTENSIONS,
    MAX_HTML_BYTES,
    USER_AGENT,
    ContentExtractor,
    ExtractedContent,
)

# ── ExtractedContent dataclass ────────────────────────────────────────────────


class TestExtractedContent:
    def test_basic_construction(self) -> None:
        c = ExtractedContent(
            url="https://example.com/article",
            title="Article",
            text="Some body text.",
        )
        assert c.url == "https://example.com/article"
        assert c.title == "Article"
        assert c.text == "Some body text."
        assert c.domain == "example.com"
        assert c.word_count == 3
        assert c.success is True
        assert c.error is None

    def test_domain_auto_extracts_from_url(self) -> None:
        c = ExtractedContent(url="https://www.example.co.uk/x", title="t", text="hi")
        assert c.domain == "example.co.uk"

    def test_to_context_string_includes_metadata(self) -> None:
        c = ExtractedContent(
            url="https://example.com/a",
            title="Article",
            text="Body here.",
            published_date="2026-10-17",
        )
        s = c.to_context_string()
        assert "Article" in s
        assert "https://example.com/a" in s
        assert "2026-10-17" in s
        assert "Body here." in s

    def test_to_context_string_without_metadata(self) -> None:
        c = ExtractedContent(url="https://example.com/a", title="t", text="body")
        s = c.to_context_string(include_metadata=False)
        assert s == "body"


# ── URL safety ────────────────────────────────────────────────────────────────


class TestUrlSafety:
    def setup_method(self) -> None:
        self.extractor = ContentExtractor()

    def test_blocked_social_domains_rejected(self) -> None:
        for domain in (
            "youtube.com",
            "twitter.com",
            "instagram.com",
            "facebook.com",
            "tiktok.com",
            "reddit.com",
            "linkedin.com",
            "pinterest.com",
        ):
            reason = self.extractor._should_skip(f"https://{domain}/page")
            assert reason is not None, f"Should reject {domain}"
            assert "Blocked domain" in reason

    def test_blocked_extensions_rejected(self) -> None:
        for ext in (".pdf", ".jpg", ".png", ".zip", ".mp4", ".mp3", ".exe"):
            url = f"https://example.com/file{ext}"
            reason = self.extractor._should_skip(url)
            assert reason is not None, f"Should reject {ext}"
            assert "Blocked file type" in reason

    def test_non_http_schemes_rejected(self) -> None:
        reason = self.extractor._should_skip("file:///etc/passwd")
        assert reason is not None
        assert "Invalid scheme" in reason

        reason = self.extractor._should_skip("ftp://example.com/x")
        assert reason is not None
        assert "Invalid scheme" in reason

    def test_valid_http_url_accepted(self) -> None:
        assert self.extractor._should_skip("https://example.com/article") is None

    def test_valid_https_url_accepted(self) -> None:
        assert self.extractor._should_skip("https://blog.example.com/post") is None


# ── Module-level constants ────────────────────────────────────────────────────


class TestConstants:
    def test_max_html_bytes_is_reasonable(self) -> None:
        # 2 MiB
        assert MAX_HTML_BYTES == 2 * 1024 * 1024

    def test_user_agent_identifies_project(self) -> None:
        assert "DeepResearch104" in USER_AGENT
        assert "github.com" in USER_AGENT

    def test_blocked_domains_set_is_frozen(self) -> None:
        # Frozenset means we can't accidentally mutate it
        with pytest.raises(AttributeError):
            BLOCKED_DOMAINS.add("evil.com")  # type: ignore

    def test_blocked_extensions_set_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            BLOCKED_EXTENSIONS.add(".evil")  # type: ignore


# ── Text cleaning ──────────────────────────────────────────────────────────────


class TestCleanText:
    def test_collapses_whitespace(self) -> None:
        cleaned = ContentExtractor._clean_text("a\n\n\n\nb   c")
        assert "\n\n\n" not in cleaned
        assert "  " not in cleaned

    def test_strips_boilerplate(self) -> None:
        text = "Real content\nSubscribe to our newsletter\nMore content"
        cleaned = ContentExtractor._clean_text(text)
        assert "Subscribe to our newsletter" not in cleaned
        assert "Real content" in cleaned

    def test_preserves_normal_text(self) -> None:
        text = "Just a normal paragraph of content here."
        cleaned = ContentExtractor._clean_text(text)
        assert "normal paragraph" in cleaned
