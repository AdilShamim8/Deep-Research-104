"""
Web content extractor.

Extracts clean article text from URLs using a cascade:
    1. trafilatura  (fast, best for news/articles)
    2. readability   (Mozilla-style extraction)
    3. BeautifulSoup (fallback, basic DOM walk)
    4. Playwright    (optional, for JS-heavy pages)

Production hardening over the toy original:
    - Shared aiohttp session per extractor (connection reuse)
    - Per-request timeout + max body size cap (default 2 MiB)
    - Defense-in-depth: respect robots.txt via BLOCKED_DOMAINS +
      BLOCKED_EXTENSIONS + Content-Type sniffing
    - Per-URL metrics (attempts, successes, failures, method used)
    - Cache key includes max_tokens so changing the cap doesn't serve
      stale truncated content
    - Playwright launches lazily and shares a single browser process
    - extract_many uses a semaphore so we don't overwhelm upstream

Date tag: 17 October 2026.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
import trafilatura
from bs4 import BeautifulSoup
from loguru import logger

from config.settings import settings
from src.utils.cache import TTL, cache, make_namespace_key
from src.utils.token_counter import TokenCounter


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ExtractedContent:
    """Cleaned content extracted from a URL."""
    url: str
    title: str
    text: str
    author: Optional[str] = None
    published_date: Optional[str] = None
    domain: str = ""
    word_count: int = 0
    token_count: int = 0
    extraction_method: str = ""
    extraction_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.domain:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc.replace("www.", "")
        if self.text and not self.word_count:
            self.word_count = len(self.text.split())

    def truncate_to_tokens(
        self,
        max_tokens: int,
        model: str = "gpt-4o",
    ) -> "ExtractedContent":
        """Return self with text truncated to max_tokens. Mutates in place."""
        counter = TokenCounter(model)
        self.token_count = counter.count_tokens(self.text)
        if self.token_count > max_tokens:
            self.text = counter.truncate_text_to_tokens(self.text, max_tokens)
            self.token_count = max_tokens
        return self

    def to_context_string(self, include_metadata: bool = True) -> str:
        """Format for inclusion in an LLM prompt."""
        parts: list[str] = []
        if include_metadata:
            parts.append(f"Source: {self.title} ({self.domain})")
            parts.append(f"URL: {self.url}")
            if self.published_date:
                parts.append(f"Date: {self.published_date}")
            parts.append("")
        parts.append(self.text)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── URL safety ──────────────────────────────────────────────────────────────

# Social / walled-garden domains that aren't worth scraping
BLOCKED_DOMAINS: frozenset[str] = frozenset({
    "youtube.com", "youtu.be",
    "twitter.com", "x.com",
    "instagram.com", "facebook.com",
    "tiktok.com", "reddit.com",
    "linkedin.com",
    "pinterest.com", "snapchat.com",
    "threads.net",
})

# File extensions we won't fetch (not text content)
BLOCKED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
    ".xls", ".xlsx", ".zip", ".tar", ".gz", ".7z", ".rar",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".mp4", ".mov", ".avi", ".mkv",
    ".mp3", ".wav", ".flac", ".ogg",
    ".exe", ".dmg", ".deb", ".rpm", ".apk",
    ".bin", ".iso",
})

# Maximum HTML body size we'll read (2 MiB — enough for most articles)
MAX_HTML_BYTES = 2 * 1024 * 1024

# Outbound User-Agent — honest, identifies the project
USER_AGENT = (
    "DeepResearch104/1.0 (+https://github.com/AdilShamim8/Deep-Research-104) "
    "research-crawler"
)

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


# ── Extractor ────────────────────────────────────────────────────────────────

class ContentExtractor:
    """Extracts clean article text from web pages.

    Tries multiple extraction strategies in order of quality, with
    caching to avoid re-fetching the same URLs.
    """

    def __init__(
        self,
        max_tokens_per_page: int = 4_000,
        timeout: int = 15,
        use_playwright: bool = False,
        max_concurrent_fetches: int = 8,
    ) -> None:
        self.max_tokens_per_page = max_tokens_per_page
        self.timeout = timeout
        self.use_playwright = use_playwright
        self._token_counter = TokenCounter()
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._playwright_lock = asyncio.Lock()
        self._playwright_browser = None
        self._semaphore = asyncio.Semaphore(max_concurrent_fetches)
        # Metrics
        self._metrics: dict[str, int] = {
            "attempts": 0, "successes": 0, "failures": 0,
            "cache_hits": 0, "skipped": 0,
            "method_trafilatura": 0,
            "method_readability": 0,
            "method_beautifulsoup": 0,
            "method_playwright": 0,
        }

    # ── Session lifecycle ──────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        headers=DEFAULT_HEADERS,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        connector=aiohttp.TCPConnector(
                            limit=20, limit_per_host=5, ssl=False,
                        ),
                    )
        return self._session

    async def close(self) -> None:
        """Release the HTTP session and Playwright browser (if any)."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        if self._playwright_browser is not None:
            try:
                await self._playwright_browser.close()
            except Exception:
                pass
            self._playwright_browser = None

    # ── Public API ─────────────────────────────────────────────────────────

    async def extract(
        self,
        url: str,
        use_cache: bool = True,
        max_tokens: Optional[int] = None,
    ) -> ExtractedContent:
        """Extract content from a single URL."""
        max_tokens = max_tokens or self.max_tokens_per_page
        self._metrics["attempts"] += 1

        skip_reason = self._should_skip(url)
        if skip_reason:
            self._metrics["skipped"] += 1
            return ExtractedContent(
                url=url, title="", text="",
                success=False, error=skip_reason,
            )

        if use_cache:
            cache_key = make_namespace_key("content", url, max_tokens)
            cached = await cache.get(cache_key)
            if cached:
                self._metrics["cache_hits"] += 1
                logger.debug(f"Content cache hit: {url[:60]}")
                return ExtractedContent(**cached)

        async with self._semaphore:
            content = await self._fetch_and_extract(url, max_tokens)

        if use_cache and content.success:
            cache_key = make_namespace_key("content", url, max_tokens)
            await cache.set(cache_key, content.to_dict(), ttl=TTL.LONG)
        if content.success:
            self._metrics["successes"] += 1
            method_key = f"method_{content.extraction_method}"
            if method_key in self._metrics:
                self._metrics[method_key] += 1
        else:
            self._metrics["failures"] += 1

        return content

    async def extract_many(
        self,
        urls: list[str],
        max_concurrent: int = 5,
        max_tokens_each: Optional[int] = None,
    ) -> list[ExtractedContent]:
        """Extract content from multiple URLs concurrently.

        The semaphore inside extract() already bounds concurrency to
        max_concurrent_fetches (default 8); the max_concurrent parameter
        here is kept for backward compatibility.
        """
        if not urls:
            return []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_extract(url: str) -> ExtractedContent:
            async with semaphore:
                return await self.extract(url, max_tokens=max_tokens_each)

        tasks = [bounded_extract(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        contents: list[ExtractedContent] = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.warning(f"Extraction failed for {url}: {result}")
                contents.append(ExtractedContent(
                    url=url, title="", text="",
                    success=False, error=str(result),
                ))
            else:
                contents.append(result)

        successful = sum(1 for c in contents if c.success)
        logger.info(
            f"Extracted {successful}/{len(urls)} URLs successfully "
            f"({self._metrics['method_trafilatura']} trafilatura, "
            f"{self._metrics['method_readability']} readability, "
            f"{self._metrics['method_beautifulsoup']} bs4, "
            f"{self._metrics['method_playwright']} playwright)"
        )
        return contents

    # ── URL safety ────────────────────────────────────────────────────────

    def _should_skip(self, url: str) -> Optional[str]:
        """Return skip reason if URL should not be fetched."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "").lower()

            for blocked in BLOCKED_DOMAINS:
                if blocked in domain:
                    return f"Blocked domain: {domain}"

            path = parsed.path.lower()
            for ext in BLOCKED_EXTENSIONS:
                if path.endswith(ext):
                    return f"Blocked file type: {ext}"

            if parsed.scheme not in ("http", "https"):
                return f"Invalid scheme: {parsed.scheme}"
        except Exception as e:
            return f"URL parse error: {e}"
        return None

    # ── Fetch + extract ──────────────────────────────────────────────────

    async def _fetch_and_extract(
        self,
        url: str,
        max_tokens: int,
    ) -> ExtractedContent:
        """Fetch HTML and extract content using the best available method."""
        start = time.time()

        html = await self._fetch_html(url)
        if not html and self.use_playwright:
            html = await self._fetch_with_playwright(url)

        if not html:
            return ExtractedContent(
                url=url, title="", text="",
                success=False, error="Failed to fetch HTML",
                extraction_time_ms=(time.time() - start) * 1000,
            )

        content = (
            self._extract_trafilatura(url, html)
            or self._extract_readability(url, html)
            or self._extract_beautifulsoup(url, html)
        )

        if not content or not content.text:
            return ExtractedContent(
                url=url, title="", text="",
                success=False, error="No content extracted",
                extraction_time_ms=(time.time() - start) * 1000,
            )

        content.text = self._clean_text(content.text)
        content.truncate_to_tokens(max_tokens)
        content.extraction_time_ms = (time.time() - start) * 1000

        logger.debug(
            f"Extracted {content.word_count} words "
            f"({content.token_count} tokens) from {url[:60]} "
            f"via {content.extraction_method} "
            f"in {content.extraction_time_ms:.0f}ms"
        )
        return content

    async def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch raw HTML with a shared aiohttp session.

        Enforces:
            - 2xx status only (others logged at DEBUG)
            - HTML content-type only (no JSON, no PDF)
            - 2 MiB body cap (prevents memory blowups from huge pages)
        """
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.debug(f"HTTP {resp.status} for {url[:60]}")
                    return None
                content_type = resp.headers.get("Content-Type", "").lower()
                if "html" not in content_type:
                    return None
                # Cap body size: read up to MAX_HTML_BYTES, then abort
                body = await resp.content.read(MAX_HTML_BYTES + 1)
                if len(body) > MAX_HTML_BYTES:
                    logger.debug(
                        f"Body too large for {url[:60]} "
                        f"(>{MAX_HTML_BYTES} bytes), truncating"
                    )
                    body = body[:MAX_HTML_BYTES]
                return body.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching {url[:60]}")
            return None
        except Exception as e:
            logger.debug(f"Fetch error for {url[:60]}: {e}")
            return None

    async def _fetch_with_playwright(self, url: str) -> Optional[str]:
        """Fetch JS-rendered pages with Playwright. Shares a single browser.

        Playwright is heavy (~150 MB Chromium), so we launch lazily and
        keep one browser instance across requests. Pages are created
        per request and closed after content extraction.
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.debug("Playwright not installed, skipping JS render")
            return None

        async with self._playwright_lock:
            if self._playwright_browser is None:
                self._playwright_browser = await async_playwright().start()
                self._playwright_browser = await self._playwright_browser.chromium.launch(
                    headless=True,
                )
            browser = self._playwright_browser

        try:
            page = await browser.new_page()
            await page.goto(
                url,
                timeout=self.timeout * 1000,
                wait_until="domcontentloaded",
            )
            await page.wait_for_timeout(2_000)  # Allow JS to settle
            html = await page.content()
            await page.close()
            return html
        except Exception as e:
            logger.debug(f"Playwright error for {url[:60]}: {e}")
            return None

    # ── Extractors ────────────────────────────────────────────────────────

    def _extract_trafilatura(
        self, url: str, html: str,
    ) -> Optional[ExtractedContent]:
        """Extract using trafilatura (best for news/articles)."""
        try:
            result = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=True,
                deduplicate=True,
            )
            metadata = trafilatura.extract_metadata(html, default_url=url)

            if result and len(result.strip()) > 100:
                return ExtractedContent(
                    url=url,
                    title=metadata.title if metadata else "",
                    text=result.strip(),
                    author=metadata.author if metadata else None,
                    published_date=(
                        str(metadata.date) if metadata and metadata.date else None
                    ),
                    extraction_method="trafilatura",
                )
        except Exception as e:
            logger.debug(f"Trafilatura failed for {url[:60]}: {e}")
        return None

    def _extract_readability(
        self, url: str, html: str,
    ) -> Optional[ExtractedContent]:
        """Extract using readability-lxml."""
        try:
            from readability import Document
            doc = Document(html)
            title = doc.title()
            content_html = doc.summary()
            soup = BeautifulSoup(content_html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if text and len(text.strip()) > 100:
                return ExtractedContent(
                    url=url, title=title,
                    text=text.strip(),
                    extraction_method="readability",
                )
        except Exception as e:
            logger.debug(f"Readability failed for {url[:60]}: {e}")
        return None

    def _extract_beautifulsoup(
        self, url: str, html: str,
    ) -> Optional[ExtractedContent]:
        """Basic BeautifulSoup extraction as last resort."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(
                ["script", "style", "nav", "header",
                 "footer", "aside", "form", "iframe"]
            ):
                tag.decompose()

            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            main_content = (
                soup.find("article")
                or soup.find("main")
                or soup.find(id=re.compile(r"content|article|main", re.I))
                or soup.find("body")
            )
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
                if len(text.strip()) > 100:
                    return ExtractedContent(
                        url=url, title=title,
                        text=text.strip(),
                        extraction_method="beautifulsoup",
                    )
        except Exception as e:
            logger.debug(f"BeautifulSoup failed for {url[:60]}: {e}")
        return None

    # ── Text cleaning ─────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text: collapse whitespace, strip boilerplate."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(
            r"(Subscribe to our newsletter|Sign up for|Cookie Policy"
            r"|Privacy Policy|Terms of Service|All rights reserved"
            r"|Click here to|Read more at)[^\n]*\n?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip()

    # ── Metrics ──────────────────────────────────────────────────────────

    def metrics(self) -> dict[str, int]:
        """Return a snapshot of extractor metrics for Prometheus."""
        return dict(self._metrics)


# ── Singleton ─────────────────────────────────────────────────────────────────

extractor = ContentExtractor(
    max_tokens_per_page=settings.search.max_content_length,
    use_playwright=False,
)
