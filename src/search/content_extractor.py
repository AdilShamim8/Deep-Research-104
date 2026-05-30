"""
Web content extractor.

Extracts clean article text from URLs using:
  1. trafilatura  (fast, best for articles)
  2. readability  (Mozilla-style extraction)
  3. BeautifulSoup (fallback, basic)
  4. Playwright   (for JavaScript-heavy pages)

Always respects token limits when returning content.
"""

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import aiohttp
import trafilatura
from bs4 import BeautifulSoup
from loguru import logger

from src.utils.cache import cache, make_cache_key
from src.utils.token_counter import TokenCounter
from config.settings import settings


# ── Extracted content ─────────────────────────────────────────────────────────

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

    def __post_init__(self):
        if not self.domain:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc.replace("www.", "")
        if self.text:
            self.word_count = len(self.text.split())

    def truncate_to_tokens(
        self,
        max_tokens: int,
        model: str = "gpt-4o"
    ) -> "ExtractedContent":
        """Return a copy with text truncated to max_tokens."""
        counter = TokenCounter(model)
        self.token_count = counter.count_tokens(self.text)
        if self.token_count > max_tokens:
            self.text = counter.truncate_text_to_tokens(
                self.text, max_tokens
            )
            self.token_count = max_tokens
        return self

    def to_context_string(self, include_metadata: bool = True) -> str:
        """Format for inclusion in LLM prompt."""
        parts = []
        if include_metadata:
            parts.append(f"Source: {self.title} ({self.domain})")
            parts.append(f"URL: {self.url}")
            if self.published_date:
                parts.append(f"Date: {self.published_date}")
            parts.append("")  # blank line
        parts.append(self.text)
        return "\n".join(parts)


# ── Extractor ─────────────────────────────────────────────────────────────────

# Domains that are typically not useful to scrape
BLOCKED_DOMAINS = {
    "youtube.com", "youtu.be",
    "twitter.com", "x.com",
    "instagram.com", "facebook.com",
    "tiktok.com", "reddit.com",
    "linkedin.com",
}

# File extensions to skip
BLOCKED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
    ".xls", ".xlsx", ".zip", ".tar", ".gz",
    ".jpg", ".jpeg", ".png", ".gif", ".mp4",
    ".mp3", ".wav", ".avi",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


class ContentExtractor:
    """
    Extracts clean article text from web pages.

    Tries multiple extraction strategies in order of quality,
    with caching to avoid re-fetching the same URLs.
    """

    def __init__(
        self,
        max_tokens_per_page: int = 4000,
        timeout: int = 15,
        use_playwright: bool = False,
    ):
        self.max_tokens_per_page = max_tokens_per_page
        self.timeout = timeout
        self.use_playwright = use_playwright
        self._token_counter = TokenCounter()

    # ── Public API ────────────────────────────────────────────────────────────

    async def extract(
        self,
        url: str,
        use_cache: bool = True,
        max_tokens: Optional[int] = None,
    ) -> ExtractedContent:
        """
        Extract content from a single URL.

        Args:
            url:        URL to extract from.
            use_cache:  Use cached content if available.
            max_tokens: Override max tokens per page.
        """
        max_tokens = max_tokens or self.max_tokens_per_page

        # Check if URL should be skipped
        skip_reason = self._should_skip(url)
        if skip_reason:
            return ExtractedContent(
                url=url,
                title="",
                text="",
                success=False,
                error=skip_reason,
            )

        # Check cache
        if use_cache:
            cache_key = make_cache_key("content", url, max_tokens)
            cached = await cache.get(cache_key)
            if cached:
                logger.debug(f"Content cache hit: {url[:60]}")
                return ExtractedContent(**cached)

        # Fetch and extract
        content = await self._fetch_and_extract(url, max_tokens)

        # Cache successful extractions
        if use_cache and content.success:
            await cache.set(
                make_cache_key("content", url, max_tokens),
                {
                    "url": content.url,
                    "title": content.title,
                    "text": content.text,
                    "author": content.author,
                    "published_date": content.published_date,
                    "domain": content.domain,
                    "word_count": content.word_count,
                    "token_count": content.token_count,
                    "extraction_method": content.extraction_method,
                    "extraction_time_ms": content.extraction_time_ms,
                    "success": content.success,
                    "error": content.error,
                },
                ttl=7200,  # Cache pages for 2 hours
            )

        return content

    async def extract_many(
        self,
        urls: list[str],
        max_concurrent: int = 5,
        max_tokens_each: Optional[int] = None,
    ) -> list[ExtractedContent]:
        """
        Extract content from multiple URLs concurrently.
        Limits concurrency to avoid overwhelming servers.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_extract(url: str) -> ExtractedContent:
            async with semaphore:
                return await self.extract(url, max_tokens=max_tokens_each)

        tasks = [bounded_extract(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        contents = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.warning(f"Extraction failed for {url}: {result}")
                contents.append(ExtractedContent(
                    url=url, title="", text="",
                    success=False, error=str(result)
                ))
            else:
                contents.append(result)

        successful = sum(1 for c in contents if c.success)
        logger.info(
            f"Extracted {successful}/{len(urls)} URLs successfully"
        )
        return contents

    # ── Private methods ───────────────────────────────────────────────────────

    def _should_skip(self, url: str) -> Optional[str]:
        """Return skip reason if URL should not be fetched."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "").lower()

            # Check blocked domains
            for blocked in BLOCKED_DOMAINS:
                if blocked in domain:
                    return f"Blocked domain: {domain}"

            # Check blocked extensions
            path = parsed.path.lower()
            for ext in BLOCKED_EXTENSIONS:
                if path.endswith(ext):
                    return f"Blocked file type: {ext}"

            # Must be http/https
            if parsed.scheme not in ("http", "https"):
                return f"Invalid scheme: {parsed.scheme}"

        except Exception as e:
            return f"URL parse error: {e}"

        return None

    async def _fetch_and_extract(
        self,
        url: str,
        max_tokens: int,
    ) -> ExtractedContent:
        """Fetch HTML and extract content using best available method."""
        start = time.time()

        # Step 1: Fetch raw HTML
        html = await self._fetch_html(url)
        if not html:
            if self.use_playwright:
                html = await self._fetch_with_playwright(url)

        if not html:
            return ExtractedContent(
                url=url, title="", text="",
                success=False, error="Failed to fetch HTML",
                extraction_time_ms=(time.time() - start) * 1000,
            )

        # Step 2: Extract content (try methods in order)
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

        # Step 3: Clean and truncate
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
        """Fetch raw HTML with aiohttp."""
        try:
            async with aiohttp.ClientSession(
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=aiohttp.TCPConnector(ssl=False),
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.debug(
                            f"HTTP {resp.status} for {url[:60]}"
                        )
                        return None
                    content_type = resp.headers.get(
                        "Content-Type", ""
                    ).lower()
                    if "html" not in content_type:
                        return None
                    return await resp.text(errors="replace")
        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching {url[:60]}")
            return None
        except Exception as e:
            logger.debug(f"Fetch error for {url[:60]}: {e}")
            return None

    async def _fetch_with_playwright(self, url: str) -> Optional[str]:
        """Fetch JavaScript-rendered pages with Playwright."""
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(
                    url,
                    timeout=self.timeout * 1000,
                    wait_until="domcontentloaded",
                )
                await page.wait_for_timeout(2000)  # Wait for JS
                html = await page.content()
                await browser.close()
                return html
        except Exception as e:
            logger.debug(f"Playwright error for {url[:60]}: {e}")
            return None

    def _extract_trafilatura(
        self, url: str, html: str
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
                    author=(
                        metadata.author if metadata else None
                    ),
                    published_date=(
                        str(metadata.date) if metadata and metadata.date
                        else None
                    ),
                    extraction_method="trafilatura",
                )
        except Exception as e:
            logger.debug(f"Trafilatura failed for {url[:60]}: {e}")
        return None

    def _extract_readability(
        self, url: str, html: str
    ) -> Optional[ExtractedContent]:
        """Extract using readability-lxml."""
        try:
            from readability import Document
            doc = Document(html)
            title = doc.title()
            content_html = doc.summary()

            # Convert to plain text
            soup = BeautifulSoup(content_html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            if text and len(text.strip()) > 100:
                return ExtractedContent(
                    url=url,
                    title=title,
                    text=text.strip(),
                    extraction_method="readability",
                )
        except Exception as e:
            logger.debug(f"Readability failed for {url[:60]}: {e}")
        return None

    def _extract_beautifulsoup(
        self, url: str, html: str
    ) -> Optional[ExtractedContent]:
        """Basic BeautifulSoup extraction as last resort."""
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove noise elements
            for tag in soup(
                ["script", "style", "nav", "header",
                 "footer", "aside", "form", "iframe"]
            ):
                tag.decompose()

            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Try article tag first, then main, then body
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
                        url=url,
                        title=title,
                        text=text.strip(),
                        extraction_method="beautifulsoup",
                    )
        except Exception as e:
            logger.debug(f"BeautifulSoup failed for {url[:60]}: {e}")
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        # Remove common boilerplate patterns
        text = re.sub(
            r'(Subscribe to our newsletter|Sign up for|Cookie Policy'
            r'|Privacy Policy|Terms of Service|All rights reserved'
            r'|Click here to|Read more at)[^\n]*\n?',
            '',
            text,
            flags=re.IGNORECASE,
        )
        return text.strip()


# Singleton
extractor = ContentExtractor(
    max_tokens_per_page=settings.search.max_content_length,
    use_playwright=False,
)