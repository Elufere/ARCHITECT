"""
Fetch raw content from URLs.

Handles:
- Regular HTML pages
- GitHub raw markdown conversion with README fallback
- Basic etiquette: optional robots.txt awareness, timeouts, user-agent, and rate limiting
- Retry with exponential backoff for transient server/network failures
- Anti-bot/login-wall detection before content is saved
"""

from __future__ import annotations

import logging
import random
import re
import time
import urllib.robotparser
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

_last_request_time: dict[str, float] = {}
_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

_MIN_REQUEST_INTERVAL = 1.0
_MAX_REQUEST_JITTER = 0.75
_DEFAULT_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.5
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_USER_AGENT = "Architect-KB-Scraper/0.1 (educational knowledge base builder; contact@example.com)"

_ANTI_BOT_PATTERNS = [
    r"(?i)verify you are human",
    r"(?i)checking your browser",
    r"(?i)cf-browser-verification",
    r"(?i)attention required!\s*\|\s*cloudflare",
    r"(?i)cloudflare ray id",
    r"(?i)captcha",
    r"(?i)unusual traffic",
    r"(?i)access denied",
    r"(?i)enable javascript and cookies",
    r"(?i)login required",
    r"(?i)sign in to continue",
]


@dataclass
class FetchResult:
    """Result of a fetch operation."""

    url: str
    content: str
    content_type: str
    status_code: int
    success: bool
    error: Optional[str] = None


def _get_domain(url: str) -> str:
    """Extract domain from URL for rate limiting."""
    return urlparse(url).netloc


def _rate_limit(domain: str) -> None:
    """Enforce a small, jittered interval between requests to same domain."""
    now = time.time()
    last = _last_request_time.get(domain, 0)
    elapsed = now - last

    target_interval = _MIN_REQUEST_INTERVAL + random.uniform(0, _MAX_REQUEST_JITTER)
    if elapsed < target_interval:
        sleep_time = target_interval - elapsed
        logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s for {domain}")
        time.sleep(sleep_time)

    _last_request_time[domain] = time.time()


def _is_github_raw_markdown(url: str) -> bool:
    """Check if URL points to a GitHub raw markdown file."""
    return "raw.githubusercontent.com" in url and url.endswith(".md")


def _github_candidate_urls(url: str) -> list[str]:
    """
    Return candidate fetch URLs, including GitHub raw fallbacks.

    Handles:
    - github.com/user/repo/blob/branch/file.md -> raw.githubusercontent.com/user/repo/branch/file.md
    - github.com/user/repo -> raw.githubusercontent.com/user/repo/main/README.md,
      then raw.githubusercontent.com/user/repo/master/README.md
    """
    if "raw.githubusercontent.com" in url:
        return [url]

    blob_match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)",
        url,
    )
    if blob_match:
        user, repo, branch, path = blob_match.groups()
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
        logger.info(f"Converted GitHub blob to raw: {raw_url}")
        return [raw_url]

    repo_match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/?$",
        url,
    )
    if repo_match:
        user, repo = repo_match.groups()
        candidates = [
            f"https://raw.githubusercontent.com/{user}/{repo}/main/README.md",
            f"https://raw.githubusercontent.com/{user}/{repo}/master/README.md",
        ]
        logger.info(f"Converted GitHub repo root to raw README candidates: {candidates}")
        return candidates

    return [url]


def _robots_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def _can_fetch(url: str) -> tuple[bool, str | None]:
    """Check robots.txt for the URL, failing open if robots.txt is unavailable."""
    parsed = urlparse(url)
    domain_key = f"{parsed.scheme}://{parsed.netloc}"

    if domain_key not in _robots_cache:
        robots = urllib.robotparser.RobotFileParser()
        robots.set_url(_robots_url(url))
        try:
            robots.read()
        except Exception as e:
            logger.warning(f"Could not read robots.txt for {domain_key}: {e}")
            return True, None
        _robots_cache[domain_key] = robots

    allowed = _robots_cache[domain_key].can_fetch(_USER_AGENT, url)
    if not allowed:
        return False, f"Blocked by robots.txt: {url}"
    return True, None


def _looks_like_anti_bot_page(content: str, url: str) -> str | None:
    """Detect common anti-bot, CAPTCHA, and login-wall responses."""
    sample = content[:20000]
    for pattern in _ANTI_BOT_PATTERNS:
        if re.search(pattern, sample):
            return f"Possible anti-bot/login-wall response from {url}"
    return None


def _content_type_for(url: str, response: requests.Response) -> str:
    """Infer whether the fetched body should be treated as HTML or markdown."""
    if _is_github_raw_markdown(url):
        return "markdown"

    content_type = response.headers.get("Content-Type", "")
    if "text/markdown" in content_type:
        return "markdown"
    if "text/plain" in content_type and url.endswith(".md"):
        return "markdown"
    return "html"


def _fetch_once(
    url: str,
    timeout: int,
    respect_robots: bool,
    enforce_robots: bool,
) -> FetchResult:
    """Fetch a URL once without retry orchestration."""
    if respect_robots:
        allowed, robots_error = _can_fetch(url)
        if not allowed:
            if enforce_robots:
                logger.warning(robots_error)
                return FetchResult(
                    url=url,
                    content="",
                    content_type="",
                    status_code=0,
                    success=False,
                    error=robots_error,
                )
            logger.warning(f"{robots_error}; continuing because enforce_robots=False")

    _rate_limit(_get_domain(url))

    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/markdown,text/plain;q=0.9,*/*;q=0.8",
    }

    logger.info(f"Fetching: {url}")
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    content = response.text
    anti_bot_error = _looks_like_anti_bot_page(content, url)
    if anti_bot_error:
        logger.warning(anti_bot_error)
        return FetchResult(
            url=url,
            content="",
            content_type="",
            status_code=response.status_code,
            success=False,
            error=anti_bot_error,
        )

    content_type = _content_type_for(url, response)
    logger.info(f"Successfully fetched {len(content)} chars ({content_type})")
    return FetchResult(
        url=url,
        content=content,
        content_type=content_type,
        status_code=response.status_code,
        success=True,
    )


def fetch_source(
    source: dict,
    timeout: int = 30,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    respect_robots: bool = False,
    enforce_robots: bool = False,
) -> FetchResult:
    """
    Fetch content from a source URL.

    Args:
        source: Source dict with at least a 'url' key.
        timeout: Request timeout in seconds.
        max_retries: Number of attempts per candidate URL.
        respect_robots: If True, check robots.txt before fetching.
        enforce_robots: If True, fail when robots.txt disallows a URL.

    Returns:
        FetchResult with content and metadata.
    """
    original_url = source["url"]
    candidate_urls = _github_candidate_urls(original_url)
    last_result = FetchResult(
        url=original_url,
        content="",
        content_type="",
        status_code=0,
        success=False,
        error="No fetch attempted",
    )

    for candidate_url in candidate_urls:
        for attempt in range(1, max_retries + 1):
            try:
                result = _fetch_once(
                    candidate_url,
                    timeout,
                    respect_robots,
                    enforce_robots,
                )
                if result.success:
                    return result

                last_result = result
                break

            except requests.exceptions.Timeout:
                error_msg = f"Timeout after {timeout}s"
                status_code = 0

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else 0
                reason = e.response.reason if e.response is not None else "unknown"
                error_msg = f"HTTP {status_code}: {reason}"

                if status_code == 404 and "raw.githubusercontent.com" in candidate_url:
                    logger.warning(f"GitHub raw candidate not found: {candidate_url}")
                    last_result = FetchResult(
                        url=candidate_url,
                        content="",
                        content_type="",
                        status_code=status_code,
                        success=False,
                        error=error_msg,
                    )
                    break

                if status_code not in _RETRYABLE_STATUS_CODES:
                    logger.warning(f"Failed to fetch {candidate_url}: {error_msg}")
                    return FetchResult(
                        url=candidate_url,
                        content="",
                        content_type="",
                        status_code=status_code,
                        success=False,
                        error=error_msg,
                    )

            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error: {str(e)[:100]}"
                status_code = 0

            except RequestException as e:
                error_msg = f"Request failed: {str(e)[:100]}"
                status_code = 0

            last_result = FetchResult(
                url=candidate_url,
                content="",
                content_type="",
                status_code=status_code,
                success=False,
                error=error_msg,
            )

            if attempt >= max_retries:
                logger.warning(f"Failed to fetch {candidate_url}: {error_msg}")
                break

            sleep_time = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                f"Fetch attempt {attempt}/{max_retries} failed for {candidate_url}: "
                f"{error_msg}; retrying in {sleep_time:.2f}s"
            )
            time.sleep(sleep_time)

    return last_result