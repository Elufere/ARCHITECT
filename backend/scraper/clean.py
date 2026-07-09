"""
Clean and normalize fetched content.

Handles:
- HTML → readable markdown/text
- Removing nav, footers, ads, boilerplate
- Normalizing whitespace
- Passing through raw markdown unchanged
"""

from __future__ import annotations

import re
import logging

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Tags to completely remove (not their content, the tags themselves)
TAGS_TO_REMOVE = [
    "script", "style", "noscript", "iframe", "svg",
    "nav", "header", "footer", "aside",
    "button", "input", "select", "textarea", "form",
]

# Tags whose content should be removed entirely
TAGS_TO_STRIP_WITH_CONTENT = [
    "nav", "footer", "aside", ".sidebar", ".navigation", ".menu",
    "#footer", "#sidebar", "#navigation", "#menu",
    ".ad", ".ads", ".advertisement", "#ad", "#ads",
    ".cookie-banner", "#cookie-banner", ".cookie-notice",
    ".popup", ".modal", ".overlay",
    ".social-share", ".share-buttons",
]

# Elements that are likely boilerplate (by class/id patterns)
BOILERPLATE_PATTERNS = [
    r"(?i)cookie", r"(?i)consent", r"(?i)banner",
    r"(?i)popup", r"(?i)modal", r"(?i)overlay",
    r"(?i)social.*share", r"(?i)share.*button",
    r"(?i)newsletter", r"(?i)subscribe",
    r"(?i)advertisement", r"(?i)sponsored",
    r"(?i)version.*selector", r"(?i)supported.?versions",
    r"(?i)unsupported.?versions", r"(?i)development.?versions",
    r"(?i)breadcrumb", r"(?i)site.?nav", r"(?i)top.?bar",
    r"(?i)page.?nav", r"(?i)doc.?nav",
]

# Lines matching these patterns get stripped from final output
_NOISE_LINE_PATTERNS = [
    # CMS artifacts
    r"(?i)^ALN semantic override",
    r"(?i)^DEFAULT h2 behavior",
    # PostgreSQL nav remnants
    r"^\[Header\]\s*$",
    r"^\[\d+\]\(/docs/",                    # [18](/docs/...)
    r"^\[\d+\.\d+\]\(/docs/",              # [9.6](/docs/...)
    r"^\[devel\]\(/docs/",
    r"^\[Current\]\s*$",
    r"^\[Documentation\]\s*$",
    r"(?i)^Supported Versions:?\s*$",
    r"(?i)^Development Versions:?\s*$",
    r"(?i)^Unsupported versions:?\s*$",
    r"^\(\s*$",                              # stray open paren
    r"^\)\s*$",                              # stray close paren
    r"^\[\s*$",                              # stray open bracket
    r"^\]\s*$",                              # stray close bracket
    # PostgreSQL chapter navigation table
    r"^\|\s*Chapter \d+",
    r"^\|\s*---\s*\|\s*---\s*\|\s*$",       # separator row from nav table
    # Generic nav debris
    r"^\s*[→/]\s*$",
    r"^\|\s*$",                              # empty table row
        # PostgreSQL trailing nav
    r"^\|\s*Prev\s*\|\s*Up\s*\|\s*Next\s*\|",
    r"(?i)^## Submit correction",
    r"(?i)^If you see anything in the documentation",
    r"(?i)^does not match",
    r"(?i)^you can also",
    r"^\|\s*\d+\.\d+\.\s*SELECT",              # Prev/Next target like "10.6. SELECT"
    r"^\|\s*Home\s*\|",                        # Home link in nav
    r"^\|\s*\d+\.\d+\.\s*Introduction\s*\|",   # Next target like "11.1. Introduction"
        # PostgreSQL trailing footer
    r"(?i)your experience with the particular feature",
    r"(?i)please use.*report a documentation issue",
    r"(?i)does not match\s*$",
    r"(?i)requires further clarification",
    r"^pgContentWrap$",
    r"^Footer$",
]


def _is_boilerplate(element: Tag) -> bool:
    """Check if an element is likely boilerplate based on class/id."""
    if element.attrs is None:
        return False

    classes = " ".join(element.attrs.get("class", []))
    element_id = element.attrs.get("id", "")
    combined = f"{classes} {element_id}"

    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, combined):
            return True
    return False


def _remove_unwanted_elements(soup: BeautifulSoup) -> None:
    """Remove scripts, styles, nav, ads, etc. from the soup."""
    # Remove tags with content
    for tag_name in TAGS_TO_STRIP_WITH_CONTENT:
        for element in soup.find_all(tag_name):
            element.decompose()

    # Remove tags but keep content
    for tag_name in TAGS_TO_REMOVE:
        for element in soup.find_all(tag_name):
            element.decompose()

    # Remove elements matching boilerplate patterns
    for element in soup.find_all(True):
        if _is_boilerplate(element):
            element.decompose()

    # Remove elements that are visually hidden but still in DOM
    for element in soup.find_all(attrs={"aria-hidden": "true"}):
        element.decompose()
    for element in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        element.decompose()


def _extract_main_content(soup: BeautifulSoup) -> BeautifulSoup:
    """Try to find the main content area, falling back to body."""
    
    main_selectors = [
        "main",
        "article",
        '[role="main"]',
        "div.doc",               
        ".doc",
        ".page-content",
        ".documentation",
        "#documentation",
        ".doc-content",
        ".markdown-body",        
        ".rst-content",        
        ".content",              
        "#content",
        ".main-content",
        "#main-content",
        ".post-content",
        ".entry-content",
    ]

    for selector in main_selectors:
        main = soup.select_one(selector)
        if main:
            logger.debug(f"Found main content with selector: {selector}")
            return main

    return soup.find("body") or soup


def _convert_to_markdown(element) -> str:
    """Convert a BeautifulSoup element to simple markdown."""
    lines: list[str] = []
    _walk_element(element, lines, heading_level=0)
    return "\n".join(lines)


def _walk_element(element, lines: list[str], heading_level: int) -> None:
    """Recursively walk the element tree and build markdown."""
    if isinstance(element, str):
        text = element.strip()
        if text:
            lines.append(text)
        return

    if not hasattr(element, "name"):
        return

    tag = element.name.lower() if element.name else ""

    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(tag[1])
        text = element.get_text(strip=True)
        if text:
            lines.append("")
            lines.append(f"{'#' * level} {text}")
            lines.append("")

    elif tag == "p":
        text = element.get_text(strip=True)
        if text:
            lines.append("")
            lines.append(text)
            lines.append("")

    elif tag in ("ul", "ol"):
        for i, li in enumerate(element.find_all("li", recursive=False)):
            text = li.get_text(strip=True)
            if text:
                prefix = f"{i + 1}. " if tag == "ol" else "- "
                lines.append(f"{prefix}{text}")

    elif tag == "pre":
        code = element.find("code")
        if code:
            text = code.get_text()
        else:
            text = element.get_text()
        lines.append("")
        lines.append("```")
        lines.append(text.rstrip())
        lines.append("```")
        lines.append("")

    elif tag == "code" and element.parent and element.parent.name != "pre":
        text = element.get_text(strip=True)
        if text:
            lines.append(f"`{text}`")

    elif tag == "a":
        href = element.get("href", "")
        text = element.get_text(strip=True)
        if text and href:
            lines.append(f"[{text}]({href})")
        elif text:
            lines.append(text)

    elif tag in ("strong", "b"):
        text = element.get_text(strip=True)
        if text:
            lines.append(f"**{text}**")

    elif tag in ("em", "i"):
        text = element.get_text(strip=True)
        if text:
            lines.append(f"*{text}*")

    elif tag == "table":
        _convert_table(element, lines)

    elif tag == "blockquote":
        text = element.get_text(strip=True)
        if text:
            for line in text.split("\n"):
                lines.append(f"> {line}")
            lines.append("")

    elif tag == "hr":
        lines.append("")
        lines.append("---")
        lines.append("")

    else:
        for child in element.children:
            _walk_element(child, lines, heading_level)


def _convert_table(table_element, lines: list[str]) -> None:
    """Convert an HTML table to markdown table format."""
    rows = table_element.find_all("tr")
    if not rows:
        return

    lines.append("")

    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        cell_texts = [cell.get_text(strip=True).replace("|", "\\|") for cell in cells]
        line = "| " + " | ".join(cell_texts) + " |"
        lines.append(line)

        if i == 0 and row.find("th"):
            separator = "| " + " | ".join(["---"] * len(cells)) + " |"
            lines.append(separator)

    lines.append("")


def _remove_noise_lines(text: str) -> str:
    """Strip lines that match known noise patterns."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        is_noise = False
        for pattern in _NOISE_LINE_PATTERNS:
            if re.search(pattern, line):
                is_noise = True
                break
        if not is_noise:
            cleaned.append(line)
    return "\n".join(cleaned)


def _strip_leading_noise(text: str) -> str:
    """
    If content starts with noise before the first real heading,
    remove everything up to and including the first '#' heading.
    
    This handles cases where nav elements leak into the extracted
    content area despite selector-based extraction.
    """
    lines = text.split("\n")
    
    # Find first heading line
    first_heading_idx = None
    for i, line in enumerate(lines):
        if line.startswith("#"):
            first_heading_idx = i
            break
    
    # If no heading found, or heading is already first, return as-is
    if first_heading_idx is None or first_heading_idx == 0:
        return text
    
    # Check if lines before the heading are noise (no long sentences)
    prefix_lines = lines[:first_heading_idx]
    noise_indicators = 0
    for line in prefix_lines:
        stripped = line.strip()
        if not stripped:
            continue  # blank lines are neutral
        # Noise: short lines, mostly links, no real prose
        if len(stripped) < 80 and ("[" in stripped or "|" in stripped or stripped in ("(", ")")):
            noise_indicators += 1
        elif len(stripped) < 30:
            noise_indicators += 1
    
    # If majority of non-blank prefix lines are noise, strip them
    non_blank = [l for l in prefix_lines if l.strip()]
    if non_blank and noise_indicators >= len(non_blank) * 0.6:
        logger.debug(f"Stripping {first_heading_idx} leading noise lines")
        return "\n".join(lines[first_heading_idx:])
    
    return text

def _normalize_whitespace(text: str) -> str:
    """Clean up whitespace in the final output."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = text.strip()
    return text


def clean_html(html: str, base_url: str = "") -> str:
    """Convert HTML to clean markdown."""
    soup = BeautifulSoup(html, "html.parser")

    _remove_unwanted_elements(soup)
    main_content = _extract_main_content(soup)
    markdown = _convert_to_markdown(main_content)
    markdown = _remove_noise_lines(markdown)
    markdown = _strip_leading_noise(markdown)
    markdown = _normalize_whitespace(markdown)
    return markdown


def clean_content(content: str, content_type: str, base_url: str = "") -> str:
    """Clean content based on its type."""
    if content_type == "markdown":
        return _normalize_whitespace(content)
    elif content_type == "html":
        return clean_html(content, base_url)
    else:
        logger.warning(f"Unknown content type '{content_type}', attempting HTML parse")
        try:
            return clean_html(content, base_url)
        except Exception:
            return _normalize_whitespace(content)