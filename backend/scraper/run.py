"""
Scraper pipeline entrypoint.

Loops through all sources, fetches + cleans each,
writes results to knowledge_base/.okf/<category>/<slug>.md
with YAML frontmatter.

Usage:
    python -m scraper.run
    python -m scraper.run --category standards
    python -m scraper.run --dry-run
    python -m scraper.run --max-retries 5
    python -m scraper.run --check-robots
    python -m scraper.run --enforce-robots
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from scraper.sources import SOURCES, get_all_categories
from scraper.fetch import fetch_source
from scraper.clean import clean_content

logger = logging.getLogger(__name__)

# Output root: backend/knowledge_base/.okf/
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "knowledge_base" / ".okf"

def _build_frontmatter(source: dict) -> str:
    """Build YAML frontmatter string for a source."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "---",
        f'source_url: "{source["url"]}"',
        f'category: "{source["category"]}"',
        f'title: "{source["title"]}"',
        f'slug: "{source["slug"]}"',
        f'date_scraped: "{date_str}"',
    ]

    if "description" in source:
        # Escape internal double quotes for valid YAML
        desc = source["description"].replace('"', '\\"')
        lines.append(f'description: "{desc}"')

    lines.append("---")
    lines.append("")  # blank line after frontmatter
    return "\n".join(lines)


def process_source(
    source: dict,
    overwrite: bool = True,
    max_retries: int = 3,
    respect_robots: bool = False,
    enforce_robots: bool = False,
) -> dict:
    """
    Process a single source: fetch → clean → write.

    Returns a result dict with status info.
    """
    slug = source["slug"]
    category = source["category"]
    url = source["url"]

    result = {
        "slug": slug,
        "category": category,
        "url": url,
        "title": source.get("title", slug),
        "status": "unknown",
        "error": None,
        "chars_fetched": 0,
        "chars_cleaned": 0,
        "output_path": None,
    }

    # Determine output path
    output_dir = OUTPUT_ROOT / category
    output_path = output_dir / f"{slug}.md"
    result["output_path"] = str(output_path)

    # Check if already exists
    if output_path.exists() and not overwrite:
        result["status"] = "skipped"
        logger.info(f"Skipped (exists): {slug}")
        return result

    # Fetch
    fetch_result = fetch_source(
        source,
        max_retries=max_retries,
        respect_robots=respect_robots,
        enforce_robots=enforce_robots,
    )
    result["chars_fetched"] = len(fetch_result.content)

    if not fetch_result.success:
        result["status"] = "fetch_failed"
        result["error"] = fetch_result.error
        logger.warning(f"Fetch failed: {slug} — {fetch_result.error}")
        return result

    # Clean
    try:
        cleaned = clean_content(fetch_result.content, fetch_result.content_type)
        result["chars_cleaned"] = len(cleaned)
    except Exception as e:
        result["status"] = "clean_failed"
        result["error"] = str(e)[:200]
        logger.error(f"Clean failed: {slug} — {e}")
        return result

    # Quality check — skip if too short (likely didn't extract real content)
    if len(cleaned) < 200:
        result["status"] = "too_short"
        result["error"] = f"Only {len(cleaned)} chars after cleaning"
        logger.warning(f"Too short ({len(cleaned)} chars): {slug}")
        return result

    # Write file
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        frontmatter = _build_frontmatter(source)
        content = frontmatter + cleaned

        output_path.write_text(content, encoding="utf-8")
        result["status"] = "success"
        logger.info(f"Saved ({len(cleaned)} chars): {output_path.name}")
    except Exception as e:
        result["status"] = "write_failed"
        result["error"] = str(e)[:200]
        logger.error(f"Write failed: {slug} — {e}")

    return result


def run_pipeline(
    category: str | None = None,
    overwrite: bool = True,
    dry_run: bool = False,
    max_retries: int = 3,
    respect_robots: bool = False,
    enforce_robots: bool = False,
) -> list[dict]:
    """
    Run the full scraping pipeline.

    Args:
        category: If set, only process sources in this category
        overwrite: If False, skip already-fetched sources
        dry_run: If True, fetch and clean but don't write files
        max_retries: Number of fetch attempts per source
        respect_robots: If True, check robots.txt before fetching
        enforce_robots: If True, fail when robots.txt disallows a URL

    Returns:
        List of result dicts
    """
    sources = SOURCES
    if category:
        sources = [s for s in SOURCES if s["category"] == category]
        logger.info(f"Filtered to category '{category}': {len(sources)} sources")

    logger.info(f"Starting pipeline: {len(sources)} sources")
    if dry_run:
        logger.info("DRY RUN — no files will be written")

    results = []
    for i, source in enumerate(sources, 1):
        logger.info(f"\n[{i}/{len(sources)}] {source['title']}")

        if dry_run:
            # Fetch and clean but don't write
            fetch_result = fetch_source(
                source,
                max_retries=max_retries,
                respect_robots=respect_robots,
                enforce_robots=enforce_robots,
            )
            if fetch_result.success:
                cleaned = clean_content(fetch_result.content, fetch_result.content_type)
                results.append({
                    "slug": source["slug"],
                    "title": source["title"],
                    "status": "dry_run",
                    "chars_fetched": len(fetch_result.content),
                    "chars_cleaned": len(cleaned),
                })
                logger.info(f"Dry run: {len(cleaned)} chars")
            else:
                results.append({
                    "slug": source["slug"],
                    "title": source["title"],
                    "status": "fetch_failed",
                    "error": fetch_result.error,
                })
        else:
            result = process_source(
                source,
                overwrite=overwrite,
                max_retries=max_retries,
                respect_robots=respect_robots,
                enforce_robots=enforce_robots,
            )
            results.append(result)

    return results


def print_summary(results: list[dict]) -> None:
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("SCRAPER SUMMARY")
    print("=" * 70)

    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    total = len(results)
    succeeded = status_counts.get("success", 0) + status_counts.get("dry_run", 0)
    skipped = status_counts.get("skipped", 0)
    failed = total - succeeded - skipped

    print(f"Total:   {total}")
    print(f"Success: {succeeded}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")
    print()

    if failed > 0:
        print("FAILED SOURCES:")
        print("-" * 70)
        for r in results:
            if r["status"] not in ("success", "skipped", "dry_run"):
                title = r.get("title", r.get("slug", "?"))
                error = r.get("error", "unknown")
                print(f" {title}")
                print(f"     {r['status']}: {error}")
        print()

    print("SUCCESSFUL SOURCES:")
    print("-" * 70)
    for r in results:
        if r["status"] in ("success", "skipped", "dry_run"):
            title = r.get("title", r.get("slug", "?"))
            chars = r.get("chars_cleaned", 0)
            status_icon = "" if r["status"] == "skipped" else "" if r["status"] == "dry_run" else ""
            print(f"  {status_icon} {title} ({chars} chars)")


def main():
    parser = argparse.ArgumentParser(description="Architect knowledge base scraper")
    parser.add_argument(
        "--category", "-c",
        choices=get_all_categories(),
        help="Only scrape sources in this category",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Fetch and clean but don't write files",
    )
    parser.add_argument(
        "--skip-existing", "-s",
        action="store_true",
        help="Skip sources that already have output files",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Fetch attempts per source before giving up",
    )
    parser.add_argument(
        "--check-robots",
        action="store_true",
        help="Check robots.txt before fetching, but only warn by default",
    )
    parser.add_argument(
        "--enforce-robots",
        action="store_true",
        help="Check robots.txt and fail sources that it disallows",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Run
    results = run_pipeline(
        category=args.category,
        overwrite=not args.skip_existing,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        respect_robots=args.check_robots or args.enforce_robots,
        enforce_robots=args.enforce_robots,
    )

    # Summary
    print_summary(results)

    # Exit code
    failed = sum(1 for r in results if r["status"] in ("fetch_failed", "clean_failed", "write_failed", "too_short"))
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
