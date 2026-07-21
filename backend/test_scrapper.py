"""
Quick test of fetch + clean on a few sources.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from scraper.sources import SOURCES
from scraper.fetch import fetch_source
from scraper.clean import clean_content


def test_fetch_and_clean():
    # Pick 3 diverse sources that are known-good
    # test_sources = [
    #     SOURCES[0],   # PostgreSQL indexes
    #     SOURCES[15],  # Stripe idempotent requests
    #     SOURCES[21],  # OWASP BOLA
    # ]

    # for source in test_sources:
    for source in SOURCES:
        print("\n" + "=" * 70)
        print(f"TESTING: {source['title']}")
        print(f"URL: {source['url']}")
        print("=" * 70)

        result = fetch_source(source)
        if not result.success:
            print(f"❌ FETCH FAILED: {result.error}")
            continue
        print(f"✅ Fetched {len(result.content)} chars ({result.content_type})")

        cleaned = clean_content(result.content, result.content_type)
        print(f"✅ Cleaned to {len(cleaned)} chars")

        print("\n--- PREVIEW (first 1500 chars) ---")
        print(cleaned[:1500])
        print("...\n")

        if len(cleaned) < 100:
            print("⚠️  WARNING: Very short output")
        if "```" in cleaned:
            print("✅ Contains code blocks")
        if "#" in cleaned:
            print("✅ Contains headings")


if __name__ == "__main__":
    test_fetch_and_clean()