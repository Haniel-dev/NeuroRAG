"""Scraper package — run all sources and deduplicate."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List

from .arxiv import ArxivScraper
from .base import Article
from .medrxiv import MedrxivScraper
from .pubmed import PubMedScraper
from .semantic_scholar import SemanticScholarScraper

logger = logging.getLogger(__name__)

__all__ = [
    "Article",
    "PubMedScraper",
    "ArxivScraper",
    "MedrxivScraper",
    "SemanticScholarScraper",
    "run_all_scrapers",
]


def run_all_scrapers(
    pubmed_email: str,
    pubmed_api_key: str = "",
    max_results: int = 200,
    output_path: str = "./data/articles.json",
) -> List[Article]:
    """Run every scraper, deduplicate by DOI/title, persist to JSON."""
    scrapers = [
        PubMedScraper(email=pubmed_email, api_key=pubmed_api_key, max_results=max_results),
        ArxivScraper(max_results=max_results),
        MedrxivScraper(max_results=max_results),
        SemanticScholarScraper(max_results=max_results),
    ]

    all_articles: List[Article] = []
    for scraper in scrapers:
        name = scraper.__class__.__name__
        logger.info("Running %s …", name)
        try:
            arts = scraper.scrape()
            logger.info("%s → %d articles", name, len(arts))
            all_articles.extend(arts)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s failed: %s", name, exc)

    # Deduplicate by DOI first, then by normalised title
    seen_dois: set[str]   = set()
    seen_titles: set[str] = set()
    unique: List[Article] = []
    for art in all_articles:
        key = art.doi.lower().strip() if art.doi else None
        title_key = art.title.lower().strip()
        if key and key in seen_dois:
            continue
        if title_key in seen_titles:
            continue
        if key:
            seen_dois.add(key)
        seen_titles.add(title_key)
        unique.append(art)

    logger.info("Total unique articles: %d", len(unique))

    # Persist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump([a.to_dict() for a in unique], fh, ensure_ascii=False, indent=2)
    logger.info("Saved to %s", output_path)

    return unique
