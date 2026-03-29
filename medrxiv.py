"""MedRxiv / BioRxiv scraper using the official REST API."""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import List

from .base import Article, BaseScraper

logger = logging.getLogger(__name__)

# medrxiv/biorxiv share the same API structure
MEDRXIV_API = "https://api.medrxiv.org/details/medrxiv"


class MedrxivScraper(BaseScraper):
    """
    Uses the MedRxiv API which returns JSON paged by date intervals.
    We walk backwards in 30-day windows until max_results is reached.
    """

    def scrape(self, query: str = "") -> List[Article]:
        articles = self._fetch_recent()
        # Filter by neurodivergent relevance
        keywords = {t.lower() for t in self.NEURODIVERGENT_TERMS}
        filtered = [
            a for a in articles
            if any(kw in (a.title + " " + a.abstract).lower() for kw in keywords)
        ]
        logger.info("MedRxiv: %d articles after keyword filter", len(filtered))
        return filtered[: self.max_results]

    def _fetch_recent(self) -> List[Article]:
        articles: List[Article] = []
        end   = date.today()
        start = end - timedelta(days=365)   # 1 year window

        cursor = 0
        while len(articles) < self.max_results:
            url = f"{MEDRXIV_API}/{start}/{end}/{cursor}/json"
            try:
                data = self._get(url).json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("MedRxiv fetch error: %s", exc)
                break

            collection = data.get("collection", [])
            if not collection:
                break

            for item in collection:
                articles.append(self._to_article(item))

            cursor += len(collection)
            total = int(data.get("messages", [{}])[0].get("total", 0))
            if cursor >= total:
                break
            time.sleep(0.5)

        return articles

    @staticmethod
    def _to_article(item: dict) -> Article:
        authors_raw = item.get("authors", "")
        authors = [a.strip() for a in authors_raw.split(";") if a.strip()]

        doi = item.get("doi", "")
        url = f"https://www.medrxiv.org/content/{doi}v{item.get('version', 1)}" if doi else ""

        return Article(
            title=item.get("title", "").strip(),
            abstract=item.get("abstract", "").strip(),
            authors=authors,
            source="medrxiv",
            url=url,
            published_date=item.get("date", ""),
            doi=doi,
        )
