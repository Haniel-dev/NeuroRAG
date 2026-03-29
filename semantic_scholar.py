"""Semantic Scholar scraper using the S2 Academic Graph API."""
from __future__ import annotations

import logging
import time
from typing import List

from .base import Article, BaseScraper

logger = logging.getLogger(__name__)

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,authors,year,externalIds,url,publicationDate"


class SemanticScholarScraper(BaseScraper):
    def scrape(self, query: str = "") -> List[Article]:
        if not query:
            query = (
                "ADHD autism dyslexia neurodivergent sensory processing "
                "executive function working memory"
            )
        return self._search(query)

    def _search(self, query: str) -> List[Article]:
        articles: List[Article] = []
        offset, limit = 0, 100

        while len(articles) < self.max_results:
            params = {
                "query": query,
                "fields": FIELDS,
                "offset": offset,
                "limit": min(limit, self.max_results - len(articles)),
            }
            try:
                data = self._get(S2_SEARCH, params=params).json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("S2 fetch error: %s", exc)
                break

            batch = data.get("data", [])
            if not batch:
                break

            for paper in batch:
                art = self._to_article(paper)
                if art.abstract:
                    articles.append(art)

            offset += len(batch)
            total  = data.get("total", 0)
            if offset >= total:
                break

            time.sleep(1)  # S2 rate limit: ~1 req/s unauthenticated

        logger.info("Semantic Scholar: collected %d articles", len(articles))
        return articles

    @staticmethod
    def _to_article(paper: dict) -> Article:
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        year    = str(paper.get("year", ""))
        pub_date = paper.get("publicationDate") or (f"{year}-01-01" if year else "")
        external = paper.get("externalIds", {})
        doi  = external.get("DOI", "")
        url  = paper.get("url") or (
            f"https://doi.org/{doi}" if doi else ""
        )

        return Article(
            title=paper.get("title", "").strip(),
            abstract=(paper.get("abstract") or "").strip(),
            authors=authors,
            source="semantic_scholar",
            url=url,
            published_date=pub_date,
            doi=doi,
        )
