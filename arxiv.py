"""ArXiv scraper using the official Atom API."""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from typing import List
from urllib.parse import urlencode

from .base import Article, BaseScraper

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

CATEGORIES = ["q-bio.NC", "cs.AI", "cs.LG", "eess.SP"]  # neuroscience + AI


class ArxivScraper(BaseScraper):
    def scrape(self, query: str = "") -> List[Article]:
        if not query:
            query = (
                "ti:ADHD OR ti:autism OR ti:dyslexia OR ti:neurodivergent "
                "OR ti:\"attention deficit\" OR ti:\"autism spectrum\""
            )
        return self._fetch(query)

    def _fetch(self, query: str) -> List[Article]:
        articles: List[Article] = []
        batch = 100
        for start in range(0, self.max_results, batch):
            params = {
                "search_query": query,
                "start": start,
                "max_results": min(batch, self.max_results - start),
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            url = f"{ARXIV_API}?{urlencode(params)}"
            xml_text = self._get(url).text
            batch_arts = self._parse(xml_text)
            articles.extend(batch_arts)
            if len(batch_arts) < batch:
                break
            time.sleep(3)  # ArXiv rate limit

        logger.info("ArXiv: collected %d articles", len(articles))
        return articles

    @staticmethod
    def _parse(xml_text: str) -> List[Article]:
        articles: List[Article] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("ArXiv XML parse error: %s", exc)
            return articles

        for entry in root.findall("atom:entry", NS):
            try:
                title    = (entry.findtext("atom:title", "", NS) or "").replace("\n", " ").strip()
                abstract = (entry.findtext("atom:summary", "", NS) or "").replace("\n", " ").strip()
                published = (entry.findtext("atom:published", "", NS) or "")[:10]
                url      = entry.findtext("atom:id", "", NS).strip()

                authors = [
                    a.findtext("atom:name", "", NS).strip()
                    for a in entry.findall("atom:author", NS)
                ]

                doi_node = entry.find("arxiv:doi", NS)
                doi = doi_node.text.strip() if doi_node is not None and doi_node.text else ""

                if title and abstract:
                    articles.append(Article(
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        source="arxiv",
                        url=url,
                        published_date=published,
                        doi=doi,
                    ))
            except Exception as exc:  # noqa: BLE001
                logger.debug("ArXiv entry skip: %s", exc)

        return articles
