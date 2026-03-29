"""PubMed scraper using NCBI E-utilities (no unofficial libs needed)."""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from typing import List

from .base import Article, BaseScraper

logger = logging.getLogger(__name__)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedScraper(BaseScraper):
    def __init__(self, email: str, api_key: str = "", max_results: int = 200):
        super().__init__(max_results)
        self.email = email
        self.api_key = api_key
        # With API key: 10 req/s; without: 3 req/s
        self._delay = 0.11 if api_key else 0.35

    # ------------------------------------------------------------------ #
    def scrape(self, query: str = "") -> List[Article]:
        if not query:
            query = self.build_query()
        pmids = self._search(query)
        logger.info("PubMed: found %d PMIDs", len(pmids))
        return self._fetch_details(pmids)

    # ------------------------------------------------------------------ #
    def _search(self, query: str) -> List[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self.max_results,
            "retmode": "json",
            "sort": "relevance",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        data = self._get(ESEARCH, params=params).json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _fetch_details(self, pmids: List[str]) -> List[Article]:
        articles: List[Article] = []
        batch_size = 50
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
                "email": self.email,
            }
            if self.api_key:
                params["api_key"] = self.api_key

            xml_text = self._get(EFETCH, params=params).text
            articles.extend(self._parse_xml(xml_text))
            time.sleep(self._delay)

        return articles

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_xml(xml_text: str) -> List[Article]:
        articles: List[Article] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("XML parse error: %s", exc)
            return articles

        for article_node in root.findall(".//PubmedArticle"):
            try:
                medline = article_node.find("MedlineCitation")
                art     = medline.find("Article")

                title = art.findtext("ArticleTitle", default="").strip()

                # Abstract can be structured (multiple AbstractText)
                abstract_parts = [
                    (node.get("Label", "") + " " + (node.text or "")).strip()
                    for node in art.findall(".//AbstractText")
                ]
                abstract = " ".join(abstract_parts).strip()

                authors = [
                    f"{a.findtext('ForeName', '')} {a.findtext('LastName', '')}".strip()
                    for a in art.findall(".//Author")
                    if a.findtext("LastName")
                ]

                # Date
                pub_date = medline.find(".//PubDate")
                year  = pub_date.findtext("Year", "0000") if pub_date is not None else "0000"
                month = pub_date.findtext("Month", "01")  if pub_date is not None else "01"
                published = f"{year}-{month}-01"

                pmid = medline.findtext("PMID", "")
                url  = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                doi_node = article_node.find(".//ArticleId[@IdType='doi']")
                doi = doi_node.text.strip() if doi_node is not None and doi_node.text else ""

                keywords = [
                    kw.text.strip()
                    for kw in medline.findall(".//Keyword")
                    if kw.text
                ]

                if title and abstract:
                    articles.append(Article(
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        source="pubmed",
                        url=url,
                        published_date=published,
                        doi=doi,
                        keywords=keywords,
                    ))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping article due to parse error: %s", exc)

        return articles
