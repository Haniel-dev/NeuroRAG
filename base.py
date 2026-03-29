"""Base scraper with retry logic and shared data model."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


@dataclass
class Article:
    """Unified article representation across all sources."""
    title: str
    abstract: str
    authors: List[str]
    source: str               # "pubmed" | "arxiv" | "medrxiv" | "semantic_scholar"
    url: str
    published_date: str       # ISO 8601
    doi: str = ""
    keywords: List[str] = field(default_factory=list)
    full_text: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "source": self.source,
            "url": self.url,
            "published_date": self.published_date,
            "doi": self.doi,
            "keywords": self.keywords,
            "full_text": self.full_text,
        }

    @property
    def text_for_embedding(self) -> str:
        """Concatenate title + abstract for embedding."""
        return f"{self.title}. {self.abstract}"


class BaseScraper(ABC):
    NEURODIVERGENT_TERMS = [
        "ADHD", "attention deficit hyperactivity disorder",
        "autism spectrum disorder", "ASD", "autism",
        "dyslexia", "dyscalculia", "dyspraxia",
        "sensory processing disorder", "executive function",
        "working memory", "neurodivergent", "neurodiversity",
        "Tourette syndrome", "OCD neurodevelopmental",
        "learning disability", "developmental coordination disorder",
    ]

    def __init__(self, max_results: int = 200):
        self.max_results = max_results
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NeuroRAG/1.0 (research project)"})

    @abstractmethod
    def scrape(self, query: str = "") -> List[Article]:
        """Scrape articles and return unified Article list."""
        ...

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _get(self, url: str, **kwargs) -> requests.Response:
        resp = self.session.get(url, timeout=15, **kwargs)
        resp.raise_for_status()
        return resp

    def build_query(self, extra_terms: List[str] | None = None) -> str:
        terms = self.NEURODIVERGENT_TERMS[:]
        if extra_terms:
            terms.extend(extra_terms)
        return " OR ".join(f'"{t}"' for t in terms[:8])  # keep query size sane
