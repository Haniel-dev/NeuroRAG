"""FAISS-backed vector store for local development."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from scraper.base import Article

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    Flat IP (inner product) index over L2-normalised embeddings →
    equivalent to cosine similarity search.
    """

    def __init__(self, dimension: int, index_path: str = "./data/faiss_index"):
        self.dimension  = dimension
        self.index_path = Path(index_path)
        self._meta_path = self.index_path.with_suffix(".meta.json")

        self.index:    faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.metadata: List[dict]        = []   # parallel list to FAISS vectors

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def add(self, embeddings: np.ndarray, articles: List[Article]) -> None:
        assert len(embeddings) == len(articles), "Mismatch between embeddings and articles"
        self.index.add(embeddings)
        self.metadata.extend([a.to_dict() for a in articles])
        logger.info("FAISS: index now contains %d vectors", self.index.ntotal)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[dict, float]]:
        """Return list of (article_dict, score) sorted by descending score."""
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, ensure_ascii=False)
        logger.info("FAISS index saved → %s", self.index_path)

    def load(self) -> bool:
        if not self.index_path.exists():
            logger.warning("FAISS index not found at %s", self.index_path)
            return False
        self.index    = faiss.read_index(str(self.index_path))
        with open(self._meta_path, encoding="utf-8") as fh:
            self.metadata = json.load(fh)
        logger.info("FAISS index loaded: %d vectors", self.index.ntotal)
        return True
