"""Embedding engine using sentence-transformers."""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Wraps a SentenceTransformer and exposes encode helpers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self.dimension)

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Return (N, D) float32 array of L2-normalised embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # cosine sim via dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Return (1, D) array for a single query string."""
        return self.encode([text], show_progress=False)
