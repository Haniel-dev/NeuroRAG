"""Pinecone vector store for cloud deployment."""
from __future__ import annotations

import logging
import time
from typing import List, Tuple

import numpy as np
from pinecone import Pinecone, ServerlessSpec

from scraper.base import Article

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


class PineconeStore:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        environment: str = "us-east-1",
    ):
        self.pc         = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension  = dimension
        self._ensure_index(environment)
        self.index = self.pc.Index(index_name)

    # ------------------------------------------------------------------ #
    def _ensure_index(self, environment: str) -> None:
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            logger.info("Creating Pinecone index '%s' …", self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=environment),
            )
            # Wait until ready
            while True:
                info = self.pc.describe_index(self.index_name)
                if info.status.get("ready", False):
                    break
                time.sleep(2)
        else:
            logger.info("Pinecone index '%s' already exists", self.index_name)

    # ------------------------------------------------------------------ #
    def add(self, embeddings: np.ndarray, articles: List[Article]) -> None:
        vectors = []
        for i, (emb, art) in enumerate(zip(embeddings, articles)):
            meta = {k: v for k, v in art.to_dict().items() if k != "full_text"}
            # Pinecone metadata values must be str/int/float/bool/list[str]
            meta["authors"]  = meta.get("authors", [])[:10]
            meta["keywords"] = meta.get("keywords", [])[:20]
            vectors.append({
                "id":       f"{art.source}_{i}_{hash(art.title) & 0xFFFFFF}",
                "values":   emb.tolist(),
                "metadata": meta,
            })

        for start in range(0, len(vectors), BATCH_SIZE):
            self.index.upsert(vectors=vectors[start : start + BATCH_SIZE])
            logger.debug("Pinecone: upserted batch %d–%d", start, start + BATCH_SIZE)

        logger.info("Pinecone: upserted %d vectors", len(vectors))

    # ------------------------------------------------------------------ #
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[dict, float]]:
        response = self.index.query(
            vector=query_vector[0].tolist(),
            top_k=top_k,
            include_metadata=True,
        )
        return [(match.metadata, match.score) for match in response.matches]
