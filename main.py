"""FastAPI backend — RAG query + index management endpoints."""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make project root importable when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from embeddings import EmbeddingEngine
from vectordb import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Globals (populated at startup) ────────────────────────────────────── #
embed_engine: Optional[EmbeddingEngine] = None
vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_engine, vector_store
    logger.info("Loading embedding model …")
    embed_engine = EmbeddingEngine(settings.embedding_model)
    logger.info("Connecting to vector store (mode=%s) …", settings.vector_db_mode)
    vector_store = get_vector_store(settings)
    yield
    logger.info("Shutting down …")


app = FastAPI(
    title="Neurodivergent RAG API",
    description="Semantic search over neurodivergent scientific literature.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────── #
class ArticleResult(BaseModel):
    title: str
    abstract: str
    authors: List[str]
    source: str
    url: str
    published_date: str
    doi: str
    score: float


class QueryResponse(BaseModel):
    query: str
    results: List[ArticleResult]
    total: int


class IndexStats(BaseModel):
    mode: str
    total_vectors: int


# ── Routes ────────────────────────────────────────────────────────────── #
@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "service": "NeuroRAG"}


@app.get("/search", response_model=QueryResponse, tags=["Search"])
def search(
    q: str = Query(..., min_length=3, description="Natural-language query"),
    top_k: int = Query(default=5, ge=1, le=20),
):
    """Semantic search over the neurodivergent literature corpus."""
    if embed_engine is None or vector_store is None:
        raise HTTPException(503, "Service not ready — index loading")

    query_vec = embed_engine.encode_single(q)
    raw = vector_store.search(query_vec, top_k=top_k)

    results = []
    for meta, score in raw:
        results.append(ArticleResult(
            title=meta.get("title", ""),
            abstract=meta.get("abstract", ""),
            authors=meta.get("authors", []),
            source=meta.get("source", ""),
            url=meta.get("url", ""),
            published_date=meta.get("published_date", ""),
            doi=meta.get("doi", ""),
            score=round(score, 4),
        ))

    return QueryResponse(query=q, results=results, total=len(results))


@app.get("/stats", response_model=IndexStats, tags=["Admin"])
def stats():
    """Return index size and current mode."""
    if vector_store is None:
        raise HTTPException(503, "Vector store not ready")

    total = 0
    if hasattr(vector_store, "index"):          # FAISS
        total = vector_store.index.ntotal
    elif hasattr(vector_store, "index_name"):   # Pinecone
        try:
            info = vector_store.index.describe_index_stats()
            total = info.total_vector_count
        except Exception:
            total = -1

    return IndexStats(mode=settings.vector_db_mode, total_vectors=total)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=True)
