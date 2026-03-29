#!/usr/bin/env python
"""
scripts/build_index.py
──────────────────────
End-to-end pipeline:
  1. Scrape articles from PubMed, ArXiv, MedRxiv, Semantic Scholar
  2. Embed with sentence-transformers
  3. Index in FAISS (local) or Pinecone (cloud)

Usage:
  python scripts/build_index.py                          # FAISS (uses .env)
  VECTOR_DB_MODE=pinecone python scripts/build_index.py  # Pinecone
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")

from config import settings
from embeddings import EmbeddingEngine
from scraper import Article, run_all_scrapers
from vectordb import FAISSStore, get_vector_store


def main():
    parser = argparse.ArgumentParser(description="Build NeuroRAG index")
    parser.add_argument("--max-results", type=int, default=settings.max_results_per_source)
    parser.add_argument("--articles-json", default="./data/articles.json",
                        help="Path to cached articles JSON (skip scraping if exists)")
    parser.add_argument("--force-scrape", action="store_true",
                        help="Re-scrape even if articles JSON exists")
    args = parser.parse_args()

    articles_path = Path(args.articles_json)

    # ── 1. Scrape ─────────────────────────────────────────────────────── #
    if articles_path.exists() and not args.force_scrape:
        logger.info("Loading existing articles from %s …", articles_path)
        with open(articles_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        articles = [Article(**r) for r in raw]
    else:
        logger.info("Starting scraping pipeline (max %d per source) …", args.max_results)
        articles = run_all_scrapers(
            pubmed_email=settings.pubmed_email,
            pubmed_api_key=settings.pubmed_api_key,
            max_results=args.max_results,
            output_path=str(articles_path),
        )

    if not articles:
        logger.error("No articles collected — aborting.")
        sys.exit(1)

    logger.info("Total articles to index: %d", len(articles))

    # ── 2. Embed ──────────────────────────────────────────────────────── #
    engine = EmbeddingEngine(settings.embedding_model)
    texts  = [a.text_for_embedding for a in articles]

    logger.info("Encoding %d texts …", len(texts))
    embeddings = engine.encode(texts, show_progress=True)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # ── 3. Index ──────────────────────────────────────────────────────── #
    store = get_vector_store(settings)
    store.add(embeddings, articles)

    if settings.vector_db_mode == "faiss":
        store.save()
        logger.info("FAISS index saved to %s", settings.faiss_index_path)
    else:
        logger.info("Vectors upserted to Pinecone index '%s'", settings.pinecone_index_name)

    logger.info("✅ Index build complete!")


if __name__ == "__main__":
    main()
