"""Unit and integration tests for NeuroRAG."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Article dataclass ─────────────────────────────────────────────────── #
class TestArticle:
    def test_text_for_embedding(self):
        from scraper.base import Article
        art = Article(
            title="ADHD and working memory",
            abstract="A study on executive function.",
            authors=["Smith J"],
            source="pubmed",
            url="https://pubmed.ncbi.nlm.nih.gov/123",
            published_date="2023-01-01",
        )
        assert "ADHD" in art.text_for_embedding
        assert "executive function" in art.text_for_embedding

    def test_to_dict_keys(self):
        from scraper.base import Article
        art = Article(
            title="Test", abstract="Abst", authors=[], source="arxiv",
            url="http://example.com", published_date="2024-01-01",
        )
        d = art.to_dict()
        for key in ("title", "abstract", "authors", "source", "url", "published_date", "doi"):
            assert key in d


# ── Embedding engine ──────────────────────────────────────────────────── #
class TestEmbeddingEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        from embeddings import EmbeddingEngine
        return EmbeddingEngine("all-MiniLM-L6-v2")

    def test_encode_shape(self, engine):
        vecs = engine.encode(["Hello world", "ADHD research"])
        assert vecs.shape == (2, 384)

    def test_encode_normalised(self, engine):
        vecs = engine.encode(["Some text"])
        norm = np.linalg.norm(vecs[0])
        assert abs(norm - 1.0) < 1e-5

    def test_encode_single(self, engine):
        vec = engine.encode_single("autism spectrum")
        assert vec.shape == (1, 384)


# ── FAISS store ───────────────────────────────────────────────────────── #
class TestFAISSStore:
    def test_add_and_search(self, tmp_path):
        from scraper.base import Article
        from vectordb.faiss_store import FAISSStore

        store = FAISSStore(dimension=4, index_path=str(tmp_path / "idx"))
        arts  = [
            Article(title="A", abstract="aa", authors=[], source="pubmed",
                    url="http://a.com", published_date="2023-01-01"),
            Article(title="B", abstract="bb", authors=[], source="arxiv",
                    url="http://b.com", published_date="2023-01-01"),
        ]
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        store.add(vecs, arts)

        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = store.search(query, top_k=1)
        assert len(results) == 1
        assert results[0][0]["title"] == "A"

    def test_save_and_load(self, tmp_path):
        from scraper.base import Article
        from vectordb.faiss_store import FAISSStore

        store1 = FAISSStore(dimension=4, index_path=str(tmp_path / "idx2"))
        art = Article(title="C", abstract="cc", authors=[], source="medrxiv",
                      url="http://c.com", published_date="2023-01-01")
        store1.add(np.array([[0, 0, 1, 0]], dtype=np.float32), [art])
        store1.save()

        store2 = FAISSStore(dimension=4, index_path=str(tmp_path / "idx2"))
        loaded = store2.load()
        assert loaded
        assert store2.index.ntotal == 1


# ── FastAPI endpoints ─────────────────────────────────────────────────── #
@pytest.mark.asyncio
class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        # Patch lifespan so we don't load real models in CI
        import unittest.mock as mock
        import numpy as np

        dummy_engine = mock.MagicMock()
        dummy_engine.encode_single.return_value = np.zeros((1, 384), dtype=np.float32)

        dummy_store = mock.MagicMock()
        dummy_store.search.return_value = [
            (
                {
                    "title": "Test Article",
                    "abstract": "Abstract here",
                    "authors": ["Author A"],
                    "source": "pubmed",
                    "url": "http://test.com",
                    "published_date": "2023-01-01",
                    "doi": "10.1234/test",
                },
                0.95,
            )
        ]
        dummy_store.index = mock.MagicMock()
        dummy_store.index.ntotal = 42

        with mock.patch("api.main.embed_engine", dummy_engine), \
             mock.patch("api.main.vector_store", dummy_store):
            from api.main import app
            with TestClient(app) as c:
                yield c

    def test_health(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_search(self, client):
        r = client.get("/search", params={"q": "ADHD executive function", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert "results" in body
        assert isinstance(body["results"], list)
