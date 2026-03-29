"""Vector DB package — factory selects FAISS or Pinecone."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .faiss_store import FAISSStore
from .pinecone_store import PineconeStore

if TYPE_CHECKING:
    from config import Settings

__all__ = ["FAISSStore", "PineconeStore", "get_vector_store"]


def get_vector_store(settings: "Settings"):
    """Return the appropriate store based on VECTOR_DB_MODE."""
    if settings.vector_db_mode == "pinecone":
        return PineconeStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            environment=settings.pinecone_environment,
        )
    # Default: FAISS
    store = FAISSStore(
        dimension=settings.embedding_dimension,
        index_path=settings.faiss_index_path,
    )
    store.load()  # no-op if index doesn't exist yet
    return store
