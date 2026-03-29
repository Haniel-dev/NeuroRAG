from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    # Pinecone
    pinecone_api_key: str = Field(default="", env="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-east-1", env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="neuro-rag", env="PINECONE_INDEX_NAME")

    # Vector DB mode
    vector_db_mode: Literal["faiss", "pinecone"] = Field(default="faiss", env="VECTOR_DB_MODE")
    faiss_index_path: str = Field(default="./data/faiss_index", env="FAISS_INDEX_PATH")

    # Embedding
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")

    # Scraping
    pubmed_email: str = Field(default="user@example.com", env="PUBMED_EMAIL")
    pubmed_api_key: str = Field(default="", env="PUBMED_API_KEY")
    max_results_per_source: int = Field(default=200, env="MAX_RESULTS_PER_SOURCE")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
