from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Pydantic BaseSettings loads environment variables from .env file automatically.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI configuration
    OPENAI_API: str = Field(..., env="OPENAI_API", description="OpenAI API Key")
    BASE_URL: str = Field("https://api.openai.com/v1", env="BASE_URL", description="OpenAI API Base URL")
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_chat_model: str = Field("gpt-4o", env="OPENAI_CHAT_MODEL")
    embedding_dimensions: int = Field(1536, env="EMBEDDING_DIMENSIONS")

    # Qdrant configuration
    qdrant_url: str = Field("qdrant_storage", env="QDRANT_URL")
    qdrant_api_key: str | None = Field(None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field("knowledge_base", env="QDRANT_COLLECTION_NAME")

    # Chunking configuration
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")

    # Retrieval configuration
    top_k: int = Field(5, env="TOP_K")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")

    @computed_field
    @property
    def qdrant_collection(self) -> str:
        return self.qdrant_collection_name


settings = Settings()
