"""
Configuration module for the RAG Document Q&A Bot.

Loads environment variables and defines all configurable constants
used throughout the pipeline.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
# Search for .env in the project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: str = str(_PROJECT_ROOT / "data")
CHROMA_DB_DIR: str = str(_PROJECT_ROOT / "chroma_db")

# ---------------------------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIMENSIONS: int = 768
EMBEDDING_BATCH_SIZE: int = 100

# ---------------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
GEMINI_MODEL: str = "gemini-2.0-flash"

# ---------------------------------------------------------------------------
# ChromaDB Collection Name
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "rag_documents"


def validate_config() -> None:
    """Validate that required configuration values are set.

    Raises:
        SystemExit: If required environment variables are missing.
    """
    errors = []

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
        errors.append(
            "GOOGLE_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/apikey "
            "and add it to your .env file."
        )

    if not os.path.isdir(DATA_DIR):
        errors.append(
            f"Data directory not found: {DATA_DIR}. "
            "Create a /data folder and add your documents."
        )

    if errors:
        print("\n❌ Configuration Error(s):\n")
        for err in errors:
            print(f"  • {err}")
        print()
        sys.exit(1)
