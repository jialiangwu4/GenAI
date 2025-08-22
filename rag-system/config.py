# rag-system/config.py
# Centralized configuration for RAG system

from dotenv import load_dotenv
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Qdrant settings
QDRANT_PATH = os.getenv('QDRANT_PATH', '/app/qdrant_data')
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'rag_pdfs')
VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', '384'))

# Allowed directories for ingestion (converted from comma-separated string)
ALLOWED_DIRECTORIES = set(
    str(Path(d).resolve()) for d in os.getenv('ALLOWED_DIRECTORIES', '.\data,/app/data').split(',')
)

# Ollama settings
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')

# Embedding model settings
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

def validate_config():
    '''Validate configuration settings.'''
    try:
        if not QDRANT_PATH:
            raise ValueError('QDRANT_PATH is not set')
        if not QDRANT_COLLECTION:
            raise ValueError('QDRANT_COLLECTION is not set')
        if VECTOR_DIMENSION <= 0:
            raise ValueError('VECTOR_DIMENSION must be positive')
        if not ALLOWED_DIRECTORIES:
            raise ValueError('ALLOWED_DIRECTORIES is empty')
        if not OLLAMA_URL:
            raise ValueError('OLLAMA_URL is not set')
        if not OLLAMA_MODEL:
            raise ValueError('OLLAMA_MODEL is not set')
        if not EMBEDDING_MODEL:
            raise ValueError('EMBEDDING_MODEL is not set')
        logger.info('Configuration validated successfully')
    except Exception as e:
        logger.error(f'Configuration validation failed: {str(e)}')
        raise

# Run validation on import
validate_config()