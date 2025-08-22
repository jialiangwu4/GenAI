# utils/qdrant_utils.py
# Shared utility for Qdrant client setup with singleton pattern

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from config import QDRANT_PATH, QDRANT_COLLECTION, VECTOR_DIMENSION
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global Qdrant client
_client = None


def get_qdrant_client(client=None):
    '''Get or initialize the Qdrant client singleton and ensure collection exists.'''
    global _client
    if client is not None:
        return client, QDRANT_COLLECTION  # Use injected client for testing
    if _client is None:
        try:
            logger.info(f'Initializing Qdrant client with path: {QDRANT_PATH}')
            _client = QdrantClient(path=QDRANT_PATH)
            if not _client.collection_exists(QDRANT_COLLECTION):
                logger.info(f'Creating Qdrant collection: {QDRANT_COLLECTION}')
                _client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE),
                )
        except Exception as e:
            logger.error(f'Failed to initialize Qdrant client: {str(e)}')
            raise
    return _client, QDRANT_COLLECTION


def reset_qdrant_client():
    '''Reset the Qdrant client singleton to handle errors or updates.'''
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception as e:
            logger.error(f'Failed to close Qdrant client: {str(e)}')
        _client = None
        logger.info('Qdrant client reset')