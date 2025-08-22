# utils/embeddings.py
# Shared utility for generating text embeddings

from sentence_transformers import SentenceTransformer
import logging
from config import EMBEDDING_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global model
_model = None


def get_model(model=None):
    '''
    Get or initialize the SentenceTransformer model.

    Parameters:
        model (SentenceTransformer, optional): If provided, this model instance will be used instead of loading a new one. Useful for dependency injection or testing.

    Returns:
        SentenceTransformer: The loaded or provided model instance.
    '''
    global _model
    if model is not None:
        return model  # Use injected model for testing
    if _model is None:
        try:
            logger.info(f'Loading SentenceTransformer model: {EMBEDDING_MODEL}')
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            logger.error(f'Failed to load model: {str(e)}')
            raise
    return _model


def reset_model():
    '''Reset the model to handle errors or updates.'''
    global _model
    _model = None
    logger.info('Model reset requested')


def embed_text(text, model=None):
    '''
    Generate embeddings for text using SentenceTransformer.

    Parameters:
        text (str or list of str): The input text or list of texts to embed.

    Returns:
        list: The generated embedding(s) as a list.
    '''
    try:
        model_instance = get_model(model)
        embeddings = model_instance.encode(text).tolist()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Generated embeddings for text: {text[:50]}...')
        return embeddings
    except Exception as e:
        logger.error(f'Embedding failed: {str(e)}')
        reset_model()
        raise