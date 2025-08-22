# retrieval/retrieve.py
# Module for querying Qdrant to retrieve relevant text chunks

from utils.embeddings import embed_text
from utils.qdrant_utils import get_qdrant_client
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File logging for query activity
file_handler = logging.FileHandler('query_log.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


def query_chunks(query_text: str, top_k: int = 5, client=None, model=None):
    """
    Retrieve the top_k most relevant text chunks from a Qdrant collection based on a query string.

    Args:
        query_text (str): The input query string to search for relevant chunks.
        top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 5.
        client (optional): An existing Qdrant client instance. Pass a custom client for testing or specific configurations.
        model (optional): The embedding model to use for encoding the query text. Pass a custom model for testing or specific configurations.

    Returns:
        dict: A dictionary with a single key 'results', containing a list of dictionaries for each retrieved chunk.
            Each dictionary contains:
                - 'text' (str): The chunk's text content.
                - 'source' (str): The source identifier of the chunk.
                - 'chunk_id' (Any): The unique identifier of the chunk.
                - 'score' (float): The relevance score of the chunk.

    Raises:
        Exception: If the retrieval process fails for any reason.
    """

    try:
        client, collection = get_qdrant_client(client)
        logger.info(f'Querying with text: {query_text}, top_k: {top_k}')
        query_vector = embed_text(query_text, model)
        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k
        ).points
        response = [
            {
                'text': point.payload['text'],
                'source': point.payload['source'],
                'chunk_id': point.payload['chunk_id'],
                'score': point.score
            } for point in results
        ]
        logger.info(f'Retrieved {len(response)} chunks for query: {query_text}')
        return {'results': response}
    except Exception as e:
        logger.error(f'Query failed for {query_text}: {str(e)}')
        raise