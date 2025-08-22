# generation/generate.py
# Module for generating responses using Ollama

import httpx
import logging
from config import OLLAMA_URL, OLLAMA_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_response(query: str, contexts: list):
    """
    Generate a response for a given query using provided contexts and the Ollama model.

    Args:
        query (str): The input query string for which a response is to be generated.
        contexts (list): A list of context strings to provide additional information for the query.

    Returns:
        dict: A dictionary containing the original query and the generated response from Ollama, 
              with keys 'query' and 'response'.
    """
    try:
        prompt = f'Query: {query}\n\nContexts:\n' + '\n'.join(contexts)
        payload = {
            'model': OLLAMA_MODEL,
            'prompt': prompt,
            'stream': False
        }
        logger.info(f'Sending request to Ollama for query: {query}')
        with httpx.Client() as client:
            response = client.post(OLLAMA_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
        logger.info(f'Received response from Ollama for query: {query}')
        return {'query': query, 'response': result.get('response', '')}
    except Exception as e:
        logger.error(f'Generation failed for query {query}: {str(e)}')
        raise