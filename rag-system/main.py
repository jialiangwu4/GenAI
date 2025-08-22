# main.py
# Entry point to run ingestion, retrieval, and generation

from ingestion.ingest import ingest_pdfs
from retrieval.retrieve import query_chunks
from generation.generate import generate_response
from utils.qdrant_utils import reset_qdrant_client
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(directory: str, query: str, chunk_size: int = 500, chunk_overlap: int = 100, top_k: int = 5):
    '''Run the full pipeline: ingest PDFs, query, and generate response.'''
    try:
        # Ingest PDFs
        logger.info(f'Starting ingestion for directory: {directory}')
        ingest_result = ingest_pdfs(directory, chunk_size, chunk_overlap)
        logger.info(f'Ingestion complete: {ingest_result}')

        # Query Qdrant
        logger.info(f'Querying with text: {query}')
        query_result = query_chunks(query, top_k)
        contexts = [result['text'] for result in query_result['results']]
        logger.info(f'Retrieved {len(contexts)} chunks')

        # Generate response
        logger.info(f'Generating response for query: {query}')
        generate_result = generate_response(query, contexts)
        logger.info(f'Generation complete: {generate_result}')

        return {
            'ingestion': ingest_result,
            'retrieval': query_result,
            'generation': generate_result
        }
    except Exception as e:
        logger.error(f'Pipeline failed: {str(e)}')
        raise
    finally:
        reset_qdrant_client()  # Ensure client is closed on exit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RAG pipeline')
    parser.add_argument('--directory', type=str, required=True, help='Directory containing PDFs')
    parser.add_argument('--query', type=str, required=True, help='Query text')
    parser.add_argument('--chunk-size', type=int, default=500, help='Size of text chunks')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Overlap between chunks')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to retrieve')
    args = parser.parse_args()
    main(args.directory, args.query, args.chunk_size, args.chunk_overlap, args.top_k)