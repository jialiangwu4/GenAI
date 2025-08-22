# ingestion/ingest.py
# Module for ingesting PDFs, chunking, embedding, and storing in Qdrant

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from qdrant_client.models import PointStruct
from utils.embeddings import embed_text
from utils.qdrant_utils import get_qdrant_client
from config import ALLOWED_DIRECTORIES
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_chunk_pdf(directory: str, chunk_size: int = 500, chunk_overlap: int = 100):
    '''Load PDFs from directory and split into chunks.'''
    pdf_path = Path(directory)
    if not pdf_path.is_dir():
        logger.error(f'Directory {directory} does not exist')
        raise ValueError(f'Directory {directory} does not exist')
    if str(pdf_path.resolve()) not in ALLOWED_DIRECTORIES:
        logger.error(f'Directory {directory} not in allowed list')
        raise ValueError(f'Directory {directory} not allowed')
    
    logger.info(f'Loading PDFs from {directory}')
    doc_loader = PyPDFDirectoryLoader(pdf_path)
    docs = doc_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def create_chunk_ids(chunks):
    '''Assign unique IDs to chunks based on source, page, and index.'''
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get('source')
        page = chunk.metadata.get('page')
        current_page_id = f'{source}:{page}'
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f'{current_page_id}:{current_chunk_index}'
        chunk.metadata['id'] = chunk_id
        last_page_id = current_page_id
    return chunks


def ingest_pdfs(directory: str, chunk_size: int = 500, chunk_overlap: int = 100, client=None):
    """
    Processes all PDF files in a specified directory by loading, chunking, embedding, and storing their content in a Qdrant vector database.

    Args:
        directory (str): Path to the directory containing PDF files to ingest.
        chunk_size (int, optional): Number of characters per text chunk. Defaults to 500.
        chunk_overlap (int, optional): Number of overlapping characters between chunks. Defaults to 100.
        client (optional): Existing Qdrant client instance. Pass a custom client for testing or specific configurations.

    Returns:
        dict: A dictionary containing the status of the operation, the number of chunks added, and the source directory.

    Raises:
        Exception: If any error occurs during the ingestion process.
    """

    try:
        client, collection = get_qdrant_client(client)
        chunks = load_and_chunk_pdf(directory, chunk_size, chunk_overlap)
        chunks_with_ids = create_chunk_ids(chunks)
        points = [
            PointStruct(
                id=i,
                vector=embed_text(chunk.page_content),
                payload={
                    'text': chunk.page_content,
                    'source': chunk.metadata.get('source'),
                    'chunk_id': chunk.metadata.get('id'),
                }
            ) for i, chunk in enumerate(chunks_with_ids)
        ]
        client.upsert(collection_name=collection, points=points)
        logger.info(f'Added {len(chunks)} chunks to Qdrant from {directory}')
        return {'status': 'success', 'chunks_added': len(chunks), 'directory': directory}
    except Exception as e:
        logger.error(f'Ingestion failed for {directory}: {str(e)}')
        raise