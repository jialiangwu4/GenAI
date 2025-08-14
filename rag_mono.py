from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import requests

model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(path='qdrant_data')
collection = 'rag_pdfs'
dimension_size = 384  # The size of VectorParams should match the dimension of the model used for embedding


if not client.collection_exists(collection):
   client.create_collection(
      collection_name=collection,
      vectors_config=VectorParams(size=dimension_size, distance=Distance.COSINE),
   )


def load_and_chunk_pdf(directory: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """
    Load a PDF file and split it into chunks.

    Args:
        file_path (str): The path to the PDF file.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        list: A list of text chunks.
    """
    # Ensure the directory exists
    pdf_path = Path(directory)
    if not pdf_path.is_dir():
        raise ValueError(f'The directory {directory} does not exist or is not a directory.')
    
    # Read the PDF file
    doc_loader = PyPDFDirectoryLoader(pdf_path)
    docs = doc_loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return text_splitter.split_documents(docs)



def embed_text(text):
    """
    Generate embeddings for the given text using a pre-trained model.

    Args:
        text (str): The input text to be embedded.

    Returns:
        list: A list of embeddings.
    """
    embeddings = model.encode(text)
    return embeddings


def create_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get('source')
        page = chunk.metadata.get('page')
        current_page_id = f'{source}:{page}'

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f'{current_page_id}:{current_chunk_index}'
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata['id'] = chunk_id

    return chunks


def add_to_db(chunks):
    """
    Add text chunks to the Qdrant database.

    Args:
        chunks (list): A list of text chunks to be added.
    """
    points = []
    for i, chunk in enumerate(chunks):
        points.append(PointStruct(
        id=i,
        vector=embed_text(chunk.page_content).tolist(),
        payload={
            'text': chunk.page_content,
            'source': chunk.metadata.get('source'),
            'chunk_id': chunk.metadata.get('id'),
        }
    ))
    
    client.upsert(
        collection_name=collection,
        points=points
    )
    print(f'Added {len(chunks)} chunks to the database.')


def query_db(query_text, top_k=5):
    """
    Query the Qdrant database for similar text chunks.

    Args:
        query_text (str): The text to query.
        top_k (int): The number of top results to return.

    Returns:
        list: A list of similar text chunks.
    """
    query_vector = embed_text(query_text).tolist()
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k
    ).points
    
    return results


def build_prompt(contexts, query):
    context = "\n\n".join(contexts)
    return f"""Answer the following question using the context below.

    Context:
    {context}

    Question: {query}
    Answer:"""



def call_ollama(prompt):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
    return response.json()['response']



if __name__ == '__main__':
    # Example usage
    file_path = 'data'
    # chunks = load_and_chunk_pdf(file_path)
    # chunks_with_ids = create_chunk_ids(chunks)
    # add_to_db(chunks_with_ids)
    # for i, chunk in enumerate(chunks_with_ids):
    #     print(f'Chunk {i+1}:\n{chunk}\n{chunk.metadata.get("source")}\n{"-"*40}')
    
    # print(embed_text('how to use the tool?'))
    
    results = query_db('how to write a document?', top_k=3)
    client.close()
    # print(results)
    prompt = build_prompt([r.payload['text'] for r in results], 'how to use the tool?')
    
    print(prompt)
    
    response = call_ollama(prompt)
    print(response)
        

        
        
