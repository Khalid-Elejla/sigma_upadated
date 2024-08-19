from .db_utils import get_db_connection, insert_document
from .pdf_utils import calculate_chunk_ids
from langchain.schema.document import Document
# from langchain.embeddings import OpenAIEmbeddings ## decrypted
from langchain_community.embeddings import OpenAIEmbeddings
from psycopg2.extras import RealDictCursor
import numpy as np

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_embedding_function():
    # Retrieve the API key from the environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Pass the API key to the embeddings instance
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return embeddings

def add_to_pgvector(chunks: list[Document]):
    # Calculate Page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Load existing IDs from the database
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT chunk_id FROM documents")
    existing_ids = {row['chunk_id'] for row in cursor.fetchall()}
    cursor.close()
    conn.close()

    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")

        # Initialize the embedding function
        embedding_function = get_embedding_function()

        # Extract the page content for embeddings
        chunk_texts = [chunk.page_content for chunk in new_chunks]

        # Generate embeddings for all chunks
        embeddings = embedding_function.embed_documents(chunk_texts)  # Adjust method name if needed

        for chunk, embedding in zip(new_chunks, embeddings):
            chunk_id = chunk.metadata["id"]
            
            #metadata = chunk.metadata
            
            
            metadata = {
                "id": chunk_id,
                "page": chunk.metadata.get("page"),
                "source": chunk.metadata.get("source"),
                "page_content": chunk.page_content  # Include page content here
            }
            
            
            embedding_vector = np.array(embedding).astype(np.float32).tolist()  # Convert to list for PostgreSQL
            insert_document(chunk_id, metadata, embedding_vector)

    else:
        print("✅ No new documents to add")