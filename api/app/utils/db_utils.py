import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import json

# PostgreSQL connection details
DB_HOST = 'localhost'
DB_NAME = 'rag_test'
DB_USER = 'postgres'
DB_PASSWORD = 'abood@2009'


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def insert_document(chunk_id, metadata, embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (chunk_id, metadata, embedding) VALUES (%s, %s, %s) ON CONFLICT (chunk_id) DO UPDATE SET metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
        (chunk_id, json.dumps(metadata), embedding)
    )
    conn.commit()
    cursor.close()
    conn.close()
