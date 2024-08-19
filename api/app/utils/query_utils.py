import numpy as np
from psycopg2.extras import RealDictCursor
# from langchain.embeddings import OpenAIEmbeddings ## decrypted
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.prompts import ChatPromptTemplate
# from langchain.llms import OpenAI # uncomment if you want to use OpenAI models
from langchain_community.llms.ollama import Ollama
from .db_utils import get_db_connection
from .embedding_utils import get_embedding_function


# Define constants
PGVECTOR_TABLE = 'documents'  # Replace with your pgvector table name
# BASIC PROMPT TEMPLATE = "Based on this {context} answer this {question}" # if you want to make simple query
PROMPT_TEMPLATE = """
You are a knowledgeable assistant with access to the following information. Use this context to provide a thorough and accurate answer to the question asked.

Context:
{context}

Question:
{question}

Instructions:
1. Carefully review the context provided above.
2. Use the information in the context to formulate a well-reasoned answer to the question.
3. If the context does not provide enough information to answer the question, state clearly that the information is not available.
4. Provide explanations or justifications for your answer if necessary.

Answer:
"""

def query_rag(query_text: str):
    # Prepare the embedding function
    embedding_function = get_embedding_function()

    # Generate embedding for the query
    # Assuming embed_documents can be used for single document as well
    query_embedding = embedding_function.embed_documents([query_text])[0]
    query_embedding_vector = np.array(query_embedding).astype(np.float32).tolist()

    # Connect to the PostgreSQL database
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Perform similarity search with pgvector
    cursor.execute(f"""
        SELECT chunk_id, metadata, embedding, 
               (embedding <=> %s::vector) AS distance 
        FROM {PGVECTOR_TABLE} 
        ORDER BY distance 
        LIMIT 5
    """, (query_embedding_vector,))


    results = cursor.fetchall()
    cursor.close()
    conn.close()

    # Format the context from the results
    # Adjust the field names based on your actual table schema
    context_text = "\n\n---\n\n".join([result['metadata'].get('page_content', '') for result in results])

    # Create and format the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Invoke the model to get a response
    #model = OpenAI(max_tokens=1024) # uncomment if you want to use OpenAI models
    model = Ollama(model = "mistral")
    response_text = model.invoke(prompt)

    # Collect sources from results
    sources = [result['chunk_id'] for result in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    print(formatted_response) # you also add both prompt and response for debugging purposes
    return response_text