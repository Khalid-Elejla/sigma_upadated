from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import tool
from .embedding_utils import get_embedding_function
import numpy as np
from .db_utils import get_db_connection
from psycopg2.extras import RealDictCursor
from .react_template import get_react_prompt_template

PGVECTOR_TABLE = 'documents'


@tool
def embed_query_and_do_similarity_check(query_text: str):
    """"
    Perform similarity searches on text using vector embeddings in a PostgreSQL database with pgvector extension.
    It takes a query text as input, generates its vector embedding,
    and retrieves the most similar documents based on cosine similarity.
    This tool is optimized for use in Retrieval-Augmented Generation (RAG) applications,
    where the goal is to enhance the contextual relevance of generated responses
    by retrieving and providing the closest matching pieces of text from a knowledge base.
    """

    # Prepare the embedding function
    embedding_function = get_embedding_function()

    # Generate embedding for the query
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
    context_text = "\n\n---\n\n".join([result['metadata'].get('page_content', '') for result in results])
    return context_text


def query_rag(query_text: str) -> str:
    """"
    Generate a formatted response for the given query text using a REACT agent.
    """
    query = "based on my postgresql database," + query_text
    # Initialize the model
    model = Ollama(model="mistral-nemo")
    llm = model

    # Define the prompt template
    prompt_template = get_react_prompt_template()

    # Define the tools and create the agent
    tools = [embed_query_and_do_similarity_check]
    agent = create_react_agent(llm, tools, prompt_template)

    # Create the agent executor with error handling for parsing errors
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors="Check you output and make sure it conforms! Do not output an action and a final answer at the same time.",)

    # Invoke the agent with the query
    try:
        response = agent_executor.invoke({"input": query})
        print(type(response))
        response = str(response)
        return response
    except ValueError as e:
        return f"An error occurred: {e}"