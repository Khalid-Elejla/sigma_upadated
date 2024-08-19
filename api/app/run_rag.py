import argparse
from .utils.pdf_utils import load_documents, split_documents
from .utils.embedding_utils import add_to_pgvector
# from .utils.query_utils import query_rag
from .utils.react_query_utils import query_rag

def main(data_path, query_string):
    # Load documents from the specified path
    documents = load_documents(data_path)

    # Split the loaded documents into chunks
    chunks = split_documents(documents)

    # Add the chunks to pgvector
    add_to_pgvector(chunks)

    # Query the system
    response = query_rag(query_string)
    return response

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process documents and query the system.")
    
    # Add arguments with default values and shortcuts
    parser.add_argument('-d', '--data_path', type=str, default='data', help='Path to the directory or file containing documents.')
    parser.add_argument('-q', '--query', type=str, help='Query string to be processed.')

    # Parse arguments
    args = parser.parse_args()

    # Call main function with arguments
    main(args.data_path, args.query)
