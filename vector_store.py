# vector_store.py

import os
from langchain_community.vectorstores import FAISS

# Define the path for storing the vector database
VECTOR_DB_PATH = "vector_db/faiss_index"

def create_or_update_vector_store(chunked_docs, embeddings_model):
    """
    Creates a new FAISS vector store or updates an existing one
    with new document chunks.

    Args:
        chunked_docs (list): A list of document chunks.
        embeddings_model: The embedding model instance.
    """
    if os.path.exists(VECTOR_DB_PATH):
        # Load the existing vector store
        vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings_model, allow_dangerous_deserialization=True)
        # Add new documents to the existing store
        vector_store.add_documents(chunked_docs)
        print("Existing vector store updated.")
    else:
        # Create a new vector store if one doesn't exist
        vector_store = FAISS.from_documents(chunked_docs, embedding=embeddings_model)
        print("New vector store created.")
    
    # Save the updated vector store to disk
    vector_store.save_local(VECTOR_DB_PATH)
    print(f"Vector store saved at: {VECTOR_DB_PATH}")

def load_vector_store(embeddings_model):
    """
    Loads the FAISS vector store from the local path.

    Args:
        embeddings_model: The embedding model instance.

    Returns:
        FAISS: The loaded vector store instance, or None if it doesn't exist.
    """
    if os.path.exists(VECTOR_DB_PATH):
        try:
            vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings_model, allow_dangerous_deserialization=True)
            print("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    return None