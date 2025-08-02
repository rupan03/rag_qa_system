# vector_store.py

import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# The name of the index you created in Pinecone
PINECONE_INDEX_NAME = "rag-qa-index" 

def create_or_update_vector_store(chunked_docs, embeddings_model):
    """
    Creates a new Pinecone vector store or updates an existing one
    with new document chunks.
    """
    try:
        # Initialize PineconeVectorStore with the index name and embedding model
        # This will create the index if it doesn't exist, or use the existing one.
        print("Adding documents to Pinecone index...")
        PineconeVectorStore.from_documents(
            documents=chunked_docs, 
            embedding=embeddings_model, 
            index_name=PINECONE_INDEX_NAME
        )
        print("Vector store updated in Pinecone.")
    except Exception as e:
        print(f"Error updating Pinecone vector store: {e}")

def load_vector_store(embeddings_model):
    """
    Loads an existing Pinecone vector store.
    """
    try:
        print("Loading existing Pinecone vector store...")
        # Just initialize the connection to the existing index
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings_model
        )
        print("Pinecone vector store loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error loading Pinecone vector store: {e}")
        # Return None if the store can't be loaded
        return None