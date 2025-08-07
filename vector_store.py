# vector_store.py

from langchain_pinecone import PineconeVectorStore

# The name of the index you created in your Pinecone account
PINECONE_INDEX_NAME = "rag-qa-index" 

def create_or_update_vector_store(chunked_docs, embeddings_model):
    """
    Adds new document chunks to the Pinecone vector store.
    """
    try:
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
    Loads an existing Pinecone vector store to be used for queries.
    """
    try:
        print("Loading existing Pinecone vector store...")
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings_model
        )
        print("Pinecone vector store loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error loading Pinecone vector store: {e}")
        return None