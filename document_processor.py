# document_processor.py

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_and_chunk_document(file_path):
    """
    Loads a document from the given file path, splits it into chunks,
    and returns the chunks.

    Args:
        file_path (str): The path to the document file.

    Returns:
        list: A list of document chunks.
    """
    try:
        # Load the document using PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_docs = text_splitter.split_documents(documents)
        
        print(f"Successfully loaded and chunked document: {os.path.basename(file_path)}")
        return chunked_docs
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")
        return None

def get_embeddings_model(model_name="all-MiniLM-L6-v2"):
    """
    Initializes and returns a sentence-transformer model for embeddings.

    Args:
        model_name (str): The name of the Hugging Face model to use.

    Returns:
        HuggingFaceEmbeddings: The embedding model instance.
    """
    # Use a pre-trained model from Hugging Face for creating embeddings
    # This model runs locally on your machine (or server)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"Embedding model '{model_name}' loaded.")
    return embeddings