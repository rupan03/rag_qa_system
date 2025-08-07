# app.py

import os
import streamlit as st
from dotenv import load_dotenv

# Import your backend functions
from document_processor import load_and_chunk_document, get_embeddings_model
from vector_store import create_or_update_vector_store, load_vector_store
from qa_system import get_answer_from_query

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Doc Q&A System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Doc Q&A System (RAG)")
st.write("Upload a PDF document and ask questions about its content.")

@st.cache_resource
def load_embedding_model():
    """Loads the embedding model and caches it."""
    return get_embeddings_model()

# --- Main Application Logic ---

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("1. Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner('Processing document... This may take a moment.'):
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                embeddings_model = load_embedding_model()
                chunked_docs = load_and_chunk_document(file_path)

                if chunked_docs:
                    create_or_update_vector_store(chunked_docs, embeddings_model)
                    st.session_state.processed_file = uploaded_file.name
                    st.success(f"File '{uploaded_file.name}' processed successfully!")
                else:
                    st.error("Failed to process the document.")
        else:
            st.info(f"File '{uploaded_file.name}' is already loaded and processed.")

# --- Chat Input and Q&A Logic ---
if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            embeddings_model = load_embedding_model()
            vector_store = load_vector_store(embeddings_model)

            if vector_store is None:
                st.warning("Knowledge base is not ready. Please upload a document first or check your Pinecone connection.")
            else:
                answer = get_answer_from_query(vector_store, prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})