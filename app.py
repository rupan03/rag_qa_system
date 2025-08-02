# app.py

import os
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv

from document_processor import load_and_chunk_document, get_embeddings_model
from vector_store import create_or_update_vector_store, load_vector_store
from qa_system import get_answer_from_query

# Load environment variables from .env file
load_dotenv()

# --- Initialization ---
app = Flask(__name__)

# Define paths
UPLOADS_FOLDER = "uploads"
VECTOR_DB_FOLDER = "vector_db"
os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# Global variables to hold the vector store and embeddings model
vector_store = None
embeddings_model = None

def initialize_app():
    """Initializes the embeddings model and loads the vector store if it exists."""
    global embeddings_model, vector_store
    print("Initializing application...")
    embeddings_model = get_embeddings_model()
    vector_store = load_vector_store(embeddings_model)
    print("Application initialized.")

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main web page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and processing."""
    global vector_store, embeddings_model
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(UPLOADS_FOLDER, file.filename)
        file.save(file_path)

        # Process the document
        chunked_docs = load_and_chunk_document(file_path)
        if chunked_docs:
            create_or_update_vector_store(chunked_docs, embeddings_model)
            # Reload the vector store to ensure it's up-to-date in memory
            vector_store = load_vector_store(embeddings_model)
            return jsonify({"success": f"File '{file.filename}' processed and added to the knowledge base."})
        else:
            return jsonify({"error": "Failed to process the document."}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handles asking a question and returning an answer."""
    global vector_store
    
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Question is required."}), 400

    if vector_store is None:
        return jsonify({"answer": "Knowledge base is not ready. Please upload a document first."})

    answer = get_answer_from_query(vector_store, question)
    return jsonify({"answer": answer})

# --- Main Execution ---

if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=5000, debug=True)