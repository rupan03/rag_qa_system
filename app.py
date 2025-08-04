# app.py

import os
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv

from document_processor import load_and_chunk_document, get_embeddings_model
from vector_store import create_or_update_vector_store, load_vector_store

# Load environment variables from .env file
load_dotenv()

# --- Initialization ---
app = Flask(__name__)

# Define paths for local uploads (will not be used on Render but good for local dev)
UPLOADS_FOLDER = "uploads"
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

# Global variable to hold the embeddings model only
embeddings_model = None

def initialize_app():
    """Initializes the embeddings model when the application starts."""
    global embeddings_model
    print("Initializing embeddings model...")
    embeddings_model = get_embeddings_model()
    print("Embeddings model initialized.")

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main web page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and processing."""
    global embeddings_model
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save file locally before processing
        file_path = os.path.join(UPLOADS_FOLDER, file.filename)
        file.save(file_path)

        # Process the document and update the vector store in Pinecone
        chunked_docs = load_and_chunk_document(file_path)
        if chunked_docs:
            create_or_update_vector_store(chunked_docs, embeddings_model)
            return jsonify({"success": f"File '{file.filename}' processed and added to the knowledge base."})
        else:
            return jsonify({"error": "Failed to process the document."}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handles asking a question and returning an answer."""
    global embeddings_model
    
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Question is required."}), 400

    # --- THIS IS THE KEY CHANGE ---
    # Load the vector store from Pinecone on every request.
    # This ensures we always have a valid connection, regardless of the worker process.
    vector_store = load_vector_store(embeddings_model)

    if vector_store is None:
        return jsonify({"answer": "Knowledge base is not ready. Please upload a document first."})

    answer = get_answer_from_query(vector_store, question)
    return jsonify({"answer": answer})

# --- Main Execution ---
if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    # This runs when Gunicorn starts the app on Render
    initialize_app()