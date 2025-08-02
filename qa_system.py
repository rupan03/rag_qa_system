# qa_system.py

import os
from langchain_groq import ChatGroq  # <-- CHANGED
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# This function combines the retrieved document chunks into a single string.
def format_documents(docs):
    """
    Formats a list of document chunks into a single string.
    
    Args:
        docs (list): A list of LangChain Document objects.

    Returns:
        str: The combined page content.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain():
    """
    Creates the RAG chain using LangChain Expression Language (LCEL).
    """
    # Define the prompt template
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context". Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # --- Initialize the LLM (Groq) ---  <-- CHANGED
    # The API key is automatically read from the GROQ_API_KEY environment variable.
    # We are using the Llama 3 8B model, which is very fast and capable.
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3)
    # ------------------------------------
    
    # Create the RAG chain using LCEL
    rag_chain = (
        {"context": lambda x: format_documents(x["input_documents"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain created using Groq and LCEL.")
    return rag_chain

def get_answer_from_query(vector_store, query):
    """
    Takes a user query, retrieves relevant documents, and generates an answer.
    """
    if vector_store is None:
        return "The document vector store is not initialized. Please upload a document first."

    try:
        # Find similar documents in the vector store
        similar_docs = vector_store.similarity_search(query, k=5)
        
        # Get the RAG chain
        rag_chain = create_rag_chain()
        
        # Run the chain with the similar documents and the query
        response = rag_chain.invoke({"input_documents": similar_docs, "question": query})
        
        return response
    except Exception as e:
        print(f"Error during question answering: {e}")
        return "An error occurred while processing your question."