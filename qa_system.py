# qa_system.py

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def format_documents(docs):
    """
    Formats a list of document chunks into a single string for the prompt context.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain():
    """
    Creates the RAG chain using LangChain Expression Language (LCEL).
    """
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
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3)
    
    rag_chain = (
        {"context": lambda x: format_documents(x["input_documents"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain created using Groq.")
    return rag_chain

def get_answer_from_query(vector_store, query):
    """
    Takes a user query, retrieves relevant documents, and generates an answer.
    """
    if vector_store is None:
        return "The document vector store is not initialized."

    try:
        similar_docs = vector_store.similarity_search(query, k=5)
        rag_chain = create_rag_chain()
        response = rag_chain.invoke({"input_documents": similar_docs, "question": query})
        return response
    except Exception as e:
        print(f"Error during question answering: {e}")
        return "An error occurred while processing your question."