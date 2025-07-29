import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq

def load_docs_from_path(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(documents)
    return final_documents

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"trust_remote_code": True}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def chat_groq(messages):
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    
    client = Groq(api_key=api_key)
    response_content = ''
    stream = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += content
    return response_content
