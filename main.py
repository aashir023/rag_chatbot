import os
import io
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

load_dotenv()

# On Vercel, it's better to raise an error if the key is missing at startup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

app = FastAPI(title="RAG-CHATBOT")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Global variables to hold the in-memory vector store and embeddings
# NOTE: These will be reset on every serverless function cold start.
VECTORSTORE = None
EMBEDDINGS = None

# Request model for the chat endpoint
class ChatRequest(BaseModel):
    query: str
    k: Optional[int] = 4

# --- Helper Functions ---

def process_pdf_and_create_store(pdf_file: io.BytesIO):
    """Loads a PDF from memory, splits it, and creates a vector store."""
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(documents)
    return final_documents

def create_vector_store(docs):
    """Creates and returns a FAISS vector store from documents."""
    global EMBEDDINGS
    EMBEDDINGS = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"trust_remote_code": True}
    )
    vectorstore = FAISS.from_documents(docs, EMBEDDINGS)
    return vectorstore

def chat_groq(messages):
    """Generates a response from the Groq API."""
    client = Groq(api_key=GROQ_API_KEY)
    response_content = ""
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=1024,
        temperature=1.0,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += content
    return response_content

# --- API Endpoints ---

@app.get("/")
async def read_root(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/index_pdf")
async def index_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, processes it, and creates an in-memory vector store.
    """
    global VECTORSTORE
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        # Read the uploaded file into memory
        pdf_bytes = await file.read()
        pdf_file_stream = io.BytesIO(pdf_bytes)
        
        # Process the PDF and create the vector store
        docs = process_pdf_and_create_store(pdf_file_stream)
        VECTORSTORE = create_vector_store(docs)
        
        return {"status": "ok", "message": "PDF uploaded and indexed successfully.", "chunks": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Answers a user query using the currently indexed documents.
    """
    global VECTORSTORE

    if VECTORSTORE is None:
        raise HTTPException(status_code=400, detail="No documents have been indexed. Please upload a PDF first.")

    # Retrieve relevant context
    hits = VECTORSTORE.similarity_search(req.query, k=req.k)
    context_snippets = [h.page_content for h in hits]

    # Compose the prompt for the LLM
    system = {"role": "system", "content": "You are a helpful assistant that uses provided context to answer."}
    user_content = "Context:\n" + ("\n\n---\n\n".join(context_snippets) if context_snippets else "No context available.") + "\n\nUser question: " + req.query
    user = {"role": "user", "content": user_content}
    
    try:
        result = chat_groq([system, user])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

    return {"answer": result, "context_count": len(context_snippets)}