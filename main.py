import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    # keep app usable during development but raise on calls that require Groq
    pass

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

# global simple store
VECTORSTORE = None
EMBEDDINGS = None
FAISS_DIR = "faiss_index"  # optional folder for saving/loading

# small request models
class IndexRequest(BaseModel):
    pdf_path: str

class ChatRequest(BaseModel):
    query: str
    k: Optional[int] = 4

# helper functions (kept small and close to your originals)
def load_docs_from_path(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(documents)
    return final_documents

def create_vector_store(docs):
    global EMBEDDINGS
    EMBEDDINGS = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"trust_remote_code": True}
    )
    vectorstore = FAISS.from_documents(docs, EMBEDDINGS)
    return vectorstore

def chat_groq(messages):
    # requires GROQ_API_KEY in env
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    client = Groq(api_key=api_key)
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

# endpoints
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/index_pdf")
def index_pdf(req: IndexRequest):
    global VECTORSTORE
    if not os.path.exists(req.pdf_path):
        raise HTTPException(status_code=400, detail="pdf_path not found")
    docs = load_docs_from_path(req.pdf_path)
    VECTORSTORE = create_vector_store(docs)
    # optional: save local
    try:
        VECTORSTORE.save_local(FAISS_DIR)
    except Exception:
        pass
    return {"status": "ok", "chunks": len(docs)}

@app.post("/chat")
def chat(req: ChatRequest):
    global VECTORSTORE, EMBEDDINGS
    
    # build retrieval context if index available
    context_snippets = []
    if VECTORSTORE is not None:
        hits = VECTORSTORE.similarity_search(req.query, k=req.k)
        context_snippets = [h.page_content for h in hits]
    else:
        # try to load from disk if available
        if os.path.isdir(FAISS_DIR):
            try:
                EMBEDDINGS = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"trust_remote_code": True}
                )
                VECTORSTORE = FAISS.load_local(FAISS_DIR, EMBEDDINGS)
                hits = VECTORSTORE.similarity_search(req.query, k=req.k)
                context_snippets = [h.page_content for h in hits]
            except Exception:
                context_snippets = []

    # compose messages simply: system + user (with context)
    system = {"role": "system", "content": "You are a helpful assistant that uses provided context to answer."}
    user_content = "Context:\n" + ("\n\n---\n\n".join(context_snippets) if context_snippets else "No context available.") + "\n\nUser question: " + req.query
    user = {"role": "user", "content": user_content}
    try:
        result = chat_groq([system, user])
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="chat generation failed: " + str(e))

    return {"answer": result, "context_count": len(context_snippets)}