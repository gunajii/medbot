import os
import uvicorn
import asyncio
from fastapi import FastAPI, Response # <-- Import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from starlette.responses import StreamingResponse
from deep_translator import GoogleTranslator

#--- FastAPI App Initialization ---
app = FastAPI(
    title="Al Medical Chatbot API",
    description="An API for a private, locally-run Al medical diagnosis and analysis chatbot.",
    version="1.0.0",
)

#--- CORS Middleware (Updated to be more explicit) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (including your GitHub page and localhost)
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"], # Explicitly allow all methods
    allow_headers=["*"], # Explicitly allow all headers
)

#--- Global Variables (Initialized on Startup) ---
vector_store = None
llm = None
prompt_template = None

#--- Pydantic Models for Request/Response ---
class ChatRequest(BaseModel):
    query: str

class TranslationRequest(BaseModel):
    text: str
    target_lang: str
    source_lang: str = "auto"

#--- Constants ---
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "medical_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
MODEL_CACHE_DIR = "model_cache"
LLM_MODEL_NAME = "phi3:mini" 

#--- Application Startup Event ---
@app.on_event("startup")
def startup_event():
    global vector_store, llm, prompt_template
    print("--- Server starting up: Initializing RAG components ---")
    
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Error: ChromaDB directory '{CHROMA_PERSIST_DIR}' not found.")
        return

    embedding_function = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=MODEL_CACHE_DIR
    )
    
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME,
    )
    print("Vector store loaded.")
    
    template = """
You are ArogyaAi, an AI medical assistant developed by Gunaji. Your role is to provide helpful, safe, and evidence-based medical information.
Your answers should be helpful but concise.
You must answer the user's question based *only* on the context provided below.
If the context does not contain the answer, state that you do not have enough information and advise the user to consult a healthcare professional.
Do not make up information or provide medical advice, diagnoses, or prescriptions.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
    
    llm = ChatOllama(model=LLM_MODEL_NAME)
    print(f"LLM '{LLM_MODEL_NAME}' initialized.")
    print("--- RAG components successfully initialized. Server is ready. ---")

#--- API Endpoints ---

# --- NEW: Manually handle the OPTIONS preflight requests ---
@app.options("/chat")
async def chat_options():
    return Response(status_code=200)

@app.options("/translate")
async def translate_options():
    return Response(status_code=200)
# ---------------------------------------------------------


@app.get("/", summary="Root endpoint for health check")
def read_root():
    return {"status": "ok", "message": "Al Medical Chatbot API is running."}

async def stream_chat_response(query: str) -> AsyncGenerator[str, None]:
    if not vector_store:
        error_message = "Vector store is not initialized. Please check server startup logs."
        yield f"data: {error_message}\n\n"
        return
        
    try:
        print(f"Searching for context for: '{query}'")
        retrieved_docs = await vector_store.asimilarity_search(query, k=5)
        
        if not retrieved_docs:
            print("No relevant context found in the database.")
            no_context_message = "I do not have enough information about that topic to provide an answer. Disclaimer: I am an Al assistant and not a substitute for professional medical advice. Please consult a qualified healthcare provider for any health concerns."
            for char in no_context_message:
                yield f"data: {char}\n\n"
                await asyncio.sleep(0.01) 
            return

        print(f"Found {len(retrieved_docs)} relevant documents.")
        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        formatted_prompt = prompt_template.format(context=context_str, question=query)
        
        llm_stream = llm.astream(formatted_prompt)
        
        async for chunk in llm_stream:
            yield f"data: {chunk.content}\n\n"
            
    except Exception as e:
        print(f"Error during streaming: {e}")
        yield f"data: An error occurred: {e}\n\n"

@app.post("/chat", summary="Process a user's chat query")
async def chat(request: ChatRequest):
    return StreamingResponse(
        stream_chat_response(request.query),
        media_type="text/event-stream"
    )

@app.post("/translate", summary="Translate text using Google Translate")
async def translate_text(request: TranslationRequest):
    try:
        result = await asyncio.to_thread(
            GoogleTranslator(source=request.source_lang, target=request.target_lang).translate,
            request.text
        )
        return {"translated_text": result}
    except Exception as e:
        print(f"Google Translate Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("--- Starting server directly (via main.py) ---")
    uvicorn.run(app, host="127.0.0.1", port=8000)