import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings # <-- CORRECTED
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

# Constants
KNOWLEDGE_BASE_FILE = "knowledge_base.txt"
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "medical_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR = "model_cache"  # <-- We will still use this

def main():
    print("--- Starting Data Ingestion Process ---")
    
    try:
        loader = TextLoader(KNOWLEDGE_BASE_FILE, encoding="utf-8")
        documents = loader.load()
        print(f"Successfully loaded '{KNOWLEDGE_BASE_FILE}'.")
    except Exception as e:
        print(f"Error loading knowledge base file: {e}")
        return
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split document into {len(texts)} chunks.")
    
    print(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}'...")
    
    if not os.path.exists(MODEL_CACHE_DIR):
        os.makedirs(MODEL_CACHE_DIR)
            
    # Use the correct class with the cache_folder argument
    embedding_function = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=MODEL_CACHE_DIR  # <-- This fixes the permission error
    )
    print("Embedding model loaded.")
    
    print(f"Creating/loading persistent vector store at '{CHROMA_PERSIST_DIR}'...")
    
    db = Chroma.from_documents(
        documents=texts,
        embedding=embedding_function,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    
    print("--- Data Ingestion Complete ---")
    print(f"Vector store created with {db._collection.count()} documents.")
    print(f"Database saved to '{CHROMA_PERSIST_DIR}'.")

if __name__ == "__main__":
    main()