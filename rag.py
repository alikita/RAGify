"""
rag.py - Core Retrieval-Augmented Generation (RAG) Pipeline

This module implements the core RAG logic:
1. Document loading (PDF and TXT)
2. Document splitting (chunking)
3. Embedding generation (OpenAI or Google Gemini)
4. Vector storage (persistent ChromaDB)
5. Question answering with source citations and conversation memory

================================================================================
INTERVIEW READINESS EXPLANATIONS:
================================================================================
1. WHAT IS RAG?
   Retrieval-Augmented Generation (RAG) is a technique that optimizes the output of 
   Large Language Models (LLMs) by querying an external knowledge base (like a vector 
   DB) before generating the response. This ensures the LLM's answer is grounded in 
   the provided documents, drastically reducing "hallucinations" and keeping information 
   current without retraining.

2. WHY IS CHUNKING NEEDED?
   LLMs have a finite context window (input limit). Large documents cannot be passed in 
   their entirety. Splitting documents into smaller, overlapping segments (chunks) 
   ensures that only the most relevant portions of text are sent to the LLM. Overlap 
   (e.g., 100 characters) ensures that context spanning across boundaries isn't lost.

3. WHY ARE EMBEDDINGS USED?
   Embeddings convert text into dense, high-dimensional numerical vectors (e.g., 1536 
   dimensions). In this vector space, texts with similar semantic meanings are placed 
   close to each other. This allows the system to understand that "dog" and "puppy" 
   are related, which keyword-based search might miss.

4. HOW DOES VECTOR SEARCH WORK?
   When a user asks a question, we generate an embedding vector for that question. 
   We then calculate a mathematical similarity metric (like Cosine Similarity or L2 
   distance) between the question vector and all the stored document chunk vectors. 
   The chunks with the highest similarity scores are retrieved as context.

5. WHY CHROMADB?
   ChromaDB is an open-source vector database designed specifically for developer 
   workflows. It runs locally, offers rapid setup, persists data directly to disk (avoiding 
   in-memory loss), and integrates natively with LangChain.
================================================================================
"""

import os
import time
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Provider-Specific Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

# Constants
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def get_api_provider() -> str:
    """
    Detects which API key is available in the environment.
    Prioritizes Gemini if GEMINI_API_KEY is present, otherwise falls back to OpenAI.
    """
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "none"

def get_embeddings_model():
    """
    Returns the appropriate embedding model based on the active API provider.
    """
    provider = get_api_provider()
    if provider == "gemini":
        # Google Generative AI Embeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    elif provider == "openai":
        # OpenAI Embeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(
            "No API keys found. Please set either OPENAI_API_KEY or GEMINI_API_KEY in your .env file."
        )

def get_llm_model():
    """
    Returns the appropriate LLM model based on the active API provider.
    """
    provider = get_api_provider()
    if provider == "gemini":
        # Google Gemini 2.5 Flash (Free tier available)
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    elif provider == "openai":
        # OpenAI GPT-4o-mini (Cost-effective and fast)
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(
            "No API keys found. Please set either OPENAI_API_KEY or GEMINI_API_KEY in your .env file."
        )

def build_index(file_paths: List[str]) -> Dict[str, Any]:
    """
    Loads documents from file paths, splits them into semantic chunks,
    generates embeddings, and stores them in a persistent Chroma vector store.

    Args:
        file_paths: List of absolute file paths (PDF or TXT)

    Returns:
        dict: Statistics about the indexing process
    """
    # 1. Input Validation
    if not file_paths:
        return {"success": False, "error": "No files provided for indexing."}
    
    # 2. Check for API key before processing to fail fast
    try:
        embeddings = get_embeddings_model()
    except Exception as e:
        return {"success": False, "error": str(e)}

    all_docs = []
    total_pages = 0
    file_details = []

    # 3. Load Documents
    for path in file_paths:
        if not os.path.exists(path):
            continue
        
        file_name = os.path.basename(path)
        file_ext = os.path.splitext(file_name)[1].lower()
        file_size_bytes = os.path.getsize(path)
        
        try:
            if file_ext == ".pdf":
                loader = PyPDFLoader(path)
                docs = loader.load()
                total_pages += len(docs)
            elif file_ext == ".txt":
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
                total_pages += 1  # Text files count as 1 page
            else:
                continue  # Skip unsupported extensions
            
            all_docs.extend(docs)
            file_details.append({
                "name": file_name,
                "size": file_size_bytes,
                "pages": len(docs)
            })
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load file '{file_name}': {str(e)}"
            }

    if not all_docs:
        return {"success": False, "error": "No valid text could be extracted from the files."}

    # 4. Split Documents
    # RecursiveCharacterTextSplitter attempts to split on paragraphs, then sentences,
    # then words, trying to keep chunks as close to chunk_size as possible.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = splitter.split_documents(all_docs)

    # 5. Store in Persistent ChromaDB
    try:
        # If the directory already exists, clear the collection instead of deleting the folder
        # to avoid WinError 32 file-locking issues on Windows when the database is active.
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            try:
                db_temp = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
                db_temp.delete_collection()
                db_temp.persist()
                db_temp = None
                import gc
                gc.collect()
                time.sleep(0.3)
            except Exception:
                pass
            
        # Initialize and build the Chroma store (automatically saves to disk)
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        db.persist()
        
        return {
            "success": True,
            "num_files": len(file_paths),
            "total_pages": total_pages,
            "num_chunks": len(chunks),
            "files": file_details
        }
    except Exception as e:
        return {"success": False, "error": f"Vector store storage failed: {str(e)}"}

def clear_vector_store() -> bool:
    """Clears the persistent Chroma vector store by resetting its collection."""
    if os.path.exists(CHROMA_DIR):
        try:
            embeddings = get_embeddings_model()
            db_temp = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            db_temp.delete_collection()
            db_temp.persist()
            db_temp = None
            import gc
            gc.collect()
            time.sleep(0.3)
            
            # Optionally attempt to delete the directory, but fail silently if locked
            import shutil
            try:
                shutil.rmtree(CHROMA_DIR)
            except Exception:
                pass
            return True
        except Exception:
            return False
    return True

def ask_question(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Retrieves relevant chunks from the vector database, formats the context,
    includes prior conversation memory, and queries the LLM for a grounded answer.

    Args:
        question: The user's query
        chat_history: Optional list of prior messages, e.g., [{"role": "user", "content": "..."}]

    Returns:
        dict: Answer, source chunks (text + scores), and retrieval performance metrics
    """
    start_time = time.time()
    
    # 1. Basic Validations
    if not question.strip():
        return {"success": False, "error": "The question cannot be empty."}
        
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        return {
            "success": False,
            "error": "No documents indexed yet. Please upload files in the sidebar first."
        }

    try:
        embeddings = get_embeddings_model()
        llm = get_llm_model()
    except Exception as e:
        return {"success": False, "error": str(e)}

    try:
        # 2. Load the persistent Vector DB
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        
        # 3. Retrieve top 4 most relevant chunks
        # similarity_search_with_relevance_scores returns tuples of (Document, score)
        # Note: Chroma's score mapping can vary; we normalize it nicely.
        raw_results = db.similarity_search_with_relevance_scores(question, k=4)
        
        if not raw_results:
            return {
                "success": False,
                "error": "Could not find any relevant information in the documents. Try asking a different question."
            }

        # Format retrieved chunks
        source_chunks = []
        context_parts = []
        
        for i, (doc, score) in enumerate(raw_results, 1):
            # Normalize scores to 0-1 range for user-friendliness if necessary
            # (Chroma distance can be L2, cosine, etc. usually mapped to similarity score)
            normalized_score = float(score)
            if normalized_score < 0:
                normalized_score = 0.0
            elif normalized_score > 1:
                normalized_score = 1.0
                
            source_chunks.append({
                "chunk_id": i,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0) + 1,  # 0-indexed to 1-indexed
                "similarity_score": normalized_score
            })
            
            # Format context with citation numbers
            context_parts.append(f"[Source {i} - File: {os.path.basename(doc.metadata.get('source', 'Unknown'))} (Page {doc.metadata.get('page', 0) + 1})]:\n{doc.page_content}")

        context_str = "\n\n".join(context_parts)

        # 4. Formulate Prompt Messages (System + History + User)
        system_instructions = (
            "You are RAGify, a professional, precise AI assistant that answers questions based on uploaded documents.\n"
            "Use the provided Document Context to answer the user's question.\n"
            "Rules:\n"
            "1. Answer the question thoroughly, drawing information ONLY from the provided context.\n"
            "2. Cite your sources in the text using bracketed numbers corresponding to the sources in the context, e.g. [Source 1] or [Source 2].\n"
            "3. If the context does not contain the answer, state clearly that you cannot find the answer in the provided documents. Do not hallucinate or use external knowledge."
        )
        
        messages = [
            SystemMessage(content=system_instructions)
        ]
        
        # Incorporate Chat History for Conversation Memory
        if chat_history:
            # Add up to 6 turns of history to prevent context explosion
            for msg in chat_history[-6:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    # Remove citation summaries block from assistant history if present
                    content = msg["content"]
                    if "---" in content:
                        content = content.split("---")[0].strip()
                    messages.append(AIMessage(content=content))

        # Add current question with context
        user_message_content = (
            f"DOCUMENT CONTEXT:\n{context_str}\n\n"
            f"QUESTION:\n{question}"
        )
        messages.append(HumanMessage(content=user_message_content))

        # 5. Query LLM
        response = llm.invoke(messages)
        answer = response.content

        # Calculate time elapsed
        elapsed_time = time.time() - start_time

        return {
            "success": True,
            "answer": answer,
            "source_chunks": source_chunks,
            "metrics": {
                "chunks_searched": len(source_chunks),
                "similarity_scores": [c["similarity_score"] for c in source_chunks],
                "response_time": round(elapsed_time, 3)
            }
        }

    except Exception as e:
        # Check for typical API Key issues
        error_msg = str(e)
        if "API key" in error_msg or "apikey" in error_msg or "authentication" in error_msg.lower():
            return {
                "success": False,
                "error": "Authentication Failed. Please check that your API Key in the .env file is correct and active."
            }
        return {
            "success": False,
            "error": f"An error occurred while answering: {error_msg}"
        }
