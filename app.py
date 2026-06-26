"""
app.py - RAGify Frontend Application

This module implements the Streamlit user interface for RAGify:
1. Sidebar panel for API status, file uploads, document indexing, and statistics.
2. Premium custom CSS styling (Indigo scheme, Outfit typography, custom metrics cards).
3. ChatGPT-like conversation interface with memory and loading spinners.
4. Explanatory metrics dashboard (similarity scores, search speed, chunks searched).
5. Grounded citation expansion panels showing text snippets and page numbers.
6. Chat log history exporter (text downloads) and database reindexing.
"""

import os
import shutil
import streamlit as st
import time
from dotenv import load_dotenv

# Import our core RAG & utility functions
from rag import build_index, ask_question, clear_vector_store, get_api_provider
from utils import validate_file, generate_history_download, extract_meta_statistics

# 1. Page Configuration and Initialization
st.set_page_config(
    page_title="RAGify - Chat With Your Documents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure upload directory exists
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 2. Inject Premium Custom CSS for styling
st.markdown("""
<style>
    /* Google Fonts & Core styling */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global Primary Colors */
    .stApp {
        background-attachment: fixed;
    }
    
    /* Metrics dashboard cards */
    .metric-card {
        background: rgba(79, 70, 229, 0.05);
        border: 1px solid rgba(79, 70, 229, 0.15);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #4F46E5;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 12px;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Custom Sidebar Stats Cards */
    .sidebar-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .sidebar-val {
        font-size: 18px;
        font-weight: 700;
        color: #4F46E5;
    }
    .sidebar-lbl {
        font-size: 11px;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Dark Mode Compatibility overrides for cards */
    @media (prefers-color-scheme: dark) {
        .sidebar-card {
            background: #1F2937;
            border-color: #374151;
        }
        .sidebar-val {
            color: #818CF8;
        }
        .metric-card {
            background: rgba(129, 140, 248, 0.07);
            border-color: rgba(129, 140, 248, 0.2);
        }
        .metric-value {
            color: #818CF8;
        }
    }
    
    /* Chat Citations Block style */
    .source-block {
        border-left: 3px solid #4F46E5;
        padding-left: 12px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 0.9em;
        color: #4B5563;
        background: rgba(243, 244, 246, 0.5);
        border-radius: 0 8px 8px 0;
        padding-top: 6px;
        padding-bottom: 6px;
    }
    
    @media (prefers-color-scheme: dark) {
        .source-block {
            color: #D1D5DB;
            background: rgba(31, 41, 55, 0.5);
            border-left-color: #818CF8;
        }
    }

    /* Primary Buttons configuration */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
</style>
""", unsafe_allow_html=True)

# 3. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    # Check if vector DB already exists on disk (app restart persistence)
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    st.session_state.indexed = os.path.exists(db_path) and len(os.listdir(db_path)) > 0
if "stats" not in st.session_state:
    st.session_state.stats = None
if "files_processed" not in st.session_state:
    st.session_state.files_processed = []

# Detect Active API Provider
provider = get_api_provider()

# 4. HEADER
st.title("🤖 RAGify")
st.markdown("### Chat with your PDFs and Text Documents")
st.write("A production-ready RAG application supporting OpenAI GPT-4o-mini and Google Gemini free tiers.")

# Warning if API key is missing
if provider == "none":
    st.warning("⚠️ **API Keys Not Configured!** Please copy `.env.example` to `.env` in the project directory and fill in your `OPENAI_API_KEY` or `GEMINI_API_KEY` to activate the AI. If you want a free API key, see details in the README or the instructions below.")

# 5. SIDEBAR
with st.sidebar:
    st.image("assets/logo.png", width=120)
    st.markdown("## Configuration & Controls")
    
    # API Indicator
    if provider == "openai":
        st.success("🟢 Connected: **OpenAI Engine**")
    elif provider == "gemini":
        st.success("🟢 Connected: **Gemini Engine (Free Tier)**")
    else:
        st.error("🔴 Disconnected: **Missing Key**")
        
    st.divider()
    
    # Document Upload Area
    st.markdown("### 📥 1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    # Save uploaded files and prepare path lists
    saved_paths = []
    if uploaded_files:
        for file in uploaded_files:
            # Validate size & extension
            is_valid, err_msg = validate_file(file.name, file.size)
            if not is_valid:
                st.error(err_msg)
                continue
            
            # Save file to server disk
            file_path = os.path.join(DATA_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(file_path)
            
        if saved_paths and not st.session_state.indexed:
            st.info("Files uploaded! Click the button below to build your vector database.")
            
    # Indexing Trigger Button
    if saved_paths:
        if st.button("🏗️ Index Documents", use_container_width=True):
            with st.spinner("Processing documents into vector DB..."):
                result = build_index(saved_paths)
                
                if result.get("success"):
                    st.session_state.indexed = True
                    st.session_state.files_processed = saved_paths
                    st.session_state.stats = extract_meta_statistics(
                        result.get("files", []), 
                        result.get("num_chunks", 0)
                    )
                    st.success("✅ Vectors successfully created & saved!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Indexing failed: {result.get('error')}")

    st.divider()

    # Document Statistics Dashboard
    st.markdown("### 📊 Document Statistics")
    if st.session_state.indexed and st.session_state.stats:
        stats = st.session_state.stats
        
        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-val">{stats['total_files']}</div>
            <div class="sidebar-lbl">Files Indexed</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-val">{stats['total_pages']}</div>
            <div class="sidebar-lbl">Total Pages</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-val">{stats['total_chunks']}</div>
            <div class="sidebar-lbl">Chunks Created</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-val">{stats['total_size_friendly']}</div>
            <div class="sidebar-lbl">Total Data Size</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No documents indexed. Upload files above to view statistics.")

    st.divider()
    
    # Database and Chat Actions
    st.markdown("### ⚙️ Reset and Export")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Rebuild Index", use_container_width=True, help="Wipes the database and clears the indexed files"):
            # Clear storage directories
            clear_vector_store()
            if os.path.exists(DATA_DIR):
                shutil.rmtree(DATA_DIR)
                os.makedirs(DATA_DIR)
            
            # Reset state
            st.session_state.indexed = False
            st.session_state.stats = None
            st.session_state.files_processed = []
            st.session_state.messages = []
            st.success("Index wiped!")
            time.sleep(1)
            st.rerun()
            
    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True, help="Clears conversation memory"):
            st.session_state.messages = []
            st.success("Chat history cleared!")
            time.sleep(1)
            st.rerun()

    # Export Chat History Button
    if st.session_state.messages:
        chat_download_data = generate_history_download(st.session_state.messages)
        st.download_button(
            label="💾 Export Chat Log (.txt)",
            data=chat_download_data,
            file_name="ragify_chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )

# 6. MAIN CHAT AREA
if not st.session_state.indexed:
    st.info("👉 **Welcome to RAGify!** To get started, upload some documents (PDF or TXT) in the sidebar on the left and click **Index Documents**.")
    
    # Quick guide for free keys
    with st.expander("🔑 How to get a free API Key?", expanded=True):
        st.markdown("""
        Don't have budget for OpenAI? Don't worry! You can use **Google Gemini** for free:
        
        1. **Get a Google Gemini API Key (Recommended & 100% Free)**:
           - Visit [Google AI Studio](https://aistudio.google.com/).
           - Click **Get API Key** and create a key.
           - Copy the key.
        2. **Configure your project**:
           - Open the `ragify` directory in VS Code.
           - Rename `.env.example` to `.env` (or create a new file named `.env`).
           - Set the variable: `GEMINI_API_KEY=your_gemini_key_here`.
        3. **Run RAGify**:
           - The system will detect your Gemini key and automatically use free Gemini embeddings and LLMs!
        """)
else:
    # Display Chat History using native Streamlit chat UI
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show sources if this was an assistant message with citations
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Sources Used & Similarity metrics"):
                    for idx, chunk in enumerate(msg["sources"], 1):
                        st.markdown(f"""
                        <div class="source-block">
                            <strong>Source {idx} - {os.path.basename(chunk['source'])} (Page {chunk['page']})</strong><br/>
                            <em>Similarity Score: {chunk['similarity_score']:.4f}</em><br/>
                            <p style='margin-top: 5px; font-style: normal;'>"{chunk['content']}"</p>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input Field
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        # Check API key before making requests
        if provider == "none":
            st.error("Cannot query model. Please set a valid OPENAI_API_KEY or GEMINI_API_KEY in your .env file.")
        else:
            # 1. Display user query
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})

            # 2. Query LLM and generate response with loading spinner
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document chunks & generating grounded answer..."):
                    # Call RAG pipeline with conversation memory
                    result = ask_question(user_query, st.session_state.messages[:-1])
                    
                    if result.get("success"):
                        answer = result["answer"]
                        sources = result["source_chunks"]
                        metrics = result["metrics"]
                        
                        # Display AI answer
                        st.markdown(answer)
                        
                        # Display Metrics Dashboard in columns
                        st.markdown("<br/>", unsafe_allow_html=True)
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{metrics['response_time']}s</div>
                                <div class="metric-label">Response Time</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{metrics['chunks_searched']}</div>
                                <div class="metric-label">Chunks Searched</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m3:
                            # Display highest similarity score
                            best_score = max(metrics['similarity_scores']) if metrics['similarity_scores'] else 0.0
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{best_score:.4f}</div>
                                <div class="metric-label">Max Similarity Score</div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("<br/>", unsafe_allow_html=True)
                        
                        # Display Source Citations
                        with st.expander("📚 Sources Used & Similarity metrics"):
                            for idx, chunk in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-block">
                                    <strong>Source {idx} - {os.path.basename(chunk['source'])} (Page {chunk['page']})</strong><br/>
                                    <em>Similarity Score: {chunk['similarity_score']:.4f}</em><br/>
                                    <p style='margin-top: 5px; font-style: normal;'>"{chunk['content']}"</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        # Save answer and sources to session history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                        # Scroll to bottom
                        st.rerun()
                    else:
                        error_msg = result.get("error", "Failed to retrieve an answer.")
                        st.error(error_msg)
