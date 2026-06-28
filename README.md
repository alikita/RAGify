# RAGify - Chat With Your Documents

RAGify is a production-ready, interview-grade Retrieval-Augmented Generation (RAG) web application. It enables users to upload PDF and TXT documents, automatically indexes them into a persistent vector database, and allows users to have a grounded chat conversation about the documents.

The application is built using **Python, Streamlit, LangChain, ChromaDB**, and supports both **OpenAI GPT-4o-mini** and **Google Gemini** (featuring a generous free tier so you can run the app completely free).

---

## 📐 Architecture Flow

```mermaid
graph TD
    A[Upload PDF/TXT] --> B[Document Loader: PyPDFLoader / TextLoader]
    B --> C[Text Splitter: RecursiveCharacterTextSplitter]
    C --> D[Generate Embeddings: OpenAI / Google Gemini]
    D --> E[(Persistent ChromaDB Vector Store)]
    
    F[User Question] --> G[Query Embedding]
    G --> H[Vector Search in ChromaDB]
    E --> H
    H -->|Top 4 chunks + scores| I[Context Builder]
    I --> J[Prompt Construction: System + Chat History + Context]
    J --> K[LLM: GPT-4o-mini / Gemini-1.5-flash]
    K --> L[Generate Grounded Answer]
    L --> M[Display Answer & Source Citations]

---

## 🔑 How to Get a Free API Key

RAGify is uniquely configured to run **entirely for free** using **Google Gemini**!

### Option A: Google Gemini API (Recommended & 100% Free)
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Log in with your Google account.
3. Click **Get API Key** in the sidebar.
4. Click **Create API Key**, select an existing project or create a new one, and copy the key (usually starts with `AIzaSy...`).
5. Open your `.env` file and set:
   ```env
   GEMINI_API_KEY=your_copied_api_key_here
   ```

### Option B: OpenAI Trial API Key
1. Go to the [OpenAI Platform](https://platform.openai.com/).
2. Register a new account. New accounts may receive $5 in free trial credits (valid for 3 months), but this requires phone verification.
3. Navigate to **API Keys** and click **Create new secret key**.
4. Open your `.env` file and set:
   ```env
   OPENAI_API_KEY=your_copied_api_key_here
   ```

---

## 📁 Project Structure

```
RAGify/
│
├── app.py             # Streamlit Frontend UI & Layout (custom CSS theme)
├── rag.py             # Core RAG pipeline (load, chunk, embed, persistent DB, QA)
├── utils.py           # Helper functions (stats, file validators, history exporter)
├── requirements.txt   # Python package dependencies
├── .env.example       # Template for environment configuration
├── .gitignore         # File to prevent uploading keys or database folder to Git
└── README.md          # Project documentation and guide
```

---

## 🚀 How to Run in VS Code

Follow these steps to run RAGify locally on your machine:

### 1. Open Project in VS Code
- Launch VS Code.
- Go to `File` -> `Open Folder...` and select the `ragify` directory.

### 2. Set Up Virtual Environment
Open the VS Code Terminal (`Ctrl + Shift + \`` or `Terminal -> New Terminal`) and run:
```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate it (Windows Command Prompt)
.\venv\Scripts\activate.bat

# Activate it (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
Make sure your virtual environment is active (you will see `(venv)` at the beginning of the command prompt), then run:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
- In the file explorer, duplicate `.env.example` and rename it to `.env`.
- Open `.env` and insert either your `GEMINI_API_KEY` (highly recommended free option) or `OPENAI_API_KEY`.
- *Note: Do not commit the `.env` file to git.*

### 5. Launch the Application
Start the Streamlit development server:
```bash
streamlit run app.py
```
This will automatically open the application in your default web browser (usually at `http://localhost:8501`).

---

## ✨ Features Checklist
- [x] **Multi-File Uploads**: Supports multiple PDF and TXT files simultaneously.
- [x] **Auto-indexing**: Loads, chunks (Recursive Splitter), embeds (OpenAI or Gemini), and persists.
- [x] **ChatGPT Conversation Interface**: Beautiful user/assistant chat history.
- [x] **Memory Management**: Remembers context from previous questions.
- [x] **Sources & Citation Tracker**: Shows document snippets, file names, page numbers, and similarity metrics.
- [x] **Performance Statistics**: Real-time response speed (seconds), chunk counts, and similarity scores.
- [x] **Download History**: Clean button to export the chat as a `.txt` file.
- [x] **Database Rebuilding**: Wipes the DB and restarts files cache cleanly.

---
