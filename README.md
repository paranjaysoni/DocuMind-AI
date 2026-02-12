# 📘 DocuMind AI

> Chat with your PDFs intelligently using Retrieval-Augmented Generation (RAG)

DocuMind AI is a conversational AI web application that allows users to upload a PDF and interact with its content using natural language.

It uses semantic search, vector embeddings, history-aware retrieval, and large language models to generate accurate, grounded responses based strictly on the uploaded document.

---

## 🚀 Features

- 📄 Upload a PDF file
- 🧠 Semantic search using embeddings
- 🔍 Context-aware retrieval
- 💬 Conversational memory support
- ⚡ Fast LLM responses using Groq
- 📊 LangSmith tracing enabled
- 🔐 Secure API key handling using Streamlit secrets
- 🔄 Reset session functionality

---

## 🏗️ Architecture Overview
User Question
↓
History-Aware Query Reformulation
↓
Vector Similarity Search (Chroma)
↓
Relevant Document Chunks
↓
LLM (LLaMA 3.3 via Groq)
↓
Grounded Answer

---

## 🧠 How It Works

### 1️⃣ PDF Processing
- PDF is loaded using `PyPDFLoader`
- Text is split into chunks using `RecursiveCharacterTextSplitter`
- Chunk overlap preserves semantic continuity

### 2️⃣ Embeddings
- Each chunk is converted into vector embeddings using: all-MiniLM-L6-v2

### 3️⃣ Vector Database
- Embeddings are stored in an in-memory `Chroma` vector database
- Enables semantic similarity search

### 4️⃣ History-Aware Retrieval
- User queries are reformulated into standalone questions
- Retriever fetches relevant document chunks
- Reduces ambiguity in follow-up questions

### 5️⃣ Response Generation
- LLM generates response using:
- Retrieved context
- Chat history
- Structured system prompts
- Ensures grounded answers

---

## 🛠️ Tech Stack

- Frontend - Streamlit
- LLM - Groq (LLaMA 3.3 70B)
- Embeddings - HuggingFace
- Vector DB - Chroma
- Orchestration - LangChain
- Observability - LangSmith 

---

## 📂 Project Structure
DocuMind-AI/
│
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml

---

## 🔐 Environment Setup

### 1️⃣ Clone the repository
git clone https://github.com/paranjaysoni/DocuMind-AI.git
cd DocuMind-AI

### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Create `.streamlit/secrets.toml`
GROQ_API_KEY = "your_groq_api_key"
HF_TOKEN = "your_huggingface_token"
LANGCHAIN_API_KEY = "your_langsmith_api_key"
LANGCHAIN_PROJECT = "DocuMind-AI"


---

## ▶️ Run the App
streamlit run app.py


---

## 📊 LangSmith Tracing
Tracing is enabled automatically using:
LANGCHAIN_TRACING_V2 = true

This allows:

- Input/output inspection
- Retriever debugging
- Token usage tracking
- Full pipeline observability

---

## 🔄 Session Handling

- Start New Session button clears:
  - Chat memory
  - Uploaded file
  - Conversation state
- File uploader resets dynamically using unique keys

---

## 🚀 Deployment

This project can be deployed on:

- Streamlit Cloud
- Render
- HuggingFace Spaces
- Railway

Make sure to configure secrets in the deployment dashboard.

---

## 📈 Future Improvements

- Source citation display
- Streaming responses
- PDF preview panel
- Multi-user support
- Cloud-based persistent vector DB
- Authentication layer

---

## 👨‍💻 Author

**Paranjay Soni**

- GitHub: https://github.com/paranjaysoni
- LinkedIn: linkedin.com/in/paranjaysoni

---

## ⭐ If You Found This Useful

Give this repository a star ⭐ and feel free to fork or contribute!
