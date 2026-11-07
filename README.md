# 🤖 RAG Chatbot Using Google Gemini API

## 📌 Overview

This project is a Retrieval-Augmented Generation (RAG) Chatbot that answers user questions based on uploaded documents. The chatbot combines semantic search with Google's Gemini API to generate accurate, context-aware responses from the document content.

The application uses Sentence Transformers for embedding generation, FAISS for vector similarity search, and Streamlit for the user interface.

---

## 🚀 Features

- Document Upload Support
- Text Extraction from Documents
- Text Chunking
- Embedding Generation
- FAISS Vector Database
- Semantic Search & Retrieval
- Context-Aware Question Answering
- Google Gemini API Integration
- Interactive Streamlit Interface

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Google Gemini API
- FAISS
- Sentence Transformers
- NumPy
- PyPDF
- Python-Docx

---

## 📂 Project Workflow

1. Upload Document
2. Extract Text
3. Split Text into Chunks
4. Generate Embeddings
5. Store Embeddings in FAISS
6. Retrieve Relevant Chunks
7. Send Context to Gemini API
8. Generate Accurate Response

---

## 🏗️ Project Architecture

```text
User Query
     │
     ▼
Document Upload
     │
     ▼
Text Extraction
     │
     ▼
Text Chunking
     │
     ▼
Embedding Generation
     │
     ▼
FAISS Vector Database
     │
     ▼
Semantic Search
     │
     ▼
Relevant Context
     │
     ▼
Google Gemini API
     │
     ▼
Generated Answer
```

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Madineni-Pradeep-Naidu/RAG-Chatbot.git
cd RAG-Chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Gemini API Key

Create a `.streamlit/secrets.toml` file:

```toml
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
```

Or use environment variables.

### Run Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
RAG Chatbot/
│
├── app.py
├── requirements.txt
├── RAG_Chatbot.txt
├── .gitignore
├── README.md
│
├── vector_db/
│
└── .streamlit/
    └── secrets.toml
```

---

## ✅ Modules Implemented

- Document Extraction
- Text Chunking
- Embedding Generation
- FAISS Vector Database
- Semantic Search
- Gemini Response Generation

---

## 🎯 Future Enhancements

- Multiple File Upload Support
- Chat History
- Source Citations
- PDF Export of Chats
- Conversation Memory
- Advanced Document Filtering
- Cloud Deployment

---

## 📸 Sample Use Case

**Question:**
> What are the examples of smart materials?

**Response:**
> Shape Memory Alloys, Piezoelectric Materials, Electrochromic Materials, Magnetostrictive Materials, Thermochromic Materials, Photochromic Materials, and Smart Polymers.

---

## 👨‍💻 Author

**Pradeep Naidu**

Aspiring Data Analyst | AI & Machine Learning Enthusiast

GitHub: https://github.com/Madineni-Pradeep-Naidu

---

## ⭐ If you found this project useful, please consider giving it a star!
