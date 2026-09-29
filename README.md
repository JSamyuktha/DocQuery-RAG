# 📄 DocQuery-RAG

DocQuery-RAG is a simple Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content.

Instead of asking an LLM to answer from general knowledge, the application first retrieves relevant sections from the uploaded document and provides them as context to the language model. This helps generate answers grounded in the document.

## ✨ Features

- Upload PDF documents through a simple web interface
- Extract text while preserving page information
- Split document text into overlapping chunks
- Generate semantic embeddings for document chunks
- Store and search embeddings using ChromaDB
- Retrieve the most relevant document sections for a question
- Generate context-based answers using Gemini
- Display source page numbers for retrieved information
- View the retrieved document sections used as context

## 🧠 How It Works

The application follows a basic RAG pipeline:

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
ChromaDB Vector Store
    ↓
User Question
    ↓
Semantic Retrieval
    ↓
Relevant Document Chunks
    ↓
Gemini
    ↓
Grounded Answer + Source Pages
```

### 1. PDF Processing

The uploaded PDF is read using PyPDF. Text is extracted page by page so that page numbers can be preserved for source references.

### 2. Text Chunking

Extracted text is divided into overlapping chunks. Overlap helps retain context that may otherwise be lost when text is split.

### 3. Embeddings

The `all-MiniLM-L6-v2` Sentence Transformer model converts each text chunk into a numerical embedding representing its semantic meaning.

### 4. Vector Storage

The embeddings, document chunks, and page metadata are stored in ChromaDB.

### 5. Semantic Retrieval

When the user asks a question, the question is also converted into an embedding. ChromaDB compares it with the stored document embeddings and retrieves the most relevant chunks.

### 6. Answer Generation

The retrieved chunks are supplied to Gemini as context. The prompt instructs the model to answer using only the retrieved document information and avoid adding unsupported information.

The application also displays the relevant source pages and allows the user to inspect the retrieved text.

## 🛠️ Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB
- Google Gemini API
- Google GenAI Python SDK
- python-dotenv
- Git & GitHub

## 📁 Project Structure

```text
DocQuery-RAG/
│
├── app.py
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
└── documents/
```

`rag.py` contains the document processing, embedding, retrieval, and answer-generation logic.

`app.py` contains the Streamlit user interface.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/JSamyuktha/DocQuery-RAG.git
cd DocQuery-RAG
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the project directory.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git through `.gitignore` so API credentials are not committed to the repository.

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit URL displayed in the terminal.

Upload a PDF, click **Process Document**, enter a question, and click **Ask Question**.

## 💡 Example

**Question:**

> What are the main challenges and limitations of RAG?

The system retrieves relevant sections from the document and generates an answer based on those sections.

Example topics identified from the test document included:

- Security vulnerabilities
- Retrieval poisoning
- Operational and efficiency trade-offs
- Limitations of mitigation strategies

The application also displays the source pages and retrieved document sections.

## 🔒 Security

API keys are stored locally using environment variables and are not committed to GitHub.

The following files/directories are ignored:

```text
.env
venv/
__pycache__/
chroma_db/
documents/*.pdf
```

## 🚀 Future Improvements

Possible improvements include:

- Support for multiple PDFs
- Improved text chunking strategies
- Hybrid keyword and semantic retrieval
- Retrieval reranking
- Persistent vector storage
- Conversation history
- Support for additional document formats
- Improved source citation and retrieval evaluation

## 📌 Project Status

Working prototype.

The current version supports PDF upload, semantic document retrieval, grounded question answering, source page display, and inspection of retrieved document sections.