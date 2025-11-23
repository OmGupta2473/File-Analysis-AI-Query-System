# PDF RAG Chatbot (Ollama + Streamlit)

A full-featured PDF-based RAG (Retrieval-Augmented Generation) system that allows users to upload PDFs or use a sample document, converts them into vector embeddings, and lets users chat with the content. Designed for learning, portfolio work, and demonstrating AI + RAG workflow skills.

## Features
- Upload any PDF and generate embeddings  
- Use sample PDF (Scammer-Agent) for testing  
- FAISS vector database for fast retrieval  
- Multi-query retrieval using LangChain  
- Local LLM inference using Ollama (llama3.2, nomic-embed-text)  
- Clean and responsive Streamlit UI  
- Delete vector DB with one click  
- Chat-style interface with memory  
- Logging for debugging and tracing  

## Folder Structure
```
├── app.py              # Main Streamlit RAG application
├── botconfig.py        # Config values (model names, sample PDF paths)
├── requirements.txt    # Python dependencies
├── pakages.txt         # System dependencies (Tesseract, Poppler)
├── start.sh            # Setup + ollama commands
└── README.md           # Project documentation
```

## Technologies Used
- **LLM Frameworks:** LangChain, Ollama  
- **Embeddings:** nomic-embed-text  
- **Vector DB:** FAISS  
- **UI Framework:** Streamlit  
- **Document Handling:** Unstructured, PDF loaders  
- **Backend:** Python  
- **Other:** Tesseract OCR, Poppler utilities  

## Getting Started

### Prerequisites
- Python 3.10+  
- pip installed  
- **Ollama installed** (required for LLM + embeddings)  
- Poppler (for PDF parsing)  
- Tesseract-OCR (for OCR PDFs)

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Install System Packages
```bash
sudo apt update
sudo apt install tesseract-ocr poppler-utils
```

### Install Ollama & Pull Models
```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2
ollama pull llama3.1
ollama pull nomic-embed-text

ollama run llama3.2 &
ollama run llama3.1 &
```

## Running the App
```bash
streamlit run app.py
```

## Environment Notes
Your config file includes:
- Sample PDF path  
- Embedding model: `nomic-embed-text`  
- Collection name: `myRAG`

## Contribution Guidelines
1. Fork the repository  
2. Create a new branch (`git checkout -b feature/your-feature`)  
3. Commit your changes  
4. Push to your branch and open a Pull Request  
