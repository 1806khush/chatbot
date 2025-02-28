# Tech Stack

An intelligent medical assistant chatbot built with Flask and LangChain that provides accurate medical information using RAG (Retrieval Augmented Generation). The chatbot leverages Pinecone for vector storage and HuggingFace embeddings to retrieve relevant medical knowledge from a curated database.

## Frontend
- HTML
- CSS (Bootstrap)
- JavaScript (jQuery)

## Libraries & Dependencies
- langchain-community (Document loaders, embeddings)
- langchain-pinecone (Vector store integration)
- sentence-transformers (HuggingFace embeddings)
- python-dotenv (Environment variables)
- PyPDF (PDF document processing)
- requests (API calls)

## Development Tools
- Python 3.8+
- Virtual Environment (venv)
- pip (Package installer)

## APIs
- OpenRouter API (for LLM integration)
- Pinecone API (for vector database)

## Document Processing
- HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)
- RecursiveCharacterTextSplitter (Document chunking)
- PyPDFLoader (PDF processing) 