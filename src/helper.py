import os
try:
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader, DirectoryLoader

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.web_scraper import fetch_wikipedia_content

def load_pdf_file(data):
    """Load PDF files from a directory"""
    pdf_loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return pdf_loader.load()

def load_web_content(url):
    """
    Load content from a web URL
    
    Args:
        url (str): URL to fetch content from
        
    Returns:
        list: List containing a Document object with the web content
    """
    content = fetch_wikipedia_content(url)
    
    # Create a Document object (compatible with langchain)
    document = Document(
        page_content=content,
        metadata={"source": url}
    )
    
    return [document]

def text_split(extracted_data):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    return text_splitter.split_documents(extracted_data)

def download_hugging_face_embeddings():
    """Download and return HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    return embeddings