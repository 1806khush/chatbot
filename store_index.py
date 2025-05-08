from src.helper import load_web_content, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
# Import ServerlessSpec if available, otherwise use alternative approach
try:
    from pinecone import ServerlessSpec
    HAS_SERVERLESS = True
except ImportError:
    HAS_SERVERLESS = False
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv("chat.env")

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

# URL of the Wikipedia page to scrape
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Random-access_memory"

# Load content from the Wikipedia URL
extracted_data = load_web_content(WIKIPEDIA_URL)
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "chatbot"

# Check if index exists, if yes, delete it to recreate
try:
    indexes = pc.list_indexes()
    index_names = [index.name for index in indexes] if hasattr(indexes[0], 'name') else indexes
    
    if index_name in index_names:
        print(f"Deleting existing index '{index_name}'")
        pc.delete_index(index_name)
except Exception as e:
    print(f"Error checking indexes: {e}")
    # Alternative approach for older Pinecone versions
    try:
        if pc.index_exists(index_name):
            print(f"Deleting existing index '{index_name}'")
            pc.delete_index(index_name)
    except Exception as e2:
        print(f"Error with alternative approach: {e2}")

print(f"Creating new index '{index_name}'")
try:
    # Try with ServerlessSpec if available
    if HAS_SERVERLESS:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    else:
        # Fallback for older Pinecone versions
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine"
        )
except Exception as e:
    print(f"Error creating index: {e}")
    # Last resort for older API versions
    try:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            environment="us-east-1"
        )
    except Exception as e2:
        print(f"Failed to create index with all methods: {e2}")

print(f"Loading {len(text_chunks)} chunks into Pinecone index")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

print("Index created and data loaded successfully!")