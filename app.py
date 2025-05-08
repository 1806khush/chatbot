from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.llms.base import LLM
import requests
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import Any, Optional
import os

load_dotenv("chat.env")

class DeepseekLLM(LLM):
    api_key: str
    temperature: float = 0.4
    max_tokens: int = 500

    def __init__(self, api_key: str, temperature: float = 0.4, max_tokens: int = 500, **kwargs: Any):
        super().__init__(
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        headers = {
            "HTTP-Referer": "https://tech-chatbot.com",  
            "X-Title": "Tech Chatbot",  
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "openai/gpt-3.5-turbo",  
            "messages": [
                {"role": "system", "content": "You are a helpful tech assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions", 
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            response_json = response.json()
            print("API Response:", response_json)
            
            if 'error' in response_json:
                raise ValueError(f"API Error: {response_json['error']}")
                
            return response_json["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {e}")
            return "I apologize, but I encountered an error processing your request."
        except (KeyError, IndexError, ValueError) as e:
            print(f"Response Processing Error: {e}")
            return "I apologize, but I received an unexpected response format."
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return "I apologize, but something went wrong."

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = DeepseekLLM(
    api_key=OPENROUTER_API_KEY,
    temperature=0.4,
    max_tokens=500
)

# System prompt for RAM-focused chatbot
system_prompt = """
You are a knowledgeable technical assistant specializing in computer memory, particularly Random-Access Memory (RAM).
Use the following pieces of context to answer the user's question about RAM.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep your answers technical but understandable, thorough, and accurate.

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response: ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)