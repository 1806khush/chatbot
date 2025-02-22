from setuptools import find_packages, setup

setup(
    name = 'Medical Chatbot',
    version = '0.0.0',
    author = 'Varun K',
    author_email = 'varun.kolte1@gmail.com',
    packages = find_packages(),
    install_requires = [
        'flask',
        'python-dotenv',
        'langchain',
        'langchain-community',
        'langchain-huggingface',
        'pinecone-client',
        'langchain-pinecone',
        'sentence-transformers',
        'requests',
        'pypdf'
    ]
)