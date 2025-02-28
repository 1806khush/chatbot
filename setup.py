from setuptools import find_packages, setup

setup(
    name='Medical-Chatbot',
    version='0.0.1',
    author='Varun K',
    author_email='varun.kolte1@gmail.com',
    packages=find_packages(),
    install_requires=[
        'flask>=3.0.0',
        'python-dotenv>=1.0.0',
        'langchain>=0.1.0',
        'langchain-community>=0.0.13',
        'langchain-core>=0.1.10',
        'langchain-openai>=0.0.2',
        'sentence-transformers>=2.2.2',
        'pinecone-client[grpc]>=3.0.1',
        'langchain-pinecone>=0.0.2',
        'requests>=2.31.0',
        'pypdf>=3.17.1',
        'gunicorn>=20.1.0'
    ]
)