# File: rag_app/ingest.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_pdf_to_vectordb(pdf_path: str, index_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(texts, embeddings)
    vectordb.save_local(index_path)
    return vectordb


def load_vectordb(index_path: str):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(index_path, embeddings)