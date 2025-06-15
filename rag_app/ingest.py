from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_pdf_to_vectordb(pdf_path: str, index_path: str):
    # load & split
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)
    # embed & store
    embeddings = OpenAIEmbeddings()
    vectordb   = FAISS.from_documents(texts, embeddings)
    vectordb.save_local(index_path)
    return vectordb