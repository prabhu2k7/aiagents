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

# File: rag_app/classifier.py
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from typing import Dict


def answer_query(query: str, dbs: Dict[str, object]) -> str:
    """
    Perform retrieval over both vectorstores and choose the domain with highest top-document similarity score.
    """
    best_domain = None
    best_score = -float('inf')
    # Compare similarity scores across domains
    for domain, vs in dbs.items():
        # similarity_search_with_score returns list of (doc, score)
        docs_and_scores = vs.similarity_search_with_score(query, k=5)
        if docs_and_scores and docs_and_scores[0][1] > best_score:
            best_score = docs_and_scores[0][1]
            best_domain = domain
    # Retrieve final answer from best domain
    retriever = dbs[best_domain].as_retriever(search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever
    )
    return qa.run(query)