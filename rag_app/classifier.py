# rag_app/classifier.py

from typing import Dict
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Expanded keyword list so that channel‐engineering questions hit oncology
DOMAIN_KEYWORDS = {
    "oncology": [
        "cancer", "tumor", "checkpoint", "egfr", "hypoxia",
        "brca", "immunotherapy", "car-t", "leukemia", "ovarian",
        "kcsa", "channel", "ph", "proton", "sensor", "activation",
        "gate", "bundle", "helical"
    ],
    "neurology": [
        "neuro", "microglia", "demyelination", "tau", "synapse",
        "inflammation", "glycoside", "glycosides", "neurodegeneration"
    ]
}

def classify_query(query: str) -> str:
    q = query.lower()
    # 1) Keyword‐based routing
    for domain, terms in DOMAIN_KEYWORDS.items():
        if any(term in q for term in terms):
            return domain
    # 2) Fallback to zero‐shot LLM
    llm = ChatOpenAI(temperature=0)
    prompt = (
        "Classify this question into either 'oncology' or 'neurology'.\n\n"
        f"Question: {query}\n\nAnswer with only the domain name."
    )
    dom = llm.predict(prompt).strip().lower()
    return dom if dom in DOMAIN_KEYWORDS else "oncology"

def answer_query(query: str, dbs: Dict[str, object]) -> str:
    domain = classify_query(query)
    vectordb = dbs[domain]
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 5})
    )
    return qa.run(query)
