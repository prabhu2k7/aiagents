# File: rag_app/classifier.py
from typing import Dict
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType


def make_tools(dbs: Dict[str, object]):
    """
    Create two tools: one for cancer immunotherapy PDF, one for neuroinflammation PDF.
    """
    tools = []
    # Tool: Cancer immunotherapy
    def cancer_tool(question: str, vs=dbs["oncology"]):
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=vs.as_retriever(search_kwargs={"k": 5})
        )
        return qa.run(question)
    tools.append(
        Tool(
            name="cancer_immunotherapy",
            func=cancer_tool,
            description="Use this to answer questions from the Cancer immunotherapy PDF."
        )
    )
    # Tool: Neuroinflammation
    def neuro_tool(question: str, vs=dbs["neurology"]):
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=vs.as_retriever(search_kwargs={"k": 5})
        )
        return qa.run(question)
    tools.append(
        Tool(
            name="neuroinflammation",
            func=neuro_tool,
            description="Use this to answer questions from the Neuroinflammation PDF."
        )
    )
    return tools


def answer_query(query: str, dbs: Dict[str, object]) -> str:
    """
    Initialize a LangChain agent with two PDF tools. It selects the appropriate one at runtime.
    """
    tools = make_tools(dbs)
    agent = initialize_agent(
        tools,
        ChatOpenAI(temperature=0),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent.run(query)