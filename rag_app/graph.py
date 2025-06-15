import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# Load vectorstores paths
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
oncology_vs = FAISS.load_local(os.path.join(root, 'oncology_index'), OpenAIEmbeddings())
neurology_vs = FAISS.load_local(os.path.join(root, 'neurology_index'), OpenAIEmbeddings())

# Define state schemas
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class State(InputState, OutputState):
    pass

# Node: Selector chooses domain based on similarity
class SelectState(TypedDict):
    question: str
    domain: str

def selector(state: InputState) -> SelectState:
    q = state['question']
    score_onc = oncology_vs.similarity_search_with_score(q, k=1)[0][1]
    score_neu = neurology_vs.similarity_search_with_score(q, k=1)[0][1]
    domain = 'oncology' if score_onc >= score_neu else 'neurology'
    print(f"[Selector] oncology={score_onc}, neurology={score_neu}, chosen={domain}")
    return {'question': q, 'domain': domain}

# Node: Oncology retrieval
def oncology_node(state: SelectState) -> Dict[str, str]:
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0), chain_type='stuff',
        retriever=oncology_vs.as_retriever(search_kwargs={'k':5})
    )
    ans = qa.run(state['question'])
    print(f"[OncologyNode] answer={ans}")
    return {'answer': ans}

# Node: Neurology retrieval
def neurology_node(state: SelectState) -> Dict[str, str]:
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0), chain_type='stuff',
        retriever=neurology_vs.as_retriever(search_kwargs={'k':5})
    )
    ans = qa.run(state['question'])
    print(f"[NeurologyNode] answer={ans}")
    return {'answer': ans}

# Assemble state graph
builder = StateGraph(State, input=InputState, output=OutputState)
builder.add_node(selector, name='select')
builder.add_node(oncology_node, name='oncology')
builder.add_node(neurology_node, name='neurology')
# Define edges
builder.add_edge(START, 'select')
builder.add_edge('select', 'oncology', condition=lambda s: s['domain']=='oncology')
builder.add_edge('select', 'neurology', condition=lambda s: s['domain']=='neurology')
builder.add_edge('oncology', END)
builder.add_edge('neurology', END)
# Compile
g = builder.compile()