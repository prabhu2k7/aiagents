# File: app.py
from flask import Flask, render_template, request
from dotenv import load_dotenv, find_dotenv
import os
from rag_app.ingest import ingest_pdf_to_vectordb
from rag_app.classifier import answer_query

# Load env & prepare DBs once
load_dotenv(find_dotenv())
API_KEY = os.getenv("OPENAI_API_KEY")
assert API_KEY, "Set OPENAI_API_KEY in .env"

project_root = os.path.dirname(os.path.abspath(__file__))
oncology_pdf = os.path.join(project_root, "oncology.pdf")
neurology_pdf = os.path.join(project_root, "neurology.pdf")
oncology_db = ingest_pdf_to_vectordb(oncology_pdf, os.path.join(project_root, "oncology_index"))
neurology_db = ingest_pdf_to_vectordb(neurology_pdf, os.path.join(project_root, "neurology_index"))
dbs = {"oncology": oncology_db, "neurology": neurology_db}

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    messages = []
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if question:
            messages.append({'role': 'user', 'text': question})
            answer = answer_query(question, dbs)
            messages.append({'role': 'assistant', 'text': answer})
    return render_template('index.html', messages=messages)

if __name__ == '__main__':
    app.run(debug=True)