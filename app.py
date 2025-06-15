import os
from flask import Flask, render_template, request
from dotenv import load_dotenv, find_dotenv
from rag_app.ingest import ingest_pdf_to_vectordb
from rag_app.graph import g

# Load env and ingest PDFs only if indexes missing
load_dotenv(find_dotenv())
assert os.getenv('OPENAI_API_KEY'), 'Set OPENAI_API_KEY in .env'
project_root = os.path.dirname(os.path.abspath(__file__))
# Ingest on first run
for pdf, idx in [( 'oncology.pdf','oncology_index'),('neurology.pdf','neurology_index')]:
    pdf_path = os.path.join(project_root, pdf)
    idx_path = os.path.join(project_root, idx)
    if not os.path.exists(idx_path):
        ingest_pdf_to_vectordb(pdf_path, idx_path)

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/', methods=['GET','POST'])
def index():
    messages = []
    if request.method=='POST':
        q = request.form['question'].strip()
        messages.append({'role':'user','text':q})
        ans = g.run(q)
        messages.append({'role':'assistant','text':ans})
    return render_template('index.html', messages=messages)

if __name__=='__main__':
    app.run(debug=True)