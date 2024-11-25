from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

# Initialize Flask app
app = Flask(__name__)

# Load environment variables (if needed for other configurations)
load_dotenv()

# Step 1: Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Step 2: Define ChromaDB persistence directory
persist_directory = "chroma_db"

# Step 3: Load the existing ChromaDB index
docsearch = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# Step 4: Define the prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Step 5: Initialize the LLM
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# Step 6: Setup RetrievalQA with ChromaDB
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Step 7: Flask Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response: ", result["result"])
    return str(result["result"])

# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
