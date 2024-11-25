from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
#!pip install chromadb
import os

# Load environment variables (if any)
load_dotenv()

# Step 1: Load PDF data and split into chunks
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Step 2: Initialize the embedding model
embeddings = download_hugging_face_embeddings()

# Step 3: Define the directory for ChromaDB persistence
persist_directory = "chroma_db"  # Directory to store ChromaDB

# Step 4: Create and store embeddings in ChromaDB
docsearch = Chroma.from_documents(
    documents=text_chunks,  # List of document objects (text_chunks should be structured as Documents)
    embedding=embeddings,
    persist_directory=persist_directory  # Path to save the ChromaDB index
)
docsearch.persist()  # Save the ChromaDB database
print(f"ChromaDB index created and stored in '{persist_directory}'.")
