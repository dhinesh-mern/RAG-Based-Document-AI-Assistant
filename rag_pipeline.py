from PyPDF2 import PdfReader
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="gpt2")

embedding_model = load_embedding_model()
generator = load_llm()

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def get_embeddings(chunks):
    return embedding_model.encode(chunks)

#Store in FAISS (Vector DB)
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def retrieve(query, index, chunks, k=2):
    query_embedding = embedding_model.encode([query])
    print("Query:", query)
    print("Embedding sum:", query_embedding.sum())
    D, I = index.search(np.array(query_embedding), k)


    return [chunks[i] for i in I[0]]

def generate_answer(question, context_chunks):
    context = " ".join(context_chunks)

    prompt = f"""
    You are a helpful assistant.

    Answer the question strictly using ONLY the provided context.
    If the answer is not found in the context, say "Not found in document."

    Context:
    {context}

    Question:
    {question}

    Answer in 3-4 clear sentences:
    """
    result = generator(prompt, max_length=150)
    return result[0]["generated_text"]