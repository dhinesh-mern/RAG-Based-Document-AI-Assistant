import streamlit as st
from rag_pipeline import *

st.title("AI Document Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = extract_text(uploaded_file)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    index = create_faiss_index(embeddings)

    question = st.text_input("Ask a question")

    if question:
        retrieved_chunks = retrieve(question, index, chunks)
        answer = generate_answer(question, retrieved_chunks)
        st.write(answer)