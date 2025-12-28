# Streamlit app minimal para demo RAG + visual
import streamlit as st
import base64
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import openai
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# Config
VECTORSTORE_PATH = "vectorstore.faiss"
DOCS_PATH = "embeddings.json"
EMB_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # o configurá GEMINI_KEY y adaptar

st.set_page_config(page_title="Portfolio Insight", layout="wide")
st.title("Portfolio Insight — Demo")
st.write("Pregunta sobre el portfolio y obtén respuesta con evidencia y gráfico.")

def load_vectorstore():
    model = SentenceTransformer(EMB_MODEL)
    dim = model.get_sentence_embedding_dimension()
    index = None
    docs = []
    if os.path.exists(VECTORSTORE_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(VECTORSTORE_PATH)
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
    return model, index, docs

model, index, docs = load_vectorstore()
if index is None:
    st.warning("Vectorstore no encontrado. Corre scripts/index_data.py primero.")
    st.stop()

def retrieve(query: str, top_k: int = 5):
    q_emb = model.encode([query])
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < len(docs):
            results.append(docs[idx])
    return results

def call_llm(prompt: str) -> str:
    key = OPENAI_KEY
    if not key:
        return "LLM no configurado. Setea OPENAI_API_KEY o adapta para Gemini."
    openai.api_key = key
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":"Eres Portfolio Insight, responde con TL;DR, evidencia y una línea accionable."},
                  {"role":"user","content":prompt}],
        max_tokens=400,
        temperature=0.0
    )
    return resp["choices"][0]["message"]["content"].strip()

st.sidebar.header("Opciones")
top_k = st.sidebar.slider("Top K documentos", 1, 10, 5)

query = st.text_input("Escribí tu pregunta aquí:")
if st.button("Preguntar"):
    if not query.strip():
        st.info("Escribe una pregunta.")
    else:
        with st.spinner("Buscando..."):
            docs_found = retrieve(query, top_k=top_k)
            context = ""
            sources = []
            for d in docs_found:
                context += f"\n---\nSOURCE: {d.get('source')}\n{d.get('text')}\n"
                sources.append(d.get('source'))
            prompt = f"Usa sólo la información del contexto para responder. Context:{context}\n\nPregunta: {query}"
            answer = call_llm(prompt)
        st.markdown("### Respuesta")
        st.write(answer)
        st.markdown("**Fuentes:**")
        for s in sources:
            st.write("-", s)

        # Si la pregunta implica series temporales, mostramos gráfico demo (si existe CSV)
        if "mensual" in query.lower() or "rendimiento" in query.lower() or "evolución" in query.lower():
            try:
                df = pd.read_csv("data/processed/monthly_returns.csv", parse_dates=["date"])
            except Exception:
                df = pd.DataFrame({
                    "date": pd.date_range("2025-01-01", periods=12, freq="M"),
                    "return": [0.02,0.01,-0.01,0.03,0.04,-0.02,0.05,0.01,0.02,-0.01,0.03,0.02]
                })
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(df["date"].dt.strftime("%Y-%m"), df["return"], marker='o')
            ax.set_title("Rendimiento mensual 2025")
            ax.set_xlabel("")
            ax.set_ylabel("Return")
            plt.xticks(rotation=45)
            st.pyplot(fig)
