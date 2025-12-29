# Streamlit app minimal para demo RAG + visual (fallback si faiss no está instalado)
import streamlit as st
import base64
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# intentamos importar faiss; si no está disponible, usamos fallback en memoria
try:
    import faiss
    FAISS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    faiss = None
    FAISS_AVAILABLE = False

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
    embeddings_np = None

    # Si existe vectorstore + faiss instalado, leemos el index
    if FAISS_AVAILABLE and os.path.exists(VECTORSTORE_PATH) and os.path.exists(DOCS_PATH):
        try:
            index = faiss.read_index(VECTORSTORE_PATH)
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                docs = json.load(f)
        except Exception as e:
            st.warning(f"Error leyendo vectorstore/faiss: {e}")
            index = None
            docs = []

    # Si no hay faiss pero sí tenemos embeddings.json, cargamos docs y precomputamos embeddings en numpy (fallback)
    elif os.path.exists(DOCS_PATH):
        try:
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                docs = json.load(f)
            texts = [d.get("text", "") for d in docs]
            if texts:
                embeddings_np = np.array(model.encode(texts, show_progress_bar=False))
        except Exception as e:
            st.warning(f"Error cargando embeddings.json en modo fallback: {e}")
            docs = []
            embeddings_np = None

    return model, index, docs, embeddings_np

model, index, docs, embeddings_np = load_vectorstore()

if index is None and embeddings_np is None:
    st.warning("Vectorstore no encontrado y faiss no está instalado. Puedes: 1) correr scripts/index_data.py y subir vectorstore.faiss + embeddings.json, 2) habilitar faiss en requirements y rebuild, o 3) seguir con demo sin recuperación avanzada.")
    # No abortamos completamente; la UI sigue disponible para que pruebes otras partes.
    # st.stop()  # opcional, pero lo dejamos comentado para que la página cargue.

def retrieve(query: str, top_k: int = 5):
    # Si faiss está disponible y hay index, usamos faiss
    if FAISS_AVAILABLE and index is not None:
        q_emb = model.encode([query])
        D, I = index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < len(docs):
                results.append(docs[idx])
        return results

    # Fallback: búsqueda por similitud en memoria usando numpy embeddings
    if embeddings_np is not None and len(embeddings_np) > 0:
        q_emb = model.encode([query], show_progress_bar=False)
        # similitud coseno
        q = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        emb_norm = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        sims = np.dot(emb_norm, q[0])
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            if idx < len(docs):
                results.append(docs[int(idx)])
        return results

    # Si no hay nada, devolvemos vacío
    return []

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
