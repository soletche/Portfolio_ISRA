"""
Indexador simple:
- Lee notebooks (.ipynb) en notebooks/
- Lee CSVs en data/processed/
- Crea embeddings con sentence-transformers
- Guarda vectorstore FAISS y metadata embeddings.json
"""
import os
import json
import glob
import nbformat
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

VECTORSTORE_PATH = "vectorstore.faiss"
DOCS_PATH = "embeddings.json"
EMB_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800  # caracteres

def load_notebooks(path="notebooks"):
    texts = []
    for fn in glob.glob(os.path.join(path, "*.ipynb")):
        try:
            nb = nbformat.read(fn, as_version=4)
            cells = []
            for i, cell in enumerate(nb.cells):
                if cell.cell_type in ("markdown", "code"):
                    # prefijamos la referencia de celda para poder citarla después
                    cells.append(f"[{os.path.basename(fn)} | cell {i}]\n" + cell.source)
            texts.append({"source": fn, "text": "\n\n".join(cells)})
        except Exception as e:
            print("Error reading", fn, e)
    return texts

def load_csvs(path="data/processed"):
    texts = []
    for fn in glob.glob(os.path.join(path, "*.csv")):
        try:
            df = pd.read_csv(fn)
            preview = df.head(200).to_csv(index=False)
            texts.append({"source": fn, "text": preview})
        except Exception as e:
            print("Error reading csv", fn, e)
    return texts

def chunk_text(text, max_len=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), max_len):
        chunks.append(text[i:i+max_len])
    return chunks

def main():
    os.makedirs("data/processed", exist_ok=True)
    docs = []
    items = load_notebooks("notebooks") + load_csvs("data/processed")
    for item in items:
        for c in chunk_text(item["text"]):
            docs.append({"source": item["source"], "text": c})
    print(f"Document chunks: {len(docs)}")
    model = SentenceTransformer(EMB_MODEL)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, VECTORSTORE_PATH)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    print("Saved vectorstore and docs.")

if __name__ == "__main__":
    main()
