# Portfolio Insight - Demo (Streamlit, local)

Resumen rápido
- Demo rápida para que un inversor pregunte sobre el portfolio usando RAG (indexado local).
- Coloca tus notebooks en `notebooks/` y CSVs procesados en `data/processed/`.

Pasos para levantar local
1. Clonar repo y crear entorno:
   python -m venv .venv
   source .venv/bin/activate
2. Instalar dependencias:
   pip install -r requirements.txt
3. Indexar documentos:
   python scripts/index_data.py
4. Configurar LLM (opcional si querés respuestas generadas):
   export OPENAI_API_KEY="tu_openai_key"
   export OPENAI_MODEL="gpt-4o-mini"  # o el que tengas
   Si preferís usar Google Gemini, adaptá `app_streamlit/streamlit_app.py`
5. Ejecutar la app:
   streamlit run app_streamlit/streamlit_app.py
6. Abrir http://localhost:8501 y probar preguntando sobre el portfolio.

Despliegue rápido
- Deploy recomendado: Hugging Face Spaces (elige Runtime: Streamlit).  
- Subir repo al espacio y definir variables de entorno (OPENAI_API_KEY si usás OpenAI).

Notas
- No subas datos sensibles. Mantén datos brutos en Google Drive y solo sube resúmenes o samples.
- Si querés escala/producción: migrar embeddings a Pinecone y backend a FastAPI (lo hacemos después).
