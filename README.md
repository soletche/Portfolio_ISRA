# Portfolio_ISRA
Repositorio para el análisis mensual del portfolio (histórico desde 2025).

Notebook fuente en Colab:
https://colab.research.google.com/drive/1kMORaZG46bA0uyqvCCK0MQpytkDBWO4k

## Streamlit App

Este repositorio incluye una aplicación Streamlit para consultar el portfolio mediante RAG (Retrieval-Augmented Generation).

### Características

- **Fallback robusto**: La app funciona con o sin `faiss` instalado
  - Si `faiss` está disponible y existe `vectorstore.faiss` + `embeddings.json`, usa búsqueda vectorial optimizada
  - Si `faiss` no está disponible pero existe `embeddings.json`, usa búsqueda por similitud coseno en memoria con numpy
  - Si ninguno está disponible, la app se inicia con una advertencia clara pero no falla
- **Compatible con Hugging Face Spaces**: Puede desplegarse en HF Docker Spaces sin necesidad de compilar `faiss`
- **Integración con LLM**: Usa OpenAI API (configurable via `OPENAI_API_KEY` y `OPENAI_MODEL`)

### Instalación local

```bash
pip install -r requirements.txt
```

Si quieres usar la búsqueda vectorial optimizada con faiss, descomenta `#faiss-cpu` en `requirements.txt` y reinstala.

### Ejecutar la app

```bash
streamlit run app_streamlit/streamlit_app.py
```

O usando el entry point para Hugging Face Spaces:

```bash
python app.py
```

### Generar el índice vectorial (opcional)

Si tienes `faiss` instalado y quieres generar el índice:

```bash
python scripts/index_data.py
```

Esto genera `vectorstore.faiss` y `embeddings.json` para búsqueda optimizada.

Estructura recomendada
- notebooks/        Notebooks por mes (ej. `analysis_2025-01.ipynb`)
- src/              Código Python reusable
- data/             Datos (mejor mantener raw fuera del repo si son grandes)
- reports/          Resultados exportados (gráficos, tablas, PDF)

Cómo empezar
1. Clonar el repo:
   git clone https://github.com/soletche/portfolio-2025.git
2. Si trabajás en Colab y los datos están en Drive:
   from google.colab import drive
   drive.mount('/content/drive')
3. Abrir el notebook en `notebooks/` o usar `File → Save a copy in GitHub` desde Colab.

Buenas prácticas
- Nombres con fecha para versión histórica.
- Extraer lógica a `src/` para facilitar reutilización y tests.
- No subir credenciales ni datos sensibles.
