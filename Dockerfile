FROM python:3.9-slim

RUN useradd -m -u 1000 user USER user ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt RUN pip install --no-cache-dir --upgrade -r requirements.txt
Copiar el  del repo

COPY --chown=user . /app

CMD ["streamlit", "run", "app_streamlit/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
