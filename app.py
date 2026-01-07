# app.py -- Entry point para Hugging Face Spaces
# Ejecuta el Streamlit app existente en app_streamlit/streamlit_app.py
import os
import sys

port = os.environ.get("PORT", "8080")
cmd = ["streamlit", "run", "app_streamlit/streamlit_app.py", "--server.port", port, "--server.address", "0.0.0.0"]

# Reemplaza el proceso actual por el comando Streamlit
os.execvp(cmd[0], cmd)
