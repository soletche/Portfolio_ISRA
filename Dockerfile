FROM python:3.9-slim

# create non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# copy and install requirements
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copy the rest of the repo
COPY --chown=user . /app

# run Streamlit on the Docker Space port (7860)
CMD ["streamlit", "run", "app_streamlit/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
