FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model (67 MB, MIT) into the image so containers start
# offline and fast. Unlike the vision weights, it isn't derived from CarDD,
# and it changes only when embeddings.py's MODEL does.
ENV EMBED_CACHE_DIR=/opt/fastembed HF_HUB_DISABLE_TELEMETRY=1
COPY embeddings.py .
RUN python -c "from embeddings import Embedder; Embedder()"

COPY main.py chunking.py ingest.py wordings.json ./

# Run as a non-root user (security best practice)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["python", "main.py"]
