FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py detector.py analysis.py worker.py storage.py ./

# Weights are not baked into the image: they're git-ignored and change on a
# different schedule from the code. Mount them at /models (see docker-compose.yml).
ENV MODEL_PATH=/models/best.onnx

# Run as a non-root user (security best practice)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
