FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YuNet face detector (232 KB, MIT, OpenCV Zoo), pinned to a commit and
# checked against its SHA-256, so a changed file fails the build instead of
# silently changing what gets redacted.
ADD --checksum=sha256:8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 \
    https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    /app/models/face_detection_yunet_2023mar.onnx

COPY main.py text_scan.py photos.py forensics.py ./

# The injection classifier (738 MB) and the AI-image detector are mounted at
# /models, like vision's weights (scripts/download_models.py fetches them).
ENV CLASSIFIER_DIR=/models/prompt-injection/onnx \
    AI_DETECTOR_DIR=/models/ai-detector \
    FACE_MODEL_PATH=/app/models/face_detection_yunet_2023mar.onnx

# Run as a non-root user (security best practice)
RUN useradd --create-home appuser && chmod a+r /app/models/*
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
