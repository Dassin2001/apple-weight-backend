FROM python:3.11-slim

WORKDIR /app

ENV YOLO_MODEL_PATH=assets/resources/yolo9-best.pt
ENV KNN_MODEL_PATH=assets/resources/knn_model.sav
ENV DATABASE_URL=sqlite:///./apple_detection.db
ENV VIDEO_UPLOAD_DIR=assets/uploaded/videos
ENV PICTURE_UPLOAD_DIR=assets/uploaded/pictures
ENV VIDEO_OUTPUT_DIR=assets/output/videos
ENV PICTURE_OUTPUT_DIR=assets/output/pictures

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "app.main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860", "--timeout", "300"]w