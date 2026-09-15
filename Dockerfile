FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Tesseract (fra+eng) + Poppler + OCRmyPDF deps (ghostscript, unpaper, qpdf, pngquant)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-fra \
        tesseract-ocr-eng \
        poppler-utils \
        ghostscript \
        unpaper \
        pngquant \
        qpdf \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

# Render injecte $PORT — fallback 8000 en local
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
