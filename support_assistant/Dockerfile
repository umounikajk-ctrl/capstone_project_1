# 1. Use Python 3.11
FROM python:3.11-slim

# 2. Create /app as the working directory inside the container
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --default-timeout=300 -r requirements.txt

COPY main.py /app/main.py
COPY graph.py /app/graph.py
COPY ingest.py /app/ingest.py
COPY prompts.py /app/prompts.py
COPY app.py /app/app.py

COPY docs /app/docs

RUN mkdir -p /app/data/chroma

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]