FROM python:3.10-slim

LABEL maintainer="you@example.com"
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/mlruns

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.enableCORS=false"]
