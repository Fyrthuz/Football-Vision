FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime AS base

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel --break-system-packages

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

FROM base AS production

COPY . .

RUN mkdir -p /app/data/uploads /app/data/results /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS training

COPY . .

RUN mkdir -p /app/data/uploads /app/data/results /app/models /app/datasets

ENTRYPOINT ["python", "-m"]
CMD ["training.train_detection"]
