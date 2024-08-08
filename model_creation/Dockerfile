FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /model_creation
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["tr -d '\\r' < /env_data/.env > /env_data/.env.unix && source /env_data/.env.unix && python model_creation.py"]