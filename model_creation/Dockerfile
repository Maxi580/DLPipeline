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

RUN chmod +x load_env.sh
ENTRYPOINT ["./load_env.sh"]
CMD ["python", "model_creation.py"]