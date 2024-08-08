FROM python:3.12-slim

# Installing Dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /model_creation

COPY model_creation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_creation .

ENV PYTHONUNBUFFERED=1

COPY model_creation/load_env.sh /model_creation
ENTRYPOINT ["./load_env.sh"]
CMD ["python", "model_creation.py"]