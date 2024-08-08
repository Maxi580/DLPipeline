FROM python:3.12-slim

# Installing Dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /augmentation

COPY augmentation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY augmentation .
COPY augmentation/fractals /data/fractals

ENV PYTHONUNBUFFERED=1

COPY load_env.sh /env_data/load_env.sh
RUN chmod +x /env_data/load_env.sh
ENTRYPOINT ["/env_data/load_env.sh"]
CMD ["python", "augmentation.py"]