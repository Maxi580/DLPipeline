FROM python:3.8-slim

# Installing Dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \

WORKDIR /model_creation

COPY model_creation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_creation .

ENV PYTHONUNBUFFERED=1

COPY load_env.sh /usr/local/bin/load_env.sh
RUN chmod +x /usr/local/bin/load_env.sh
ENTRYPOINT ["/usr/local/bin/load_env.sh"]
CMD ["python", "model_creation.py"]