FROM python:3.12-slim

#Dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \

WORKDIR /inference

COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference .

ENV PYTHONUNBUFFERED=1

CMD ["python", "inference.py"]