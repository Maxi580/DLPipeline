FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /preprocessing
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["tr -d '\\r' < /env_data/.env > /env_data/.env.unix && source /env_data/.env.unix && python preprocessing.py"]