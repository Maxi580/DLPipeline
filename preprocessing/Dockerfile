FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /preprocessing
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x load_env.sh
ENTRYPOINT ["./load_env.sh"]
CMD ["python", "preprocessing.py"]