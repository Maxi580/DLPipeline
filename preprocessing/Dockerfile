FROM python:3.8-slim

WORKDIR /preprocessing

COPY preprocessing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocessing .

ENV PYTHONUNBUFFERED=1

RUN chmod +x ./load_env.sh
ENTRYPOINT ["./load_env.sh"]
CMD ["python", "preprocessing.py"]