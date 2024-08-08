FROM python:3.8-slim

WORKDIR /preprocessing

COPY preprocessing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocessing .

ENV PYTHONUNBUFFERED=1

COPY load_env.sh /script/load_env.sh
RUN chmod +x /script/load_env.sh
ENTRYPOINT ["/script/load_env.sh"]
CMD ["python", "preprocessing.py"]