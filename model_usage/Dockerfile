FROM python:3.8-slim

WORKDIR /model_usage

COPY model_usage/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_usage .

ENV PYTHONUNBUFFERED=1

COPY load_env.sh /usr/local/bin/load_env.sh
RUN chmod +x /usr/local/bin/load_env.sh
ENTRYPOINT ["/usr/local/bin/load_env.sh"]
CMD ["python", "model_usage.py"]