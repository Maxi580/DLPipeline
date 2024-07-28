FROM python:3.12-slim

WORKDIR /augmentation

COPY augmentation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY augmentation .
COPY augmentation/fractals /data/fractals

ENV PYTHONUNBUFFERED=1

COPY preprocessing/load_env.sh /usr/local/bin/load_env.sh
RUN chmod +x /usr/local/bin/load_env.sh
ENTRYPOINT ["/usr/local/bin/load_env.sh"]
CMD ["python", "augmentation.py"]