FROM python:3.12-slim

WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    docker.io \
    curl \
    gnupg \
    lsb-release && \
    apt-get clean

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce-cli

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app .

COPY load_env.sh /usr/local/bin/load_env.sh
RUN chmod +x /usr/local/bin/load_env.sh
ENTRYPOINT ["/usr/local/bin/load_env.sh"]
CMD ["python", "app.py"]