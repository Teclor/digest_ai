FROM python:3.11-slim

WORKDIR /app

# Установка нужных утилит
RUN apt update && \
    apt install -y --no-install-recommends \
        iproute2 \
        iputils-ping \
        netcat-openbsd \
        telnet \
        dnsutils \
        procps \
        curl \
        tcpdump && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Копируем зависимости и устанавливаем Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порты
EXPOSE 22
EXPOSE 5000

ENV PYTHONPATH=/app

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]