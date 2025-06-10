FROM python:3.10-slim

WORKDIR /app

RUN echo "deb http://mirror.yandex.ru/debian bookworm main" > /etc/apt/sources.list && \
    echo "deb http://mirror.yandex.ru/debian bookworm-updates main" >> /etc/apt/sources.list && \
    echo "deb http://mirror.yandex.ru/debian-security bookworm-security main" >> /etc/apt/sources.list && \
    apt-get update -o Acquire::Retries=1 && \
    apt-get install -y --no-install-recommends \
        iproute2 \
        iputils-ping \
        netcat-openbsd \
        telnet \
        dnsutils \
        procps \
        curl \
        tcpdump && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY wheel/*.whl /wheels/
RUN pip install --no-index --find-links=/wheels/ /wheels/*.whl

EXPOSE 22
EXPOSE 5000

ENV PYTHONPATH=/app

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]