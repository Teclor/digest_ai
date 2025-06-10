FROM node:20-slim

# Установка нужных утилит
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


COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 5173

ENTRYPOINT ["/entrypoint.sh"]