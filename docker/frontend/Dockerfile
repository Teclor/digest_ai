FROM node:20-slim

# Установка нужных утилит
RUN apt update && \
    apt install -y --no-install-recommends \
        iproute2 \
        iputils-ping \
        netcat-openbsd \
        telnet \
        dnsutils \
        procps && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*


COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 5173

ENTRYPOINT ["/entrypoint.sh"]