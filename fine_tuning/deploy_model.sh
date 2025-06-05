#!/bin/bash
set -euo pipefail

if [ -e "./merged_model" ]; then
    sudo rm -rf "./merged_model"
fi
if [ -e "./gemma-1b-it.gguf" ]; then
    sudo rm -rf "./gemma-1b-it.gguf"
fi


echo "=== Запуск мерджа модели ==="
python merge_model.py

echo "=== Запуск конвертации в GGUF ==="
python convert_to_gguf.py

MERGED_GGUF="gemma-1b-it.gguf"
OLLAMA_BLOBS_DIR="../../ollama-docker/ollama/ollama/models/gemma"

echo "=== Копирование GGUF модели в Ollama models ==="
sudo mkdir -p "$OLLAMA_BLOBS_DIR"
if [ -e "$OLLAMA_BLOBS_DIR/$MERGED_GGUF" ]; then
    sudo rm "$OLLAMA_BLOBS_DIR/$MERGED_GGUF"
fi
sudo cp "$MERGED_GGUF" "$OLLAMA_BLOBS_DIR/"

# Название контейнера Ollama (замени при необходимости)
OLLAMA_CONTAINER_NAME="ollama"

echo "=== Удаление старой модели в Ollama (если есть) ==="
docker exec "$OLLAMA_CONTAINER_NAME" ollama rm gemma_tuned || true

echo "=== Создание новой модели в Ollama ==="
docker exec "$OLLAMA_CONTAINER_NAME" ollama create gemma_tuned -f /root/.ollama/models/Modelfile

echo "=== Готово ==="
