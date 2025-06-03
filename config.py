# config.py
from datetime import timedelta

CONFIG = {
    "models": [
        # "qwen3:8b",
        # "qwen3:4b",
        "gemma3:4b-it-qat",
        # "gemma3:4b",
        "gemma3:1b",
        # "qwen3:1.7b"
    ],
    "max_runtime": timedelta(minutes=150),  # Время тестирования в минутах
    "dataset": {
        "name": "RussianNLP/Mixed-Summarization-Dataset",
        "split": "train"
    },
    "output": {
        "metrics_file": "metrics.json",
        "backup_metrics_file": "backup_metrics.json",
        "failed_summaries_file": "failed_summaries.json",
        "results_dir": "results"
    },
    "save_interval": 20,  # Сохранять каждые N текстов
    "ollama_api_url": "http://localhost:7869/api/generate"
}