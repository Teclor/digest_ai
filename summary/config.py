from pydantic import BaseModel

class KafkaConfig(BaseModel):
    bootstrap_servers: str = "kafka:9092"
    max_messages_to_read: int = 1000
    poll_timeout_ms: int = 1000  # ms
    max_poll_records: int = 500
    fetch_max_bytes: int = 52428800  # 50 MB
    max_partition_fetch_bytes: int = 10485760  # 10 MB


class OllamaConfig(BaseModel):
    host: str = "http://ollama:11434"
    default_model: str = "gemma3-1b-ftr-75k:latest"
    default_tokenizer: str = "unsloth/gemma-3-1b-it-qat"
    timeout: int = 20  # seconds
    max_input_text_tokens: int = 512


class AppConfig(BaseModel):
    kafka: KafkaConfig = KafkaConfig()
    ollama: OllamaConfig = OllamaConfig()

config = AppConfig()