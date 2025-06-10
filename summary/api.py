import os

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional

from kafka import KafkaAdminClient

from config import config
from messages.kafka_message_reader import KafkaMessageReader
from ollama.summary_service import SummaryService
import json
from utils.logger_configurator import LoggerConfigurator

# Импорт класса производителя Kafka
from messages.kafka_chat_producer import KafkaChatProducer
from config import OllamaConfig

# Инициализируем producer для использования в эндпоинтах
kafka_producer = KafkaChatProducer()

app = FastAPI()

# CORS (на случай, если фронт будет отдельно)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger_config = LoggerConfigurator("api.json", log_format="json")
logger = logger_config.get_logger("API")

@app.get("/api/topics", response_model=List[str])
def get_topics():
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=config.kafka.bootstrap_servers,
            client_id="web_topic_lister"
        )
        all_topics = admin.list_topics()
        admin.close()
        visible_topics = sorted([t for t in all_topics if t != "__consumer_offsets"])
        return visible_topics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении топиков: {e}")


@app.get("/api/messages")
def get_messages(topic: str = Query(...), limit: int = Query(10, ge=1, le=1000)):
    reader = KafkaMessageReader()
    try:
        chat_id, chat_name, messages = reader.get_chat_info_and_messages(topic, limit)
        if chat_name is None:
            return {"topic": topic, "chat_name": None, "messages": messages[-limit:]}
        return {"topic": chat_id, "chat_name": chat_name, "messages": messages[-limit:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summary")
def get_summary(text: str = Body(...), topic: str = Body(...)):
    summarizer = SummaryService()
    try:
        summary = summarizer.summarize(text, template_type=topic.split("_")[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при пересказе текста: {e}")

    chat_name = ''

    if os.path.exists("resources/chats/topic_map.json"):
        with open("resources/chats/topic_map.json", "r") as f:
            topic_map = json.load(f)
            if topic in topic_map:
                chat_name = topic_map[topic]

    return {
        "topic": topic,
        "chat_name": chat_name,
        "summary": summary
    }

@app.post("/api/messages/add")
def add_messages(
    topic: str = Query(..., description="Название топика Kafka"),
    messages: List[Dict] = Body(..., description="Список сообщений в формате JSON")
):
    try:
        logger.info(f"Получено {len(messages)} сообщений для отправки в топик '{topic}'")

        # Формируем записи в нужном формате
        records = [{"data": msg} for msg in messages]

        for record in records:
            future = kafka_producer.producer.send(topic, value=record)
            future.add_callback(kafka_producer._on_send_success(topic))
            future.add_errback(kafka_producer._on_send_error(topic))

        kafka_producer.producer.flush()
        return {"status": "success", "topic": topic, "sent_messages_count": len(messages)}

    except Exception as e:
        logger.error(f"Ошибка при отправке сообщений в топик '{topic}': {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при отправке сообщений: {e}")


@app.post("/api/messages/init")
def init_messages(resource_dir: Optional[str] = Query(None)):
    """
    Запускает отправку сообщений из папки по умолчанию (или указанной) в Kafka.
    """
    try:
        # Если resource_dir не указан, будет использован default_resource_dir
        kafka_producer.send_from_folder(resource_dir)

        return {"status": "success", "message": "Сообщения успешно отправлены из файлов"}
    except Exception as e:
        logger.error(f"Ошибка при инициализации сообщений из файлов: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при инициализации сообщений: {e}")


summarizer_service = SummaryService()

@app.post("/api/truncated")
async def truncate_text(text: str = Body(..., description="Текст для обрезки")):
    max_tokens = config.ollama.max_input_text_tokens
    try:
        truncated_text = summarizer_service.limit_tokens(text, max_tokens)
        return truncated_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обрезке текста: {str(e)}")