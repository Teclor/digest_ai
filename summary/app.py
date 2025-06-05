from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from kafka import KafkaConsumer, KafkaAdminClient
import json

BOOTSTRAP_SERVERS = "kafka:9092"
MAX_MESSAGES_TO_READ = 1000
POLL_TIMEOUT_MS = 1000  # 1 секунда

app = FastAPI()

# CORS (на случай, если фронт будет отдельно)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/topics", response_model=List[str])
def get_topics():
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=BOOTSTRAP_SERVERS,
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
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        group_id=None,
        fetch_max_bytes=52428800,
        max_partition_fetch_bytes=10485760,
        max_poll_records=500,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    messages = []
    try:
        while len(messages) < MAX_MESSAGES_TO_READ:
            records = consumer.poll(timeout_ms=POLL_TIMEOUT_MS)
            if not records:
                break
            for tp, msgs in records.items():
                for msg in msgs:
                    messages.append(msg.value)
                    if len(messages) >= MAX_MESSAGES_TO_READ:
                        break
                if len(messages) >= MAX_MESSAGES_TO_READ:
                    break
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при чтении сообщений: {e}")
    finally:
        consumer.close()

    if not messages:
        return {"messages": []}

    return {
        "messages": messages[-limit:]
    }
