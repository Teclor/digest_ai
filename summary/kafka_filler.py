import os
import json
import glob
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Настройки Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
RESOURCE_DIR = "./resources"


def load_json_files(resource_dir):
    files = glob.glob(os.path.join(resource_dir, "messages_*.json"))
    return files


def send_to_kafka(producer, topic_name, messages):
    for message in messages:
        record = {
            "data": message
        }

        print(f"📤 Отправка в {topic_name}: {record}")

        future = producer.send(topic_name, value=record)

        # Добавляем callback для успешной отправки
        def on_send_success(record_metadata):
            print(f"✅ Сообщение успешно отправлено")
            print(f"  Топик: {record_metadata.topic}")
            print(f"  Партиция: {record_metadata.partition}")
            print(f"  Offset: {record_metadata.offset}")

        # Добавляем callback для ошибок
        def on_send_error(excp):
            print(f"❌ Ошибка при отправке сообщения: {excp}")
            if isinstance(excp, KafkaError):
                print(f"  Код ошибки: {excp.args}")
                print(f"  Подробности: {excp.message}")

        # Регистрируем колбэки
        future.add_callback(on_send_success)
        future.add_errback(on_send_error)

    print("⏳ Ожидаем окончания отправки...")
    producer.flush()


def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
        batch_size=16384,
        linger_ms=5,
        buffer_memory=33554432,
        max_in_flight_requests_per_connection=5,
        acks=1
    )

    files = load_json_files(RESOURCE_DIR)

    if not files:
        print(f"Файлы не найдены в {RESOURCE_DIR}")
        return

    for file_path in files:
        file_name = os.path.basename(file_path)

        if not file_name.startswith("messages_"):
            print(f"Пропущен некорректный файл: {file_name}")
            continue

        topic_name = file_name.replace("messages_", "").replace(".json", "")

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                messages = json.load(f)

                # Валидация: проверяем, что это список
                if not isinstance(messages, list):
                    print(f"⚠️ Ожидался список сообщений, получен тип: {type(messages)}")
                    continue

                send_to_kafka(producer, topic_name, messages)

            except json.JSONDecodeError as e:
                print(f"❌ Ошибка в JSON файле {file_path}: {e}")

    producer.flush()
    producer.close()
    print("✅ Все сообщения отправлены")


if __name__ == "__main__":
    main()