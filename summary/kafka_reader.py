import json
from kafka import KafkaConsumer, KafkaAdminClient

# Настройки
BOOTSTRAP_SERVERS = "kafka:9092"


def choose_topic(all_topics):
    print("\nДоступные топики (чаты):")
    for i, topic in enumerate([topic for topic in all_topics if topic != '__consumer_offsets'], 1):
        print(f"{i}. {topic}")
    choice = int(input("Выберите номер топика: "))
    return all_topics[choice]


def choose_limit():
    return int(input("Сколько последних сообщений вывести? "))


def main():
    MAX_MESSAGES_TO_READ = 1000
    POLL_TIMEOUT_MS = 1000  # Ждать максимум 1 секунду на каждом poll

    # Создаём AdminClient для получения списка топиков
    admin_client = KafkaAdminClient(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        client_id='topic_list_reader'
    )

    # Получаем список топиков
    all_topics = admin_client.list_topics()
    admin_client.close()

    if not all_topics:
        print("❌ Нет доступных топиков в Kafka.")
        return

    selected_topic = choose_topic(sorted(all_topics))
    limit = choose_limit()

    print(f"\nЧтение последних сообщений из топика '{selected_topic}'...\n")

    # Создаём consumer под конкретный топик
    consumer = KafkaConsumer(
        selected_topic,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        group_id=None,

        fetch_max_bytes=52428800,
        max_partition_fetch_bytes=10485760,
        max_poll_records=500,

        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    all_messages = []

    try:
        while len(all_messages) < MAX_MESSAGES_TO_READ:
            records = consumer.poll(timeout_ms=POLL_TIMEOUT_MS)

            if not records:
                print("📦 Больше нет новых сообщений.")
                break

            for topic_partition, messages in records.items():
                for message in messages:
                    all_messages.append(message.value)

                    if len(all_messages) >= MAX_MESSAGES_TO_READ:
                        print(f"⚠️ Достигнут лимит в {MAX_MESSAGES_TO_READ} сообщений")
                        break
                if len(all_messages) >= MAX_MESSAGES_TO_READ:
                    break

    finally:
        consumer.close()

    if not all_messages:
        print("❌ В выбранном топике нет сообщений.")
        return

    print(f"\nПоследние {min(limit, len(all_messages))} сообщений:\n")
    for msg in all_messages[-limit:]:
        print(json.dumps(msg, indent=2, ensure_ascii=False))
        print("-" * 60)


if __name__ == "__main__":
    main()