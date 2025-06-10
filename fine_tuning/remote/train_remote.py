import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "4"  # Под ваше количество ядер
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
# os.environ["TF_NUM_INTEROP_THREADS"] = "4"
# Попробуйте установить 'PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync'
# или 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' если expandable_segments не помогает
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

torch.cuda.empty_cache()

# === Параметры ===
MODEL_NAME = "unsloth/gemma-3-1b-it-qat"
OUTPUT_DIR = "output_summarization_gemma3"  # Изменено для ясности
# MAX_SEQ_LENGTH должен учитывать и текст, и саммари, и шаблон промпта
MAX_SEQ_LENGTH = 768  # Увеличено, чтобы вместить промпт, текст и саммари. Подберите под ваши данные.
TOKENIZED_DIR = "resources/tokenized_data_summarization_gemma3"  # Изменено для нового формата

# === Загрузка датасета ===
try:
    train_df = pd.read_parquet("resources/summarization_ds_train.parquet")
    test_df = pd.read_parquet("resources/summarization_ds_test.parquet")

    # Для отладки можно использовать меньшую часть датасета
    # train_df = train_df.sample(n=1000, random_state=42)
    # test_df = test_df.sample(n=200, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
except Exception as e:
    print(f"Ошибка при загрузке датасета: {e}")
    raise

# === Токенизатор ===
# Убедитесь, что токенизатор загружается корректно и содержит необходимые спец. токены
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Gemma использует <bos>, <eos>, <start_of_turn>, <end_of_turn>
# Установка pad_token:
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Установлен pad_token = eos_token: {tokenizer.eos_token}")
    else:
        # Если нет eos_token, добавляем новый pad_token. Это менее предпочтительно для Gemma.
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Добавлен новый pad_token: [PAD]")
        # Если добавляется новый токен, может потребоваться model.resize_token_embeddings(len(tokenizer))
        # Но для QLoRA это может быть проблематично. Лучше убедиться, что eos_token есть.

# Шаблон промпта для instruction-tuned модели Gemma
# Важно использовать формат, на котором модель обучалась
PROMPT_TEMPLATE_START = "<start_of_turn>user\nПерескажи текст кратко, сохраняя его смысл. В ответе должен быть только сам пересказ.:\n"
PROMPT_TEMPLATE_END = "<end_of_turn>\n<start_of_turn>model\n"  # Модель должна генерировать после этого


def tokenize_function(examples):
    input_ids = []
    attention_masks = []
    labels = []

    for i in range(len(examples["text"])):
        try:
            # Формируем промпт (prompt для пользователя)
            chat_user_content = f"Перескажи следующий текст:\n{examples['text'][i]}"
            chat_model_content = examples['summary'][i]

            # Применяем шаблон чата для формирования входа модели
            chat_full = [
                {"role": "user", "content": chat_user_content},
                {"role": "assistant", "content": chat_model_content}
            ]

            tokenized_full = tokenizer.apply_chat_template(
                chat_full,
                add_generation_prompt=False,  # Мы хотим получить весь диалог
                padding="max_length",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt"
            ).to("cpu")

            # Получаем длину prompt части (до ответа модели)
            chat_prefix = [{"role": "user", "content": chat_user_content}]
            tokenized_prefix = tokenizer.apply_chat_template(
                chat_prefix,
                add_generation_prompt=True,  # Чтобы включить начало ответа модели
                padding=False,
                truncation=True,
                return_tensors="pt"
            ).to("cpu")

            prefix_length = tokenized_prefix.shape[-1]

            # Маскируем prompt часть меткой -100
            label_ids = tokenized_full.clone()
            label_ids[:, :prefix_length] = -100

            # Если есть паддинг — тоже маскируем
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is not None:
                pad_indices = (tokenized_full == pad_token_id).nonzero(as_tuple=True)
                if len(pad_indices) > 1:
                    label_ids[pad_indices[0], pad_indices[1]] = -100

            # Сохраняем результаты
            input_ids.append(tokenized_full.squeeze().tolist())
            attention_masks.append((tokenized_full != pad_token_id).long().squeeze().tolist())
            labels.append(label_ids.squeeze().tolist())

        except Exception as e:
            print(f"Ошибка при обработке примера {i}: {e}")
            # В случае ошибки добавляем пустые/недействительные данные
            input_ids.append([tokenizer.pad_token_id] * MAX_SEQ_LENGTH)
            attention_masks.append([0] * MAX_SEQ_LENGTH)
            labels.append([-100] * MAX_SEQ_LENGTH)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }


if os.path.exists(TOKENIZED_DIR) and os.path.exists(os.path.join(TOKENIZED_DIR, "train")) and os.path.exists(
        os.path.join(TOKENIZED_DIR, "test")):
    print("Загрузка токенизированного датасета из кэша...")
    try:
        tokenized_train = load_from_disk(os.path.join(TOKENIZED_DIR, "train"))
        tokenized_test = load_from_disk(os.path.join(TOKENIZED_DIR, "test"))
        print("Токенизированный датасет успешно загружен из кэша.")
    except Exception as e:
        print(f"Ошибка загрузки кэшированного датасета: {e}. Токенизация с нуля...")
        # Удаляем некорректную папку кэша
        import shutil

        if os.path.exists(TOKENIZED_DIR):
            shutil.rmtree(TOKENIZED_DIR)
        os.makedirs(TOKENIZED_DIR, exist_ok=True)

        print("Токенизация обучающего датасета...")
        tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
        print("Токенизация тестового датасета...")
        tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)

        print("Сохранение токенизированного датасета...")
        tokenized_train.save_to_disk(os.path.join(TOKENIZED_DIR, "train"))
        tokenized_test.save_to_disk(os.path.join(TOKENIZED_DIR, "test"))
        print("Токенизированный датасет сохранен.")
else:
    print("Токенизация с нуля...")
    os.makedirs(TOKENIZED_DIR, exist_ok=True)  # Создаем папку, если ее нет

    print("Токенизация обучающего датасета...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    print("Токенизация тестового датасета...")
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)

    print("Сохранение токенизированного датасета...")
    tokenized_train.save_to_disk(os.path.join(TOKENIZED_DIR, "train"))
    tokenized_test.save_to_disk(os.path.join(TOKENIZED_DIR, "test"))
    print("Токенизированный датасет сохранен.")

# === Загрузка модели с квантованием и PEFT ===
# Используйте bfloat16 для compute_dtype, если ваша GPU это поддерживает (Ampere+)
# для лучшей стабильности обучения. Иначе float16.
compute_dtype = torch.float16
print(f"Используемый compute_dtype для BitsAndBytes: {compute_dtype}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"  # nf4 обычно лучше для точности
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True,
    # Попробуйте "flash_attention_2" для ускорения и экономии памяти, если поддерживается.
    # Если нет, "sdpa" (Scaled Dot Product Attention) лучше, чем "eager" для PyTorch 2.0+.
    # Unsloth модели часто оптимизированы для Flash Attention.
    attn_implementation="flash_attention_2",  # или "flash_attention_2"
    device_map="auto",
    # offload_folder и offload_state_dict могут замедлить обучение.
    # Попробуйте убрать, если модель помещается в память после других оптимизаций.
    # offload_folder="offload",
    # offload_state_dict=True,
    torch_dtype=compute_dtype,  # Должен совпадать с bnb_4bit_compute_dtype или быть torch.float32 для загрузки
)

model.config.use_cache = False  # Обязательно при gradient checkpointing
model.gradient_checkpointing_enable()

# Вручную отправляем модель на GPU
model.to("cuda")  # <-- Новое

# `prepare_model_for_kbit_training` подготавливает модель для обучения с квантованием.
# `use_reentrant=False` для gradient_checkpointing_kwargs может быть эффективнее,
# но Unsloth иногда рекомендует True для своих реализаций. Оставляем False как в оригинале.
model = prepare_model_for_kbit_training(
    model,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# Конфигурация LoRA
# r и lora_alpha: alpha обычно 2*r. Можно экспериментировать.
# target_modules: Важно указать правильные модули для вашей модели.
# Для Gemma это обычно все линейные слои в блоках внимания и MLP.
# ["q_proj", "k_proj", "v_proj", "o_proj"] - для внимания
# ["gate_proj", "up_proj", "down_proj"] - для MLP слоев Gemma
# Unsloth может автоматически определять их или иметь свои рекомендации.
peft_config = LoraConfig(
    r=16,  # Ранг LoRA матриц. 8, 16, 32, 64. Больше = больше параметров, но не всегда лучше.
    lora_alpha=32,  # Масштабирующий фактор. Часто 2*r.
    lora_dropout=0.05,  # Dropout для LoRA слоев. 0.05 или 0.1
    bias="none",  # 'none', 'all', or 'lora_only'. 'none' обычно достаточно.
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"  # MLP layers
    ]
    # Для некоторых моделей Unsloth может предлагать использовать `modules_to_save`
    # для сохранения других частей модели, например, `embed_tokens` или `lm_head`, если вы их дообучаете.
    # Но для стандартного LoRA это не требуется.
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Посмотреть, какая часть модели обучается

# === Аргументы обучения ===
# Ключевые изменения: learning_rate, eval_steps, save_steps.
# gradient_accumulation_steps * per_device_train_batch_size = эффективный размер батча.
# 4 * 2 = 8. Это небольшой батч. Если возможно, увеличьте gradient_accumulation_steps.
# Но это увеличит время одного шага.

# Количество шагов на эпоху: 20000 строк / (2 batch_size * 4 grad_accum) = 20000 / 8 = 2500 шагов/эпоха.
# Общее количество шагов: 2500 * 3 эпохи = 7500 шагов.

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # 3 эпохи - хорошее начало
    per_device_train_batch_size=8,  # Ограничено памятью
    per_device_eval_batch_size=8,  # Ограничено памятью
    gradient_accumulation_steps=2,  # Увеличено для лучшей стабильности, эффективный батч 16. Подберите под вашу память.
    # Если памяти не хватает, верните 4.
    # Оптимизатор (AdamW - по умолчанию)

    optim="adamw_torch_fused" if torch.cuda.is_available() and hasattr(torch.optim,
                                                                       'AdamW') and 'fused' in torch.optim.AdamW.__init__.__kwdefaults__ else "adamw_torch",
    # Попробуйте fused AdamW для скорости

    # Скорость обучения (Learning Rate) - КРИТИЧЕСКИ ВАЖНО!
    # Для LoRA обычно используют значения порядка 1e-4, 2e-4, 3e-4.
    # 5e-7 было слишком мало.
    learning_rate=2e-4,  # Значительно увеличено. Это типичное значение для QLoRA.

    # Планировщик скорости обучения
    lr_scheduler_type="cosine",  # 'linear' (по умолчанию) или 'cosine' часто хорошо работают.
    warmup_ratio=0.05,
    # 5% шагов на разогрев. (0.05 * 7500 / (3 эпохи * (20000/(2*8)))) = 0.05 * (7500 * 8 / 20000) = 0.05 * 3 = 0.15
    # warmup_steps = int(total_train_steps * warmup_ratio)
    # warmup_steps = int((20000 / (2*8)) * 3 * 0.05) = int(2500 * 3 * 0.05) = int(7500 * 0.05) = 375 шагов
    warmup_steps=300,  # Явно зададим количество шагов для разогрева

    # fp16/bf16 обучение
    fp16=(compute_dtype == torch.float16),  # Включить, если compute_dtype float16
    bf16=(compute_dtype == torch.bfloat16),  # Включить, если compute_dtype bfloat16

    # Логирование, оценка и сохранение
    # Оценивайте реже, чтобы не замедлять обучение.
    # Например, каждые 0.1 - 0.25 эпохи. 2500 шагов/эпоха * 0.1 = 250 шагов.
    # `eval_steps` и `save_steps` должны быть согласованы.
    # `logging_steps` может быть чаще.
    logging_strategy="steps",
    logging_steps=10,  # Как часто логировать loss (каждые 25 * grad_accum шагов оптимизатора)
    eval_strategy="steps",  # или "epoch"
    eval_steps=50,  # Оценивать каждые 50 шагов оптимизатора
    save_strategy="steps",  # или "epoch"
    save_steps=50,  # Сохранять чекпоинт каждые 50 шагов оптимизатора
    save_total_limit=2,  # Хранить последние 3 чекпоинта + лучший
    load_best_model_at_end=True,  # Загрузить лучшую модель в конце обучения
    metric_for_best_model="eval_loss",  # Метрика для выбора лучшей модели
    greater_is_better=False,  # Меньший eval_loss лучше

    # Другие параметры
    report_to=["tensorboard"],  # Логирование в TensorBoard
    dataloader_num_workers=4,  # Количество воркеров для загрузки данных
    label_names=["labels"], # Уже по умолчанию для Trainer, если есть колонка "labels"
    remove_unused_columns=True,  # Удалять неиспользуемые колонки из датасета
    max_grad_norm=0.3,  # Клиппинг градиента для стабильности, типичное значение для QLoRA
    weight_decay=0.01,  # Небольшая регуляризация L2
    max_steps=-1,  # Если > 0, переопределяет num_train_epochs
)

# Data Collator
# label_pad_token_id=-100 говорит data collator заменять padding в labels на -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,  # Игнорировать pad_token в вычислении лосса
    pad_to_multiple_of=8  # Может дать небольшое ускорение на некоторых GPU
)

# === Тренер ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    tokenizer=tokenizer,  # Передаем токенизатор для сохранения и возможного использования в коллбэках
)

# === Запуск обучения ===
# Поиск последнего чекпоинта
checkpoint_path = None
if os.path.isdir(OUTPUT_DIR):
    checkpoint_files = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split("-")[-1]))
        checkpoint_path = checkpoint_files[-1]
        print(f"Возобновление обучения с чекпоинта: {checkpoint_path}")

print("Начало обучения...")
try:
    trainer.train(resume_from_checkpoint=checkpoint_path)
except Exception as e:
    print(f"Произошла ошибка во время обучения: {e}")
    # Полезно сохранить текущее состояние, если это возможно и нужно
    # trainer.save_model(os.path.join(OUTPUT_DIR, "error_checkpoint"))
    # tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "error_checkpoint"))
    # print(f"Модель сохранена в {os.path.join(OUTPUT_DIR, 'error_checkpoint')} из-за ошибки.")
    raise

# === Сохранение модели и адаптеров ===
print(f"Сохранение лучшей модели в {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)  # Сохраняет лучшую модель (адаптеры LoRA)
tokenizer.save_pretrained(OUTPUT_DIR)  # Сохраняет токенизатор

# Если вы хотите сохранить полную модель (слияние адаптеров с базовой моделью):
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained(os.path.join(OUTPUT_DIR, "final_merged_model"))
# tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_merged_model"))
# print(f"Полная объединенная модель сохранена в {os.path.join(OUTPUT_DIR, 'final_merged_model')}")
# Внимание: merged_model будет занимать значительно больше места и потребует больше VRAM для инференса.

print("Обучение завершено!")