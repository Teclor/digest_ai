import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"  # Под твоё количество ядер
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from datasets import load_from_disk

# === Параметры ===
MODEL_NAME = "unsloth/gemma-3-1b-it-qat"
OUTPUT_DIR = "output"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128

# === Загрузка датасета ===
train_df = pd.read_parquet("resources/summarization_ds_train.parquet")
test_df = pd.read_parquet("resources/summarization_ds_test.parquet")

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# === Токенизатор ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # для корректной генерации

def tokenize_function(examples):
    # Токенизируем исходный текст
    model_inputs = tokenizer(
        examples["text"],
        max_length=MAX_SOURCE_LENGTH,
        padding="max_length",
        truncation=True,
    )
    # Токенизируем summary
    labels = tokenizer(
        examples["summary"],
        max_length=MAX_SOURCE_LENGTH,  # сделать одинаковую длину
        padding="max_length",
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

TOKENIZED_DIR = "resources/tokenized_data"

if os.path.exists(TOKENIZED_DIR):
    print("Загрузка токенизированного датасета из кэша...")
    tokenized_train = load_from_disk(os.path.join(TOKENIZED_DIR, "train"))
    tokenized_test = load_from_disk(os.path.join(TOKENIZED_DIR, "test"))
else:
    print("Токенизация с нуля...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    print("Сохранение токенизированного датасета...")
    tokenized_train.save_to_disk(os.path.join(TOKENIZED_DIR, "train"))
    tokenized_test.save_to_disk(os.path.join(TOKENIZED_DIR, "test"))


# === Загрузка модели с квантованием и PEFT ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",  # рекомендация от Gemma
)

model.config.use_cache = False  # обязательно при checkpointing
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(
    model,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)

# === Аргументы обучения ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
    logging_dir="logs",
    logging_strategy="steps",
    logging_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=["tensorboard"],
    dataloader_num_workers=2,
    dataloader_prefetch_factor=2,
    label_names=["labels"],
)

# === Тренер ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)

# === Запуск обучения ===
checkpoint_path = None
if os.path.isdir(OUTPUT_DIR):
    checkpoint_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("checkpoint")]
    if checkpoint_files:
        checkpoint_path = os.path.join(OUTPUT_DIR, sorted(checkpoint_files)[-1])  # последний чекпоинт

trainer.train(resume_from_checkpoint=checkpoint_path)

# === Сохранение модели и адаптеров ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
