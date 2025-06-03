import os
from datasets import load_dataset
import pandas as pd

def save_dataset_as_parquet():
    print("Загрузка датасета...")
    dataset = load_dataset("RussianNLP/Mixed-Summarization-Dataset")

    os.makedirs("resources", exist_ok=True)

    # Сохраняем train и test
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    train_df.to_parquet("resources/summarization_ds_train.parquet", index=False)
    test_df.to_parquet("resources/summarization_ds_test.parquet", index=False)

    print("Датасет успешно сохранён:")
    print("→ resources/summarization_ds_train.parquet")
    print("→ resources/summarization_ds_test.parquet")

if __name__ == "__main__":
    save_dataset_as_parquet()
