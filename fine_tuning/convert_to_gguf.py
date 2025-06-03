import os
import argparse
import subprocess

def convert_model(model_dir, output_name):
    print("🚀 Начинаем конвертацию модели в GGUF...")

    convert_script = os.path.join("llama.cpp", "convert.py")
    if not os.path.isfile(convert_script):
        raise FileNotFoundError("Не найден convert.py из llama.cpp. Убедись, что ты клонировал репозиторий.")

    cmd = [
        "python3", convert_script,
        "--outfile", output_name,
        "--tokenizer_path", model_dir,
        "--model_path", model_dir,
        "--model_type", "gemma"
    ]

    print(f"🛠 Команда запуска: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print("✅ Конвертация завершена!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Конвертировать модель HuggingFace в GGUF формат для llama.cpp")
    parser.add_argument("--model_dir", type=str, default="merged_model", help="Путь до модели (merged PEFT)")
    parser.add_argument("--output_name", type=str, default="gemma-1b-it.gguf", help="Имя выходного GGUF-файла")

    args = parser.parse_args()
    convert_model(args.model_dir, args.output_name)
