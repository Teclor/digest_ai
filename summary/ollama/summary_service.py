import requests
from config import config
from ollama.prompt_template_factory import PromptTemplateFactory
from transformers import AutoTokenizer
from typing import Optional

class SummaryService:
    def __init__(
        self,
        ollama_host: str = config.ollama.host,
        model: str = config.ollama.default_model,
        tokenizer_name: Optional[str] = None
    ):
        self.ollama_host = ollama_host
        self.model = model
        self.tokenizer_name = tokenizer_name or config.ollama.default_tokenizer or model
        self._tokenizer = None  # Ленивая инициализация

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=True
            )
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def limit_tokens(self, text: str, max_tokens: int) -> str:
        """
        Обрезает текст до заданного числа токенов.
        :param text: Входной текст
        :param max_tokens: Максимальное количество токенов
        :return: Обрезанный текст
        """
        if max_tokens <= 0:
            return ""

        # Токенизируем текст
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) <= max_tokens:
            return text

        # Берём только первые max_tokens токенов
        limited_tokens = tokens[:max_tokens]

        # Декодируем обратно в строку
        limited_text = self.tokenizer.convert_tokens_to_string(limited_tokens)

        return limited_text

    def summarize(self, text: str, template_type: str = "message") -> str:
        """
        :param text: Текст для пересказа
        :param template_type: Тип текста (используется для выбора шаблона)
        :return: Пересказанный текст от модели
        """
        prompt = PromptTemplateFactory.get_prompt(template_type).format(text=text)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=config.ollama.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.RequestException as e:
            raise RuntimeError(f"Ошибка при обращении к Ollama: {e}")