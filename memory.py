"""Менеджер памяти диалога: стратегии buffer и summary + entity memory."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class MemoryManager:
    """Хранит историю сообщений с поддержкой двух стратегий обрезки."""

    def __init__(self, strategy: str = "buffer", max_messages: int = 20, model=None):
        self.strategy = strategy
        self.max_messages = max_messages
        self.model = model
        self.messages: list = []
        self.summary: str = ""
        self.entities: dict = {}

    def add_user_message(self, text: str):
        self.messages.append(HumanMessage(content=text))
        self._trim()

    def add_ai_message(self, text: str):
        self.messages.append(AIMessage(content=text))
        self._trim()

    def get_history(self) -> list:
        """Возвращает историю для подстановки в MessagesPlaceholder."""
        result = []
        if self.strategy == "summary" and self.summary:
            result.append(SystemMessage(
                content=f"Краткое содержание предыдущего разговора:\n{self.summary}"
            ))
        result.extend(self.messages)
        return result

    def _trim(self):
        if self.strategy == "buffer":
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]
        elif self.strategy == "summary":
            if len(self.messages) > self.max_messages and self.model:
                self._summarize()

    def _summarize(self):
        """Суммаризирует старые сообщения, оставляя последние 10."""
        keep_last = 10
        old_messages = self.messages[:-keep_last]
        if not old_messages:
            return

        conversation = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in old_messages
        )

        prompt = (
            "Сделай краткое содержание этого диалога. "
            "Сохрани все ключевые факты (имена, предпочтения, важные детали):\n\n"
            f"{conversation}\n\nКраткое содержание:"
        )

        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            self.summary = response.content
            self.messages = self.messages[-keep_last:]
        except Exception:
            self.messages = self.messages[-self.max_messages:]

    def clear(self):
        """Очищает историю и саммари, но сохраняет сущности."""
        self.messages = []
        self.summary = ""

    def set_strategy(self, strategy: str):
        self.strategy = strategy

    def message_count(self) -> int:
        return len(self.messages)

    # --- Entity Memory (расширение) ---

    def extract_entities(self, text: str):
        """Извлекает именованные сущности из текста через LLM."""
        if not self.model:
            return
        try:
            response = self.model.invoke([HumanMessage(content=(
                "Извлеки из текста именованные сущности (имена людей, города, "
                "языки программирования, предпочтения, факты о пользователе). "
                "Верни ТОЛЬКО валидный JSON-словарь без markdown-обёртки, "
                "где ключ — категория, значение — значение. "
                "Если сущностей нет, верни пустой словарь {}.\n\n"
                f"Текст: {text}"
            ))])
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            entities = json.loads(content)
            if isinstance(entities, dict):
                self.entities.update(entities)
        except Exception:
            pass

    def get_entities_summary(self) -> str:
        """Возвращает строку с известными фактами о пользователе."""
        if not self.entities:
            return ""
        facts = ", ".join(f"{k}: {v}" for k, v in self.entities.items())
        return f"Известные факты о пользователе: {facts}"
