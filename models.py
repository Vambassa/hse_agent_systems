"""Pydantic-модели для структурированного ввода/вывода ассистента."""

from enum import Enum

from pydantic import BaseModel, Field


class RequestType(str, Enum):
    """Типы пользовательских запросов."""
    QUESTION = "question"
    TASK = "task"
    SMALL_TALK = "small_talk"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


class Classification(BaseModel):
    """Результат классификации пользовательского запроса."""
    request_type: RequestType = Field(description="Тип запроса пользователя")
    confidence: float = Field(ge=0, le=1, description="Уверенность классификации от 0 до 1")
    reasoning: str = Field(description="Краткое обоснование выбранного типа")


class AssistantResponse(BaseModel):
    """Итоговый ответ ассистента с метаданными."""
    content: str = Field(description="Текст ответа")
    request_type: RequestType = Field(description="Определённый тип запроса")
    confidence: float = Field(ge=0, le=1, description="Уверенность классификации")
    tokens_used: int = Field(default=0, description="Приблизительное число использованных токенов")
