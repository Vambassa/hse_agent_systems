"""Обработчики для каждого типа запроса и механизм роутинга."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from characters import CHARACTER_PROMPTS, HANDLER_PROMPTS


def create_handler(model, request_type: str, character: str = "friendly"):
    """Создаёт LCEL-цепочку (prompt → model → StrOutputParser) для конкретного типа запроса."""
    character_prompt = CHARACTER_PROMPTS.get(character, CHARACTER_PROMPTS["friendly"])
    handler_prompt = HANDLER_PROMPTS.get(request_type, HANDLER_PROMPTS["unknown"])

    system = f"{character_prompt}\n\n{handler_prompt}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
    ])

    return prompt | model | StrOutputParser()


def create_all_handlers(model, character: str = "friendly") -> dict:
    """Создаёт словарь обработчиков для всех типов запросов."""
    return {
        request_type: create_handler(model, request_type, character)
        for request_type in HANDLER_PROMPTS
    }


def route(handlers: dict, request_type: str, query: str, history: list) -> str:
    """Направляет запрос в нужный обработчик по типу."""
    handler = handlers.get(request_type, handlers["unknown"])
    return handler.invoke({"query": query, "history": history})


def route_stream(handlers: dict, request_type: str, query: str, history: list):
    """Потоковая версия роутинга — yield'ит чанки ответа."""
    handler = handlers.get(request_type, handlers["unknown"])
    return handler.stream({"query": query, "history": history})
