"""Классификатор пользовательских запросов на базе LCEL-цепочки."""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from models import Classification, RequestType

CLASSIFICATION_SYSTEM_PROMPT = """\
Ты — классификатор запросов. Твоя единственная задача — определить тип запроса пользователя.

Типы запросов:
- question: вопрос, требующий информации («Что такое Python?», «Как работает GIL?», «Объясни рекурсию»)
- task: просьба что-то сделать («Напиши стих», «Расскажи анекдот», «Переведи текст»)
- small_talk: приветствие, болтовня, знакомство («Привет!», «Как дела?», «Меня зовут Алексей»)
- complaint: жалоба, недовольство («Это ужасно работает!», «Почему так долго?», «Я разочарован»)
- unknown: бессмыслица или нераспознанный запрос («asdfghjkl», «!!!???»)

Примеры:
- «Привет! Как тебя зовут?» → small_talk, confidence: 0.95
- «Что такое LCEL в LangChain?» → question, confidence: 0.93
- «Напиши хайку про программирование» → task, confidence: 0.90
- «Всё сломалось, ничего не работает!» → complaint, confidence: 0.88
- «йцукен фывапр» → unknown, confidence: 0.85
- «Как меня зовут?» → question, confidence: 0.90
- «Расскажи анекдот» → task, confidence: 0.92
- «Меня зовут Даша» → small_talk, confidence: 0.94
- «Мне не нравится твой ответ» → complaint, confidence: 0.86

{format_instructions}"""


def create_classifier_chain(model):
    """Создаёт LCEL-цепочку классификации: вход → промпт → модель → PydanticOutputParser."""
    parser = PydanticOutputParser(pydantic_object=Classification)

    prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFICATION_SYSTEM_PROMPT),
        ("human", "Запрос: {query}"),
    ])

    chain = (
        {
            "query": RunnablePassthrough(),
            "format_instructions": lambda _: parser.get_format_instructions(),
        }
        | prompt
        | model
        | parser
    )
    return chain


def classify(chain, query: str) -> Classification:
    """Классифицирует запрос, возвращая fallback при ошибке парсинга."""
    try:
        return chain.invoke(query)
    except Exception:
        return Classification(
            request_type=RequestType.UNKNOWN,
            confidence=0.5,
            reasoning="Ошибка парсинга ответа модели",
        )
