# Умный ассистент с характером

CLI чат-бот на базе LangChain, который классифицирует запросы, выбирает стратегию ответа и помнит контекст диалога.

## Установка

```bash
pip install -r requirements.txt
```

## Настройка API-ключа

Создайте файл `.env` в корне проекта:

```bash
cp .env.example .env
```

Откройте `.env` и впишите свой ключ:

```
OPENAI_API_KEY=sk-ваш-ключ
```

## Запуск

```bash
# По умолчанию (friendly, buffer memory, streaming)
python smart_assistant.py

# С выбранным характером
python smart_assistant.py --character sarcastic

# Со стратегией памяти summary
python smart_assistant.py --memory summary

# С конкретной моделью
python smart_assistant.py --model gpt-5-mini

# С entity memory (запоминает факты даже после /clear)
python smart_assistant.py --entities

# Без потокового вывода
python smart_assistant.py --no-stream

# Комбинация флагов
python smart_assistant.py --character pirate --memory summary --entities --model gpt-4o-mini
```

### Все аргументы

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--character` | `friendly` | Характер: `friendly`, `professional`, `sarcastic`, `pirate` |
| `--memory` | `buffer` | Стратегия памяти: `buffer`, `summary` |
| `--model` | `gpt-4o-mini` | Модель LLM |
| `--fallback-model` | `gpt-3.5-turbo` | Резервная модель при ошибках |
| `--entities` | выкл | Включить извлечение сущностей |
| `--no-stream` | — | Отключить потоковый вывод |
| `--no-cache` | — | Отключить кэширование |
| `--no-fallback` | — | Отключить fallback-модель |

## Команды в чате

| Команда | Действие |
|---|---|
| `/character <name>` | Сменить характер |
| `/memory <strategy>` | Сменить стратегию памяти |
| `/clear` | Очистить историю диалога |
| `/status` | Показать текущие настройки |
| `/help` | Справка по командам |
| `/quit` | Выход |

## Пример сессии

```
🤖 Умный ассистент с характером
Характер: friendly | Память: buffer
────────────────────────────────

> Привет! Меня зовут Алексей
[small_talk] Привет, Алексей! Рад знакомству! Чем могу помочь?
confidence: 0.95 | tokens: ~42

> Что такое LCEL?
[question] LCEL — это LangChain Expression Language, декларативный синтаксис
для описания цепочек обработки данных через оператор |
confidence: 0.91 | tokens: ~68

> /character sarcastic
✓ Характер изменён на: sarcastic

> Как меня зовут?
[question] Ну вы же только что представились — Алексей. Память у меня
получше, чем у некоторых. 😏
confidence: 0.93 | tokens: ~35
```

## Тесты

```bash
python -m pytest tests/test_assistant.py -v
```

## Структура проекта

```
├── smart_assistant.py   # CLI-интерфейс и класс SmartAssistant
├── models.py            # Pydantic-модели (RequestType, Classification, AssistantResponse)
├── classifier.py        # LCEL-цепочка классификации запросов
├── handlers.py          # Обработчики по типам запросов + роутинг
├── characters.py        # Промпты характеров и обработчиков
├── memory.py            # MemoryManager (buffer/summary + entity memory)
├── requirements.txt     # Зависимости
├── .env.example         # Шаблон для API-ключа
└── tests/
    └── test_assistant.py
```
