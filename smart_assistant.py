"""
Умный ассистент с характером — CLI-приложение.

Запуск:
    python smart_assistant.py                          # по умолчанию
    python smart_assistant.py --character sarcastic     # с выбранным характером
    python smart_assistant.py --memory summary          # с суммаризацией памяти
    python smart_assistant.py --model gpt-5-mini        # с конкретной моделью
    python smart_assistant.py --entities                # с entity memory
    python smart_assistant.py --no-stream               # без потокового вывода

Требуется переменная окружения OPENAI_API_KEY (или файл .env).
"""

import argparse
import sys

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from characters import CHARACTER_PROMPTS
from classifier import classify, create_classifier_chain
from handlers import create_all_handlers, route, route_stream
from memory import MemoryManager
from models import AssistantResponse


class SmartAssistant:
    """Основной класс ассистента: связывает классификатор, роутинг, память и характер."""

    def __init__(
        self,
        character: str = "friendly",
        memory_strategy: str = "buffer",
        model_name: str = "gpt-4o-mini",
        use_cache: bool = True,
        use_fallback: bool = True,
        use_entities: bool = False,
        fallback_model_name: str = "gpt-3.5-turbo",
    ):
        self.character = character
        self.memory_strategy = memory_strategy
        self.model_name = model_name
        self.use_entities = use_entities

        main_model = ChatOpenAI(model=model_name, temperature=0.7)

        if use_fallback:
            fallback_model = ChatOpenAI(model=fallback_model_name, temperature=0.7)
            self.model = main_model.with_fallbacks([fallback_model])
        else:
            self.model = main_model

        if use_cache:
            set_llm_cache(InMemoryCache())

        self.classifier_chain = create_classifier_chain(self.model)
        self.handlers = create_all_handlers(self.model, self.character)
        self.memory = MemoryManager(
            strategy=memory_strategy,
            model=self.model,
        )

    def process(self, user_input: str) -> AssistantResponse:
        """Обрабатывает запрос: классификация → роутинг → ответ (без стриминга)."""
        if self.use_entities:
            self.memory.extract_entities(user_input)

        classification = classify(self.classifier_chain, user_input)

        history = self._build_history()

        response_text = route(
            self.handlers,
            classification.request_type.value,
            user_input,
            history,
        )

        self.memory.add_user_message(user_input)
        self.memory.add_ai_message(response_text)

        tokens_used = len(response_text.split()) * 2

        return AssistantResponse(
            content=response_text,
            request_type=classification.request_type,
            confidence=classification.confidence,
            tokens_used=tokens_used,
        )

    def process_stream(self, user_input: str) -> AssistantResponse:
        """Обрабатывает запрос с потоковым выводом в терминал."""
        if self.use_entities:
            self.memory.extract_entities(user_input)

        classification = classify(self.classifier_chain, user_input)

        history = self._build_history()

        print(f"[{classification.request_type.value}] ", end="", flush=True)

        full_response = []
        for chunk in route_stream(
            self.handlers,
            classification.request_type.value,
            user_input,
            history,
        ):
            print(chunk, end="", flush=True)
            full_response.append(chunk)

        response_text = "".join(full_response)
        print()

        self.memory.add_user_message(user_input)
        self.memory.add_ai_message(response_text)

        tokens_used = len(response_text.split()) * 2

        response = AssistantResponse(
            content=response_text,
            request_type=classification.request_type,
            confidence=classification.confidence,
            tokens_used=tokens_used,
        )
        print(f"confidence: {response.confidence:.2f} | tokens: ~{response.tokens_used}")

        return response

    def _build_history(self) -> list:
        """Собирает контекст: entity-факты + история сообщений."""
        history = self.memory.get_history()
        entities_summary = self.memory.get_entities_summary()
        if entities_summary:
            history = [SystemMessage(content=entities_summary)] + history
        return history

    def set_character(self, character: str) -> bool:
        if character not in CHARACTER_PROMPTS:
            return False
        self.character = character
        self.handlers = create_all_handlers(self.model, self.character)
        return True

    def set_memory_strategy(self, strategy: str) -> bool:
        if strategy not in ("buffer", "summary"):
            return False
        self.memory_strategy = strategy
        self.memory.set_strategy(strategy)
        return True

    def clear_memory(self):
        self.memory.clear()

    def status(self) -> str:
        entities_info = self.memory.entities if self.memory.entities else "нет"
        return (
            f"Характер: {self.character}\n"
            f"Память: {self.memory_strategy} ({self.memory.message_count()} сообщений)\n"
            f"Модель: {self.model_name}\n"
            f"Entity memory: {'вкл' if self.use_entities else 'выкл'}\n"
            f"Сущности: {entities_info}"
        )


def handle_command(command: str, assistant: SmartAssistant):
    """Обрабатывает slash-команды CLI."""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        print("До свидания!")
        sys.exit(0)

    elif cmd == "/help":
        print("Доступные команды:")
        print("  /clear              — очистить историю диалога")
        print("  /character <name>   — сменить характер (friendly, professional, sarcastic, pirate)")
        print("  /memory <strategy>  — сменить стратегию памяти (buffer, summary)")
        print("  /status             — показать текущие настройки")
        print("  /help               — эта справка")
        print("  /quit               — выход")

    elif cmd == "/clear":
        assistant.clear_memory()
        print("✓ История диалога очищена")

    elif cmd == "/character":
        if not arg:
            print(f"Доступные характеры: {', '.join(CHARACTER_PROMPTS.keys())}")
            return
        if assistant.set_character(arg):
            print(f"✓ Характер изменён на: {arg}")
        else:
            print(f"✗ Неизвестный характер: {arg}. Доступные: {', '.join(CHARACTER_PROMPTS.keys())}")

    elif cmd == "/memory":
        if not arg:
            print("Доступные стратегии: buffer, summary")
            return
        if assistant.set_memory_strategy(arg):
            print(f"✓ Стратегия памяти изменена на: {arg}")
        else:
            print(f"✗ Неизвестная стратегия: {arg}. Доступные: buffer, summary")

    elif cmd == "/status":
        print(assistant.status())

    else:
        print(f"✗ Неизвестная команда: {cmd}. Введите /help для справки")


def main():
    parser = argparse.ArgumentParser(description="Умный ассистент с характером")
    parser.add_argument(
        "--character", default="friendly",
        choices=list(CHARACTER_PROMPTS.keys()),
        help="Характер ассистента (default: friendly)",
    )
    parser.add_argument(
        "--memory", default="buffer",
        choices=["buffer", "summary"],
        help="Стратегия памяти (default: buffer)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Модель LLM (default: gpt-4o-mini)")
    parser.add_argument("--fallback-model", default="gpt-3.5-turbo", help="Fallback-модель")
    parser.add_argument("--no-stream", action="store_true", help="Отключить потоковый вывод")
    parser.add_argument("--no-cache", action="store_true", help="Отключить кэширование")
    parser.add_argument("--no-fallback", action="store_true", help="Отключить fallback-модель")
    parser.add_argument("--entities", action="store_true", help="Включить entity memory")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    assistant = SmartAssistant(
        character=args.character,
        memory_strategy=args.memory,
        model_name=args.model,
        use_cache=not args.no_cache,
        use_fallback=not args.no_fallback,
        use_entities=args.entities,
        fallback_model_name=args.fallback_model,
    )

    use_stream = not args.no_stream

    print("🤖 Умный ассистент с характером")
    print(f"Характер: {args.character} | Память: {args.memory}")
    print("─" * 40)
    print("Введите /help для списка команд\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nДо свидания!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            handle_command(user_input, assistant)
            print()
            continue

        try:
            if use_stream:
                assistant.process_stream(user_input)
            else:
                response = assistant.process(user_input)
                print(f"[{response.request_type.value}] {response.content}")
                print(f"confidence: {response.confidence:.2f} | tokens: ~{response.tokens_used}")
        except Exception as e:
            print(f"Ошибка: {e}")

        print()


if __name__ == "__main__":
    main()
