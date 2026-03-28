"""Тесты для ключевых компонентов ассистента."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import AssistantResponse, Classification, RequestType
from characters import CHARACTER_PROMPTS, HANDLER_PROMPTS
from memory import MemoryManager


# --- Тесты Pydantic-моделей ---


class TestRequestType:
    def test_all_values_exist(self):
        assert RequestType.QUESTION == "question"
        assert RequestType.TASK == "task"
        assert RequestType.SMALL_TALK == "small_talk"
        assert RequestType.COMPLAINT == "complaint"
        assert RequestType.UNKNOWN == "unknown"

    def test_enum_count(self):
        assert len(RequestType) == 5


class TestClassification:
    def test_valid_classification(self):
        c = Classification(
            request_type=RequestType.QUESTION,
            confidence=0.95,
            reasoning="Пользователь задал вопрос",
        )
        assert c.request_type == RequestType.QUESTION
        assert c.confidence == 0.95
        assert c.reasoning == "Пользователь задал вопрос"

    def test_confidence_upper_bound(self):
        with pytest.raises(Exception):
            Classification(
                request_type=RequestType.QUESTION,
                confidence=1.5,
                reasoning="Invalid",
            )

    def test_confidence_lower_bound(self):
        with pytest.raises(Exception):
            Classification(
                request_type=RequestType.QUESTION,
                confidence=-0.1,
                reasoning="Invalid",
            )

    def test_boundary_values(self):
        c_zero = Classification(request_type=RequestType.UNKNOWN, confidence=0.0, reasoning="min")
        c_one = Classification(request_type=RequestType.UNKNOWN, confidence=1.0, reasoning="max")
        assert c_zero.confidence == 0.0
        assert c_one.confidence == 1.0


class TestAssistantResponse:
    def test_valid_response(self):
        r = AssistantResponse(
            content="Привет!",
            request_type=RequestType.SMALL_TALK,
            confidence=0.9,
            tokens_used=10,
        )
        assert r.content == "Привет!"
        assert r.tokens_used == 10

    def test_default_tokens(self):
        r = AssistantResponse(
            content="test",
            request_type=RequestType.UNKNOWN,
            confidence=0.5,
        )
        assert r.tokens_used == 0


# --- Тесты памяти ---


class TestMemoryBuffer:
    def test_add_messages(self):
        mem = MemoryManager(strategy="buffer")
        mem.add_user_message("Hello")
        mem.add_ai_message("Hi")
        assert mem.message_count() == 2

    def test_buffer_trimming(self):
        mem = MemoryManager(strategy="buffer", max_messages=4)
        for i in range(6):
            mem.add_user_message(f"msg {i}")
        assert mem.message_count() == 4

    def test_clear_preserves_entities(self):
        mem = MemoryManager()
        mem.add_user_message("Hello")
        mem.add_ai_message("Hi")
        mem.entities = {"имя": "Алексей"}
        mem.clear()
        assert mem.message_count() == 0
        assert mem.entities == {"имя": "Алексей"}

    def test_get_history_returns_copies(self):
        from langchain_core.messages import AIMessage, HumanMessage

        mem = MemoryManager()
        mem.add_user_message("Hello")
        mem.add_ai_message("Hi")
        history = mem.get_history()
        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)

    def test_set_strategy(self):
        mem = MemoryManager(strategy="buffer")
        mem.set_strategy("summary")
        assert mem.strategy == "summary"


class TestMemorySummary:
    def test_summary_strategy_with_mock_model(self):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Пользователь представился как Алексей."
        mock_model.invoke.return_value = mock_response

        mem = MemoryManager(strategy="summary", max_messages=4, model=mock_model)
        for i in range(6):
            mem.add_user_message(f"msg {i}")

        assert mem.summary != ""
        assert mem.message_count() < 6


class TestEntityMemory:
    def test_get_entities_summary_empty(self):
        mem = MemoryManager()
        assert mem.get_entities_summary() == ""

    def test_get_entities_summary_with_data(self):
        mem = MemoryManager()
        mem.entities = {"имя": "Даша", "язык": "Python"}
        summary = mem.get_entities_summary()
        assert "Даша" in summary
        assert "Python" in summary

    def test_extract_entities_without_model(self):
        mem = MemoryManager()
        mem.extract_entities("Меня зовут Алексей")
        assert mem.entities == {}


# --- Тесты характеров ---


class TestCharacters:
    def test_all_characters_defined(self):
        expected = {"friendly", "professional", "sarcastic", "pirate"}
        assert expected == set(CHARACTER_PROMPTS.keys())

    def test_all_handler_prompts_defined(self):
        expected = {"question", "task", "small_talk", "complaint", "unknown"}
        assert expected == set(HANDLER_PROMPTS.keys())

    def test_prompts_are_non_empty_strings(self):
        for name, prompt in CHARACTER_PROMPTS.items():
            assert isinstance(prompt, str) and len(prompt) > 10, f"Empty prompt: {name}"
        for name, prompt in HANDLER_PROMPTS.items():
            assert isinstance(prompt, str) and len(prompt) > 10, f"Empty prompt: {name}"


# --- Тест классификатора (с моком) ---


class TestClassifier:
    def test_classify_fallback_on_error(self):
        from classifier import classify

        broken_chain = MagicMock()
        broken_chain.invoke.side_effect = ValueError("parse error")

        result = classify(broken_chain, "test query")
        assert result.request_type == RequestType.UNKNOWN
        assert result.confidence == 0.5
        assert "Ошибка" in result.reasoning


# --- Тест SmartAssistant (с моком модели) ---


class TestSmartAssistantSetters:
    def _make_assistant(self):
        with patch("smart_assistant.ChatOpenAI"):
            with patch("smart_assistant.create_classifier_chain"):
                with patch("smart_assistant.create_all_handlers", return_value={}):
                    from smart_assistant import SmartAssistant
                    return SmartAssistant(
                        use_cache=False,
                        use_fallback=False,
                        use_entities=False,
                    )

    def test_set_character_valid(self):
        assistant = self._make_assistant()
        assert assistant.set_character("pirate") is True
        assert assistant.character == "pirate"

    def test_set_character_invalid(self):
        assistant = self._make_assistant()
        assert assistant.set_character("nonexistent") is False

    def test_set_memory_strategy_valid(self):
        assistant = self._make_assistant()
        assert assistant.set_memory_strategy("summary") is True
        assert assistant.memory_strategy == "summary"

    def test_set_memory_strategy_invalid(self):
        assistant = self._make_assistant()
        assert assistant.set_memory_strategy("magic") is False

    def test_status_output(self):
        assistant = self._make_assistant()
        status = assistant.status()
        assert "friendly" in status
        assert "buffer" in status
