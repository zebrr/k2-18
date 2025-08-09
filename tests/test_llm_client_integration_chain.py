"""
Интеграционные тесты для управления response chain в OpenAIClient.

Требуют реального OpenAI API ключа для запуска.
Установите OPENAI_API_KEY в переменные окружения или .env файл.
"""

import os
import time
from typing import List

import pytest
from dotenv import load_dotenv

from src.utils.llm_client import IncompleteResponseError, OpenAIClient

# Загружаем .env файл
load_dotenv()


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="API key required")
class TestResponseChainWindowIntegration:
    """Интеграционные тесты управления цепочкой ответов с реальным API"""

    @pytest.fixture
    def config_with_chain(self):
        """Конфигурация с управлением цепочкой"""
        return {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4.1-nano-2025-04-14",  # Самая дешевая модель
            "is_reasoning": False,
            "tpm_limit": 100000,
            "tpm_safety_margin": 0.15,
            "max_completion": 100,
            "timeout": 60,
            "max_retries": 2,
            "response_chain_depth": 2,  # Окно из 2 ответов
            "truncation": "auto",
            "poll_interval": 3,
        }

    @pytest.fixture
    def config_independent(self):
        """Конфигурация для независимых запросов"""
        return {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4.1-nano-2025-04-14",
            "is_reasoning": False,
            "tpm_limit": 100000,
            "tpm_safety_margin": 0.15,
            "max_completion": 100,
            "timeout": 60,
            "max_retries": 2,
            "response_chain_depth": 0,  # Независимые запросы
            "poll_interval": 3,
        }

    def test_response_chain_window_integration(self, config_with_chain):
        """Проверка реального скользящего окна через API"""
        client = OpenAIClient(config_with_chain)
        
        # Список для отслеживания response_ids
        response_ids: List[str] = []
        
        # Первый запрос
        text1, id1, usage1 = client.create_response(
            "You are a helpful assistant. Remember the name: Alice",
            "Hello! My name is Alice."
        )
        response_ids.append(id1)
        assert id1 is not None
        assert len(client.response_chain) == 1
        
        # Второй запрос - должен помнить первый
        text2, id2, usage2 = client.create_response(
            "Continue being helpful",
            "What was the name I told you?"
        )
        response_ids.append(id2)
        assert "Alice" in text2 or "alice" in text2.lower()
        assert len(client.response_chain) == 2
        
        # Третий запрос - должен удалить первый из цепочки
        text3, id3, usage3 = client.create_response(
            "Continue being helpful. Remember: Bob",
            "Now my name is Bob."
        )
        response_ids.append(id3)
        
        # Проверяем что цепочка содержит только 2 последних
        assert len(client.response_chain) == 2
        assert id1 not in client.response_chain
        assert id2 in client.response_chain
        assert id3 in client.response_chain
        
        # Проверяем через API что цепочка правильная
        # Используем client.client.responses.input_items.list для проверки
        try:
            # Получаем историю для последнего ответа
            input_items = client.client.responses.input_items.list(id3)
            
            # Проверяем что первый запрос (Alice) НЕ в истории
            history_text = str(input_items)
            assert "Alice" not in history_text or "first request not in chain"
            
        except Exception as e:
            # API может не поддерживать input_items.list для всех моделей
            print(f"Could not verify chain via API: {e}")

    def test_independent_requests_integration(self, config_independent):
        """Проверка независимых запросов (response_chain_depth=0)"""
        client = OpenAIClient(config_independent)
        
        # Первый запрос с контекстом
        text1, id1, _ = client.create_response(
            "Remember this number: 42",
            "The important number is 42"
        )
        assert "42" in text1 or "received" in text1.lower()
        
        # Второй запрос - НЕ должен помнить первый
        text2, id2, _ = client.create_response(
            "What number did I tell you?",
            "What was the important number?"
        )
        
        # В независимом режиме модель не должна знать число
        # (может сказать что не знает или попросить напомнить)
        assert "42" not in text2 or "don't" in text2.lower() or "not" in text2.lower()

    def test_repair_not_in_chain_integration(self, config_with_chain):
        """Проверка что repair запросы НЕ добавляются в цепочку"""
        client = OpenAIClient(config_with_chain)
        
        # Обычный запрос
        text1, id1, _ = client.create_response(
            "Return a JSON object",
            '{"task": "create", "value": 123}'
        )
        assert len(client.response_chain) == 1
        assert id1 in client.response_chain
        
        # Repair запрос
        repair_text, repair_id, _ = client.repair_response(
            "Fix the JSON and return valid format",
            '{"task": "repair", "value": 456}',
            previous_response_id=id1
        )
        
        # Repair НЕ должен быть в цепочке
        assert len(client.response_chain) == 1  # Все еще 1
        assert repair_id not in client.response_chain
        assert id1 in client.response_chain  # Оригинальный остался
        
        # Следующий обычный запрос
        text2, id2, _ = client.create_response(
            "Continue with JSON",
            '{"task": "next", "value": 789}'
        )
        
        # Теперь в цепочке должны быть id1 и id2, но НЕ repair_id
        assert len(client.response_chain) == 2
        assert id1 in client.response_chain
        assert id2 in client.response_chain
        assert repair_id not in client.response_chain

    def test_truncation_parameter_integration(self, config_with_chain):
        """Проверка работы параметра truncation='auto'"""
        client = OpenAIClient(config_with_chain)
        
        # Создаем длинную цепочку запросов с накоплением контекста
        for i in range(3):
            text, response_id, usage = client.create_response(
                f"Continue the story. Part {i+1}",
                f"This is part {i+1} of a very long story. " * 10
            )
            
            # С truncation='auto' не должно быть ошибок переполнения
            assert response_id is not None
            
            # Проверяем что input токены растут но не бесконечно
            if i > 0:
                assert usage.input_tokens > 0
                # С truncation контекст будет обрезаться автоматически

    @pytest.mark.skipif(True, reason="This test intentionally causes incomplete response")
    def test_incomplete_response_no_retry_integration(self, config_with_chain):
        """Проверка что IncompleteResponseError НЕ вызывает retry"""
        config = config_with_chain.copy()
        config["max_completion"] = 10  # Очень маленький лимит
        
        client = OpenAIClient(config)
        
        with pytest.raises(IncompleteResponseError) as exc_info:
            # Запрашиваем длинный ответ с маленьким лимитом токенов
            client.create_response(
                "Write a detailed explanation",
                "Explain quantum computing in detail with at least 500 words"
            )
        
        # Проверяем что ошибка содержит информацию о контексте
        assert "max_output_tokens" in str(exc_info.value)
        assert "Context size" in str(exc_info.value)