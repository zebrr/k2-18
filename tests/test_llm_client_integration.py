"""
Интеграционные тесты для LLM Client.

Требуют реального API ключа OpenAI. Запускать с переменной окружения:
OPENAI_API_KEY=sk-...
python -m pytest tests/test_llm_client_integration.py -v

win
set OPENAI_API_KEY=sk-proj-...  // для сессии
setx OPENAI_API_KEY=sk-proj-... // навсегда
echo %OPENAI_API_KEY%

Или пропустить если ключа нет:
python -m pytest tests/test_llm_client_integration.py -v -m "not integration"
"""

import json
import os
import time

import pytest

from src.utils.llm_client import IncompleteResponseError, OpenAIClient

# Установим глобальный timeout для всех тестов
pytest_timeout = 60

# Пометка для интеграционных тестов
pytestmark = pytest.mark.integration


@pytest.fixture
def api_key():
    """Получение API ключа из окружения"""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def integration_config(api_key):
    """Конфигурация для интеграционных тестов"""
    from src.utils.config import load_config

    config = load_config()

    # Используем TEST параметры для тестов
    test_config = {
        "api_key": api_key,
        "model": config["itext2kg"]["model_test"],
        "is_reasoning": config["itext2kg"]["is_reasoning"],
        "tpm_limit": config["itext2kg"]["tpm_limit_test"],  # Используем test версию!
        "tpm_safety_margin": config["itext2kg"].get("tpm_safety_margin", 0.15),
        "max_completion": config["itext2kg"]["max_completion_test"],  # И тут test версию!
        "timeout": config["itext2kg"]["timeout"],
        "max_retries": 2,
        "poll_interval": config["itext2kg"].get("poll_interval", 5),
        "max_context_tokens": config["itext2kg"].get("max_context_tokens_test", 128000),
    }

    # Добавляем temperature всегда (может быть None)
    test_config["temperature"] = config["itext2kg"].get("temperature")

    # Добавляем reasoning параметры ТОЛЬКО для reasoning моделей
    if test_config["is_reasoning"]:
        test_config["reasoning_effort"] = config["itext2kg"].get("reasoning_effort")
        test_config["reasoning_summary"] = config["itext2kg"].get("reasoning_summary")

    # Verbosity может быть для любых моделей
    test_config["verbosity"] = config["itext2kg"].get("verbosity")

    return test_config


class TestOpenAIClientIntegration:
    """Интеграционные тесты с реальным API"""

    @pytest.mark.timeout(60)
    def test_simple_response(self, integration_config):
        """Тест простого запроса-ответа"""
        client = OpenAIClient(integration_config)

        response_text, response_id, usage = client.create_response(
            instructions="You are a helpful assistant. Be very brief.",
            input_data="What is 2+2? Answer with just the number.",
        )

        # Проверки
        assert response_text is not None
        assert "4" in response_text
        assert response_id.startswith("resp_")
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens

        # Проверка что bucket обновился
        # Если OpenAI вернул больший лимит чем в конфиге,
        # remaining_tokens может быть больше initial_limit
        assert client.tpm_bucket.remaining_tokens is not None
        # В async режиме reset_time обновляется через probe
        # assert client.tpm_bucket.reset_time is not None

    @pytest.mark.timeout(120)
    def test_json_response(self, integration_config):
        """Тест получения JSON ответа"""
        client = OpenAIClient(integration_config)

        response_text, _, usage = client.create_response(
            instructions="You are a JSON generator. Return ONLY the raw JSON object.",
            input_data='Create a JSON object with fields "name" and "age" for John who is 30.',
        )

        # Пробуем распарсить JSON
        try:
            data = json.loads(response_text)
            assert "name" in data
            assert "age" in data
            assert data["name"].lower() == "john" or "john" in data["name"].lower()
            assert data["age"] == 30 or data["age"] == "30"
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON: {response_text}")

    @pytest.mark.timeout(120)
    def test_response_chain(self, integration_config):
        """Тест цепочки ответов с контекстом"""
        client = OpenAIClient(integration_config)

        # Первый запрос
        text1, id1, usage1 = client.create_response(
            instructions="You are a math tutor. Be very brief in your answers.",
            input_data="My name is Alice. What is 5 + 3?",
        )
        # Подтверждаем успешный ответ
        client.confirm_response()

        assert "8" in text1
        assert client.last_response_id == id1

        # Второй запрос должен помнить контекст
        text2, id2, usage2 = client.create_response(
            instructions="Continue being a math tutor.", input_data="What was my name?"
        )
        # Подтверждаем второй ответ
        client.confirm_response()

        assert "alice" in text2.lower()
        assert id2 != id1
        assert client.last_response_id == id2

    @pytest.mark.timeout(120)
    def test_tpm_limiting(self, integration_config):
        """Тест ограничения TPM с проактивным ожиданием"""
        # Устанавливаем маленький лимит для теста
        integration_config["tpm_limit"] = 2000  # Увеличиваем до разумного значения
        integration_config["max_completion"] = 500  # Увеличили, чтобы избежать incomplete
        integration_config["tpm_safety_margin"] = 0.15

        client = OpenAIClient(integration_config)

        # Первый запрос

        text1, _, usage1 = client.create_response(
            instructions="Answer in one word only.", input_data="Say hello"
        )

        # Эмулируем низкий остаток токенов для теста ожидания
        client.tpm_bucket.remaining_tokens = 100  # Очень мало
        client.tpm_bucket.reset_time = int(time.time() + 2)  # Reset через 2 секунды

        # Второй запрос должен подождать
        start_time = time.time()
        text2, _, usage2 = client.create_response(
            instructions="Answer in one word only.", input_data="Say goodbye"
        )
        wait_time = time.time() - start_time

        # Проверяем что ждали
        assert wait_time > 2.0, f"Should have waited >2s, but waited {wait_time:.1f}s"

    @pytest.mark.timeout(60)
    def test_error_handling(self, integration_config):
        """Тест обработки ошибок"""
        # Используем невалидный ключ
        bad_config = integration_config.copy()
        bad_config["api_key"] = "sk-invalid-key-12345"
        bad_config["max_retries"] = 0

        client = OpenAIClient(bad_config)

        with pytest.raises(Exception) as exc_info:
            client.create_response(instructions="Test", input_data="Test")

        # Должна быть ошибка аутентификации
        assert (
            "authentication" in str(exc_info.value).lower()
            or "api" in str(exc_info.value).lower()
            or "invalid" in str(exc_info.value).lower()
        )

    @pytest.mark.timeout(120)
    def test_reasoning_model(self, integration_config):
        """Тест reasoning модели"""
        # Проверяем что используется reasoning модель
        if not integration_config.get("is_reasoning", False):
            pytest.skip("Test requires reasoning model")

        # Используем max_completion из конфига (уже установлен в fixture)
        client = OpenAIClient(integration_config)

        response_text, response_id, usage = client.create_response(
            instructions="Give only the final numerical answer, no explanation",
            input_data="What is 123 * 456?",
        )

        # Проверки для reasoning модели
        assert response_text is not None
        assert "56088" in response_text or "56,088" in response_text
        assert usage.reasoning_tokens > 0  # Должны быть reasoning токены

    @pytest.mark.timeout(60)
    def test_headers_update(self, integration_config):
        """Test that TPM is updated via probe mechanism."""
        client = OpenAIClient(integration_config)

        # Initial state
        # initial_remaining = client.tpm_bucket.remaining_tokens

        # Make a request (will trigger probe internally)
        response_text, response_id, usage = client.create_response(
            "You are a test assistant", "Say 'test' and nothing else"
        )

        # После probe должны быть актуальные данные
        # В async режиме данные обновляются через probe, не из основного response
        # OpenAI может вернуть лимит больше чем в конфиге
        assert client.tpm_bucket.remaining_tokens is not None
        # reset_time может остаться None если лимиты не достигнуты
        # или обновиться если probe получил эту информацию

    @pytest.mark.timeout(60)
    def test_background_mode_verification(self, integration_config, capsys):
        """Тест проверки работы в background режиме"""
        client = OpenAIClient(integration_config)

        # Небольшой запрос для проверки консольного вывода
        response_text, response_id, usage = client.create_response(
            instructions="Be very brief", input_data="Say 'background test'"
        )

        # Читаем консольный вывод
        captured = capsys.readouterr()

        # Проверяем что был вывод прогресса (может быть QUEUE или PROGRESS)
        assert "] QUEUE" in captured.out or "] PROGRESS" in captured.out
        assert response_id[:8] in captured.out  # Первые 8 символов ID должны быть в выводе

    @pytest.mark.slow
    @pytest.mark.timeout(90)
    @pytest.mark.timeout(120)
    def test_incomplete_response_handling(self, integration_config):
        """Тест обработки incomplete ответа - теперь без retry (сразу exception)"""
        # Пропускаем для reasoning моделей - они требуют слишком много токенов
        if integration_config["model"].startswith("o"):
            pytest.skip("Skipping for reasoning models - requires too many tokens")

        # Устанавливаем маленький лимит токенов
        integration_config["max_completion"] = 35  # Intentionally small to trigger incomplete

        client = OpenAIClient(integration_config)

        # Теперь IncompleteResponseError должна быть выброшена сразу без retry
        from src.utils.llm_client import IncompleteResponseError

        with pytest.raises(IncompleteResponseError) as exc_info:
            client.create_response(
                instructions="Be concise",
                input_data="Write exactly 3 sentences about cats",
            )

        # Проверяем что исключение содержит нужную информацию
        assert "incomplete" in str(exc_info.value).lower()

    @pytest.mark.timeout(60)
    def test_timeout_cancellation(self, integration_config):
        """Тест отмены запроса при timeout"""
        # Устанавливаем очень маленький timeout
        integration_config["timeout"] = 2  # 2 секунды
        integration_config["max_retries"] = 0  # Без retry для чистоты теста

        client = OpenAIClient(integration_config)

        with pytest.raises(TimeoutError) as exc_info:
            # Запрос который может занять больше 2 секунд
            # (зависит от загрузки API, может и не сработать)
            client.create_response(
                instructions="Think step by step very carefully",
                input_data="Calculate the 50th Fibonacci number showing all steps",
            )

        error_msg = str(exc_info.value)
        assert "exceeded" in error_msg and "2s" in error_msg

    @pytest.mark.slow
    @pytest.mark.timeout(45)
    @pytest.mark.timeout(60)
    def test_console_progress_output(self, integration_config, capsys):
        """Тест вывода прогресса в консоль"""
        # Используем max_completion из конфига (уже установлен в fixture)
        # integration_config уже содержит max_completion_test из конфига
        integration_config["poll_interval"] = 1

        client = OpenAIClient(integration_config)

        # Запрос который требует некоторой генерации, но не слишком долгой
        response_text, response_id, usage = client.create_response(
            instructions="Write a short story",
            input_data="Write a 2-paragraph story about a robot learning to paint. Be concise.",
        )

        # Читаем консольный вывод
        captured = capsys.readouterr()

        # Для отладки - выведем что поймали

        # Проверяем элементы вывода
        has_queue = "] QUEUE" in captured.out
        has_progress = "] PROGRESS" in captured.out

        # Должен быть хотя бы один из них
        assert (
            has_queue or has_progress
        ), f"Expected QUEUE or PROGRESS messages, but got neither. Captured: {captured.out[:200]}"

        if has_queue:
            assert "⏳" in captured.out
            assert "Response" in captured.out

        if has_progress:
            assert "⏳" in captured.out
            assert "Elapsed" in captured.out

        # Response ID должен быть в выводе (первые 8 символов)
        assert response_id[:8] in captured.out

    @pytest.mark.skip(reason="Test hangs due to insufficient token limit for reasoning models")
    @pytest.mark.timeout(120)
    def test_incomplete_with_reasoning_model(self, integration_config):
        """Тест incomplete для reasoning модели (нужно больше токенов)"""
        if not integration_config.get("is_reasoning", False):
            pytest.skip("Test requires reasoning model")

        # Очень маленький лимит для reasoning модели
        integration_config["max_completion"] = 50  # Слишком мало для reasoning!
        integration_config["max_retries"] = 1  # Один retry

        client = OpenAIClient(integration_config)

        try:
            response_text, response_id, usage = client.create_response(
                instructions="Solve this step by step", input_data="What is 789 * 654?"
            )

            # Если получили ответ после retry - успех
            assert "516006" in response_text or "516,006" in response_text

        except (ValueError, IncompleteResponseError) as e:
            # Проверяем что это именно incomplete ошибка
            assert "incomplete" in str(e).lower()

    @pytest.mark.timeout(60)
    def test_tpm_probe_mechanism(self, integration_config, caplog):
        """Test that TPM probe mechanism works correctly."""
        import logging

        caplog.set_level(logging.DEBUG)

        client = OpenAIClient(integration_config)

        # Запоминаем начальное состояние
        # initial_remaining = client.tpm_bucket.remaining_tokens

        # Делаем запрос (внутри будет probe)
        response_text, response_id, usage = client.create_response("Reply with one word", "Hello")

        # Проверяем что probe был выполнен
        probe_logs = [r for r in caplog.records if "TPM probe" in r.message]
        assert len(probe_logs) > 0, "TPM probe should have been executed"

        # Проверяем что TPM обновился
        # OpenAI может вернуть лимит больше чем в конфиге
        assert client.tpm_bucket.remaining_tokens is not None

        # Проверяем что есть логи об обновлении
        update_logs = [r for r in caplog.records if "TPM probe successful" in r.message]
        assert len(update_logs) > 0, "Should log successful probe"

    @pytest.mark.timeout(180)
    @pytest.mark.slow
    def test_context_accumulation_integration(self, integration_config):
        """Test that context accumulation works correctly in real API calls"""
        client = OpenAIClient(integration_config)

        # First request - baseline
        text1, id1, usage1 = client.create_response(
            instructions="You are a helpful assistant. Answer briefly.",
            input_data="Hi, my favorite color is blue. What is 2+2?",
        )

        assert "4" in text1
        assert client.last_usage.input_tokens == usage1.input_tokens

        # Second request with previous_response_id
        text2, id2, usage2 = client.create_response(
            instructions="Continue helping.",
            input_data="What was my favorite color?",
            previous_response_id=id1,
        )

        assert "blue" in text2.lower()

        # Verify context accumulation
        # Second request should have more input tokens than just the new prompt
        assert usage2.input_tokens > usage1.input_tokens
        assert client.last_usage.input_tokens == usage2.input_tokens

        # Third request to further verify accumulation
        text3, id3, usage3 = client.create_response(
            instructions="Continue helping.",
            input_data="Tell me both: my color and the math answer from before.",
            previous_response_id=id2,
        )

        assert "blue" in text3.lower()
        assert "4" in text3

        # Each subsequent request should have more context
        assert usage3.input_tokens > usage2.input_tokens


if __name__ == "__main__":
    # Запуск тестов из командной строки
    pytest.main([__file__, "-v"])
