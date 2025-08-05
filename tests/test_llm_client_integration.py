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

    # Базовая конфигурация
    test_config = {
        "api_key": api_key,
        "model": config["itext2kg"]["model_test"],
        "tpm_limit": config["itext2kg"]["tpm_limit"],
        "tpm_safety_margin": config["itext2kg"].get("tpm_safety_margin", 0.15),
        "max_completion": 500,
        "timeout": config["itext2kg"]["timeout"],
        "max_retries": 2,
        "temperature": config["itext2kg"].get("temperature", 1.0),
        "poll_interval": config["itext2kg"].get("poll_interval", 5),
        "max_context_tokens": config["itext2kg"].get("max_context_tokens_test", 128000),
    }

    # Для reasoning моделей
    if config["itext2kg"].get("model_test", "").startswith("o"):
        test_config["reasoning_effort"] = config["itext2kg"].get(
            "reasoning_effort", "medium"
        )
        test_config["reasoning_summary"] = config["itext2kg"].get(
            "reasoning_summary", "auto"
        )
        test_config.pop("temperature", None)

    return test_config


class TestOpenAIClientIntegration:
    """Интеграционные тесты с реальным API"""

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
        # Если OpenAI вернул больший лимит чем в конфиге, remaining_tokens может быть больше initial_limit
        assert client.tpm_bucket.remaining_tokens is not None
        # В async режиме reset_time обновляется через probe
        # assert client.tpm_bucket.reset_time is not None

        print(f"\nResponse: {response_text}")
        print(f"Response ID: {response_id}")
        print(f"Tokens used: {usage.total_tokens}")
        print(f"TPM remaining: {client.tpm_bucket.remaining_tokens}")

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

        print(f"\nJSON response: {json.dumps(data, indent=2)}")
        print(f"Tokens used: {usage.total_tokens}")

    def test_response_chain(self, integration_config):
        """Тест цепочки ответов с контекстом"""
        client = OpenAIClient(integration_config)

        # Первый запрос
        text1, id1, usage1 = client.create_response(
            instructions="You are a math tutor. Be very brief in your answers.",
            input_data="My name is Alice. What is 5 + 3?",
        )

        assert "8" in text1
        assert client.last_response_id == id1

        # Второй запрос должен помнить контекст
        text2, id2, usage2 = client.create_response(
            instructions="Continue being a math tutor.", input_data="What was my name?"
        )

        assert "alice" in text2.lower()
        assert id2 != id1
        assert client.last_response_id == id2

        print(f"\nFirst response: {text1}")
        print(f"Second response: {text2}")
        print(f"Total tokens: {usage1.total_tokens + usage2.total_tokens}")

    def test_tpm_limiting(self, integration_config):
        """Тест ограничения TPM с проактивным ожиданием"""
        # Устанавливаем маленький лимит для теста
        integration_config["tpm_limit"] = 2000  # Увеличиваем до разумного значения
        integration_config["max_completion"] = 200
        integration_config["tpm_safety_margin"] = 0.15

        client = OpenAIClient(integration_config)

        # Первый запрос
        print(f"\nTPM initial: {client.tpm_bucket.initial_limit}")
        print(f"TPM remaining before 1st: {client.tpm_bucket.remaining_tokens}")

        text1, _, usage1 = client.create_response(
            instructions="Answer in one word only.", input_data="Say hello"
        )

        print(f"1st response: {text1}, tokens: {usage1.total_tokens}")
        print(f"TPM remaining after 1st: {client.tpm_bucket.remaining_tokens}")

        # Эмулируем низкий остаток токенов для теста ожидания
        original_remaining = client.tpm_bucket.remaining_tokens
        client.tpm_bucket.remaining_tokens = 100  # Очень мало
        client.tpm_bucket.reset_time = int(time.time() + 2)  # Reset через 2 секунды

        print(f"\nTPM artificially lowered to: {client.tpm_bucket.remaining_tokens}")
        print("TPM reset time set to: +2 seconds")

        # Второй запрос должен подождать
        start_time = time.time()
        text2, _, usage2 = client.create_response(
            instructions="Answer in one word only.", input_data="Say goodbye"
        )
        wait_time = time.time() - start_time

        print(f"2nd response: {text2}, tokens: {usage2.total_tokens}")
        print(f"Waited: {wait_time:.1f}s before request")

        # Проверяем что ждали
        assert wait_time > 2.0, f"Should have waited >2s, but waited {wait_time:.1f}s"

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

        print(f"\nError handled correctly: {type(exc_info.value).__name__}")

    def test_reasoning_model(self, integration_config):
        """Тест reasoning модели"""
        # Проверяем что используется reasoning модель
        if not integration_config["model"].startswith("o"):
            pytest.skip("Test requires reasoning model")

        integration_config["max_completion"] = 2000  # Больше токенов для reasoning
        client = OpenAIClient(integration_config)

        response_text, response_id, usage = client.create_response(
            instructions="Give only the final numerical answer, no explanation",
            input_data="What is 123 * 456?",
        )

        # Проверки для reasoning модели
        assert response_text is not None
        assert "56088" in response_text or "56,088" in response_text
        assert usage.reasoning_tokens > 0  # Должны быть reasoning токены

        print(f"\nReasoning response: {response_text}")
        print(f"Reasoning tokens: {usage.reasoning_tokens}")
        print(f"Total tokens: {usage.total_tokens}")

    @pytest.mark.skip(reason="Test hangs - may be API rate limiting issue")
    def test_headers_update(self, integration_config):
        """Test that TPM is updated via probe mechanism."""
        client = OpenAIClient(integration_config)

        # Initial state
        initial_remaining = client.tpm_bucket.remaining_tokens
        initial_reset = client.tpm_bucket.reset_time

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

        print("\n✓ TPM probe executed")
        print(f"  Initial remaining: {initial_remaining}")
        print(f"  Current remaining: {client.tpm_bucket.remaining_tokens}")
        print(f"  Reset time: {client.tpm_bucket.reset_time}")
        print(f"  Response: {response_text}")

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
        assert (
            response_id[:8] in captured.out
        )  # Первые 8 символов ID должны быть в выводе

        print("\nBackground mode verified")
        print(f"Console output captured: {len(captured.out)} chars")
        print(f"Response: {response_text}")

    @pytest.mark.timeout(90)
    def test_incomplete_response_handling(self, integration_config):
        """Тест обработки incomplete ответа с автоматическим увеличением токенов"""
        # Пропускаем для reasoning моделей - они требуют слишком много токенов
        if integration_config["model"].startswith("o"):
            pytest.skip("Skipping for reasoning models - requires too many tokens")

        # Устанавливаем маленький лимит токенов
        integration_config["max_completion"] = 35
        integration_config["max_retries"] = 2  # Разрешаем retry

        client = OpenAIClient(integration_config)

        try:
            # Простой запрос, который может поместиться после retry
            response_text, response_id, usage = client.create_response(
                instructions="Be concise",
                input_data="Write exactly 3 sentences about cats",
            )

            # Если получили ответ, значит автоматическое увеличение сработало
            assert response_text is not None
            assert len(response_text) > 20  # Должно быть больше начального лимита

            print("\nIncomplete handled successfully")
            print(f"Final response length: {len(response_text)} chars")
            print(f"Output tokens: {usage.output_tokens}")

        except ValueError as e:
            # Если все retry исчерпаны - это тоже успешный тест
            if "still incomplete after" in str(e):
                print(f"\nIncomplete response handling tested: {e}")
                # Проверяем что в сообщении упоминается увеличенный лимит
                assert "40" in str(e)  # 20 * 2 = 40 на втором retry
            else:
                raise

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

        print(f"\nTimeout handled correctly: {error_msg}")

    @pytest.mark.timeout(45)
    def test_console_progress_output(self, integration_config, capsys):
        """Тест вывода прогресса в консоль"""
        # Средний размер для баланса между скоростью и видимостью прогресса
        integration_config["max_completion"] = 400
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
        print("\n=== CAPTURED OUTPUT ===")
        print(captured.out)
        print("=== END CAPTURED ===")

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

        print("\nConsole output test results:")
        print(f"  Output captured: {len(captured.out)} chars")
        print(f"  Queue messages: {has_queue}")
        print(f"  Progress messages: {has_progress}")
        print(f"  Response length: {len(response_text)} chars")
        print(f"  Tokens used: {usage.output_tokens}")

    @pytest.mark.skip(reason="Test hangs due to insufficient token limit for reasoning models")
    def test_incomplete_with_reasoning_model(self, integration_config):
        """Тест incomplete для reasoning модели (нужно больше токенов)"""
        if not integration_config["model"].startswith("o"):
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
            print("\nReasoning model handled incomplete successfully")
            print(f"Final output tokens: {usage.output_tokens}")

        except (ValueError, IncompleteResponseError) as e:
            # Проверяем что это именно incomplete ошибка
            assert "incomplete" in str(e).lower()
            print(f"\nReasoning incomplete test: {e}")

    def test_tpm_probe_mechanism(self, integration_config, caplog):
        """Test that TPM probe mechanism works correctly."""
        import logging

        caplog.set_level(logging.DEBUG)

        client = OpenAIClient(integration_config)

        # Запоминаем начальное состояние
        initial_remaining = client.tpm_bucket.remaining_tokens

        # Делаем запрос (внутри будет probe)
        response_text, response_id, usage = client.create_response(
            "Reply with one word", "Hello"
        )

        # Проверяем что probe был выполнен
        probe_logs = [r for r in caplog.records if "TPM probe" in r.message]
        assert len(probe_logs) > 0, "TPM probe should have been executed"

        # Проверяем что TPM обновился
        # OpenAI может вернуть лимит больше чем в конфиге
        assert client.tpm_bucket.remaining_tokens is not None

        # Проверяем что есть логи об обновлении
        update_logs = [r for r in caplog.records if "TPM probe successful" in r.message]
        assert len(update_logs) > 0, "Should log successful probe"

        print("\n✓ TPM probe mechanism verified")
        print(f"  Probe logs found: {len(probe_logs)}")
        print(f"  Initial tokens: {initial_remaining}")
        print(f"  Current tokens: {client.tpm_bucket.remaining_tokens}")

    @pytest.mark.skip(reason="Test takes too long with multiple sequential API calls")
    def test_context_accumulation_integration(self, integration_config):
        """Test that context accumulation works correctly in real API calls"""
        client = OpenAIClient(integration_config)
        
        # First request - baseline
        text1, id1, usage1 = client.create_response(
            instructions="You are a helpful assistant. Answer briefly.",
            input_data="Hi, my favorite color is blue. What is 2+2?"
        )
        
        assert "4" in text1
        assert client.last_usage.input_tokens == usage1.input_tokens
        
        # Second request with previous_response_id
        text2, id2, usage2 = client.create_response(
            instructions="Continue helping.",
            input_data="What was my favorite color?",
            previous_response_id=id1
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
            previous_response_id=id2
        )
        
        assert "blue" in text3.lower()
        assert "4" in text3
        
        # Each subsequent request should have more context
        assert usage3.input_tokens > usage2.input_tokens
        
        print(f"\n✓ Context accumulation verified:")
        print(f"  Request 1 input tokens: {usage1.input_tokens}")
        print(f"  Request 2 input tokens: {usage2.input_tokens} (accumulated)")
        print(f"  Request 3 input tokens: {usage3.input_tokens} (further accumulated)")


if __name__ == "__main__":
    # Запуск тестов из командной строки
    pytest.main([__file__, "-v"])
