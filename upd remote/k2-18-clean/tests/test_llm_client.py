"""
Тесты для LLM Client с OpenAI Responses API в асинхронном режиме
"""

import logging
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import openai
import pytest

from src.utils.llm_client import IncompleteResponseError, OpenAIClient, TPMBucket


# Fixtures для конфигурации
@pytest.fixture
def test_config():
    """Тестовая конфигурация"""
    return {
        "api_key": "test-key-123",
        "model": "gpt-4o",
        "is_reasoning": False,  # Явное указание типа модели
        "tpm_limit": 120000,
        "tpm_safety_margin": 0.15,
        "max_completion": 4096,
        "timeout": 45,
        "max_retries": 3,
        "temperature": 1.0,
        "reasoning_effort": "medium",
        "reasoning_summary": "auto",
        "poll_interval": 5,
    }


@pytest.fixture
def reasoning_config():
    """Конфигурация для reasoning модели"""
    return {
        "api_key": "test-key-123",
        "model": "o4-mini",  # reasoning модель
        "is_reasoning": True,  # Явное указание типа модели
        "tpm_limit": 60000,
        "tpm_safety_margin": 0.15,
        "max_completion": 2048,
        "timeout": 30,
        "max_retries": 2,
        "reasoning_effort": "high",
        "reasoning_summary": "detailed",
        "poll_interval": 5,
    }


# Тесты для TPMBucket
class TestTPMBucket:
    """Тесты для контроля лимитов токенов через headers"""

    def test_init(self):
        """Проверка инициализации"""
        bucket = TPMBucket(120000)
        assert bucket.initial_limit == 120000
        assert bucket.remaining_tokens == 120000
        assert bucket.reset_time is None

    def test_update_from_headers(self):
        """Проверка обновления из headers"""
        bucket = TPMBucket(120000)

        headers = {
            "x-ratelimit-remaining-tokens": "95000",
            "x-ratelimit-reset-tokens": "1736622600",  # некоторое время в будущем
        }

        bucket.update_from_headers(headers)

        assert bucket.remaining_tokens == 95000
        assert bucket.reset_time == 1736622600

    def test_update_from_headers_partial(self):
        """Проверка частичного обновления headers"""
        bucket = TPMBucket(120000)

        # Только remaining
        bucket.update_from_headers({"x-ratelimit-remaining-tokens": "80000"})
        assert bucket.remaining_tokens == 80000
        assert bucket.reset_time is None

        # Только reset time
        bucket.update_from_headers({"x-ratelimit-reset-tokens": "1736622600"})
        assert bucket.remaining_tokens == 80000  # не изменилось
        assert bucket.reset_time == 1736622600

    def test_wait_if_needed_sufficient_tokens(self):
        """Проверка когда токенов достаточно"""
        bucket = TPMBucket(120000)
        bucket.remaining_tokens = 10000

        # Запрашиваем 5000, с margin 15% = 5750, достаточно
        with patch("src.utils.llm_client.time.sleep") as mock_sleep:
            bucket.wait_if_needed(5000, safety_margin=0.15)
            mock_sleep.assert_not_called()

    def test_wait_if_needed_insufficient_tokens(self):
        """Проверка ожидания при недостатке токенов"""
        bucket = TPMBucket(120000)
        bucket.remaining_tokens = 1000

        # Устанавливаем reset time в будущем
        current_time = 1000.0  # Фиксированное время для предсказуемости
        bucket.reset_time = int(current_time + 10)  # через 10 секунд

        with patch("src.utils.llm_client.time.sleep") as mock_sleep:
            with patch("src.utils.llm_client.time.time", return_value=current_time):
                # Запрашиваем 5000, с margin 15% = 5750, недостаточно
                bucket.wait_if_needed(5000, safety_margin=0.15)

                # Должны ждать 10 + 0.1 секунд (0.1 добавляется в коде)
                mock_sleep.assert_called_once()
                sleep_time = mock_sleep.call_args[0][0]
                assert abs(sleep_time - 10.1) < 0.01  # Проверяем с точностью

                # После ожидания токены восстанавливаются
                assert bucket.remaining_tokens == 120000

    def test_wait_if_needed_no_reset_time(self):
        """Проверка когда нет reset time"""
        bucket = TPMBucket(120000)
        bucket.remaining_tokens = 100
        bucket.reset_time = None

        with patch("src.utils.llm_client.time.sleep") as mock_sleep:
            # Не должны ждать если нет reset time
            bucket.wait_if_needed(10000, safety_margin=0.15)
            mock_sleep.assert_not_called()


# Mock классы для OpenAI Responses
@dataclass
class MockResponseContent:
    text: str = "Test response"
    type: str = "output_text"


@dataclass
class MockResponseOutput:
    content: list
    status: str = "completed"
    type: str = "message"

    def __init__(self, text="Test response"):
        self.content = [MockResponseContent(text=text)]
        self.type = "message"


@dataclass
class MockUsageDetails:
    reasoning_tokens: int = 0


@dataclass
class MockUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    total_tokens: int = 150
    output_tokens_details: MockUsageDetails = None

    def __init__(self, reasoning_tokens=0):
        self.output_tokens_details = MockUsageDetails(reasoning_tokens)


@dataclass
class MockResponse:
    id: str = "resp_12345"
    output: list = None
    usage: MockUsage = None
    status: str = "completed"
    _headers: dict = None

    def __init__(self, text="Test response", reasoning_tokens=0, headers=None, status="completed"):
        self.output = [MockResponseOutput(text)]
        self.usage = MockUsage(reasoning_tokens)
        self._headers = headers or {}
        self.status = status


@dataclass
class MockReasoningResponse:
    """Mock для reasoning модели с двумя outputs"""

    id: str = "resp_12345"
    output: list = None
    usage: MockUsage = None
    status: str = "completed"
    _headers: dict = None

    def __init__(self, text="Test response", reasoning_tokens=500, headers=None):
        # Reasoning модели имеют два output: reasoning и message
        reasoning_output = Mock(type="reasoning", id="rs_123")
        message_output = MockResponseOutput(text)
        message_output.type = "message"
        self.output = [reasoning_output, message_output]
        self.usage = MockUsage(reasoning_tokens)
        self._headers = headers or {}
        self.status = "completed"


# Helper функция для создания mock retrieve sequence
def create_async_response_sequence(final_response, intermediate_statuses=None):
    """Создает последовательность response объектов для имитации polling"""
    if intermediate_statuses is None:
        intermediate_statuses = ["queued"]

    responses = []

    # Промежуточные статусы
    for status in intermediate_statuses:
        mock_resp = Mock()
        mock_resp.id = final_response.id
        mock_resp.status = status
        responses.append(mock_resp)

    # Финальный ответ
    responses.append(final_response)

    return responses


# Тесты для OpenAIClient
class TestOpenAIClient:
    """Тесты для клиента OpenAI в асинхронном режиме"""

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_init_regular_model(self, mock_openai_class, mock_get_encoding, test_config):
        """Проверка инициализации с обычной моделью"""
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder

        client = OpenAIClient(test_config)

        assert client.config == test_config
        assert client.is_reasoning_model is False
        assert client.last_response_id is None
        assert client.encoder == mock_encoder
        mock_openai_class.assert_called_once_with(api_key="test-key-123", timeout=45)
        mock_get_encoding.assert_called_once_with("o200k_base")

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_init_reasoning_model(self, mock_openai_class, mock_get_encoding, reasoning_config):
        """Проверка инициализации с reasoning моделью"""
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder

        client = OpenAIClient(reasoning_config)

        assert client.is_reasoning_model is True

    def test_init_missing_is_reasoning(self):
        """Проверка что без is_reasoning выбрасывается ошибка"""
        config = {
            "api_key": "test-key",
            "model": "gpt-4o",
            # is_reasoning отсутствует намеренно
            "tpm_limit": 120000,
            "max_completion": 4096,
            "timeout": 45,
            "max_retries": 3,
        }

        with pytest.raises(ValueError) as exc_info:
            OpenAIClient(config)

        assert "is_reasoning" in str(exc_info.value)
        assert "required" in str(exc_info.value).lower()

    @pytest.mark.parametrize("intermediate_status", ["queued", "in_progress"])
    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_successful_response(
        self, mock_openai_class, mock_get_encoding, mock_probe, test_config, intermediate_status
    ):
        """Проверка успешного вызова API с асинхронной логикой (queued и in_progress)"""
        # Настройка mock encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))  # 100 токенов
        mock_get_encoding.return_value = mock_encoder

        # Настройка mock client
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Mock headers
        initial_headers = {
            "x-ratelimit-remaining-tokens": "100000",
            "x-ratelimit-reset-tokens": "60000ms",
        }
        status_headers = {
            "x-ratelimit-remaining-tokens": "95000",
            "x-ratelimit-reset-tokens": "1736622600",
        }

        # Initial response
        initial_response = Mock(id="resp_12345", status=intermediate_status)
        mock_raw_initial = MagicMock()
        mock_raw_initial.headers = initial_headers
        mock_raw_initial.parse.return_value = initial_response

        # Final response
        final_response = MockResponse("Generated text response", headers=status_headers)

        # Создаем sequence для retrieve
        retrieve_responses = create_async_response_sequence(
            final_response, intermediate_statuses=[intermediate_status]
        )

        # Настраиваем mock для create
        mock_client_instance.responses.with_raw_response.create.return_value = mock_raw_initial

        # Настраиваем mock для retrieve с последовательностью
        mock_raw_responses = []
        for resp in retrieve_responses:
            mock_raw = MagicMock()
            mock_raw.headers = status_headers
            mock_raw.parse.return_value = resp
            mock_raw_responses.append(mock_raw)

        mock_client_instance.responses.with_raw_response.retrieve.side_effect = mock_raw_responses

        # Создаем клиент и делаем запрос
        client = OpenAIClient(test_config)

        # Настраиваем mock probe чтобы имитировать обновление bucket
        def update_bucket_to_95000():
            client.tpm_bucket.update_from_headers(
                {
                    "x-ratelimit-remaining-tokens": "95000",
                    "x-ratelimit-reset-tokens": "1736622600",
                }
            )

        mock_probe.side_effect = update_bucket_to_95000

        # Патчим time.sleep чтобы тест не висел
        with patch("src.utils.llm_client.time.sleep"):
            response_text, response_id, usage = client.create_response(
                "System prompt", "User input"
            )

        # Проверки результата
        assert response_text == "Generated text response"
        assert response_id == "resp_12345"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.reasoning_tokens == 0

        # Проверка что TPM bucket обновился
        assert client.tpm_bucket.remaining_tokens == 95000

        # Проверка вызова create с background=True
        create_call_args = mock_client_instance.responses.with_raw_response.create.call_args[1]
        assert create_call_args["background"] is True

        # Проверка что был polling через retrieve
        assert mock_client_instance.responses.with_raw_response.retrieve.call_count == len(
            retrieve_responses
        )

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_reasoning_model_response(self, mock_openai_class, mock_get_encoding, reasoning_config):
        """Проверка работы с reasoning моделью в асинхронном режиме"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Headers
        headers = {
            "x-ratelimit-remaining-tokens": "58000",
            "x-ratelimit-reset-tokens": "30000ms",
        }

        # Initial response
        initial_response = Mock(id="resp_12345", status="queued")
        mock_raw_initial = MagicMock()
        mock_raw_initial.headers = headers
        mock_raw_initial.parse.return_value = initial_response

        # Final response
        final_response = MockReasoningResponse("Reasoning response", reasoning_tokens=500)

        # Настраиваем mock
        mock_client_instance.responses.with_raw_response.create.return_value = mock_raw_initial

        # Для retrieve - сразу completed
        mock_raw_final = MagicMock()
        mock_raw_final.headers = headers
        mock_raw_final.parse.return_value = final_response
        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_final

        client = OpenAIClient(reasoning_config)

        with patch("src.utils.llm_client.time.sleep"):
            response_text, response_id, usage = client.create_response(
                "System prompt", "User input"
            )

        # Проверки
        assert response_text == "Reasoning response"
        assert usage.reasoning_tokens == 500

        # Проверка параметров вызова create
        call_args = mock_client_instance.responses.with_raw_response.create.call_args[1]
        assert call_args["background"] is True
        assert "temperature" not in call_args
        assert call_args["reasoning"] == {"effort": "high", "summary": "detailed"}

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_previous_response_id(
        self, mock_openai_class, mock_get_encoding, mock_probe, test_config
    ):
        """Проверка цепочки ответов с previous_response_id в асинхронном режиме"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Первый запрос
        initial_response1 = Mock(id="resp_001", status="queued")
        mock_raw_initial1 = MagicMock()
        mock_raw_initial1.headers = {}
        mock_raw_initial1.parse.return_value = initial_response1

        final_response1 = MockResponse("First response")
        final_response1.id = "resp_001"
        mock_raw_final1 = MagicMock()
        mock_raw_final1.headers = {}
        mock_raw_final1.parse.return_value = final_response1

        # Второй запрос
        initial_response2 = Mock(id="resp_002", status="queued")
        mock_raw_initial2 = MagicMock()
        mock_raw_initial2.headers = {}
        mock_raw_initial2.parse.return_value = initial_response2

        final_response2 = MockResponse("Second response")
        final_response2.id = "resp_002"
        mock_raw_final2 = MagicMock()
        mock_raw_final2.headers = {}
        mock_raw_final2.parse.return_value = final_response2

        # Настраиваем последовательность
        mock_client_instance.responses.with_raw_response.create.side_effect = [
            mock_raw_initial1,
            mock_raw_initial2,
        ]

        mock_client_instance.responses.with_raw_response.retrieve.side_effect = [
            mock_raw_final1,
            mock_raw_final2,
        ]

        client = OpenAIClient(test_config)

        with patch("src.utils.llm_client.time.sleep"):
            # Первый запрос
            _, response_id1, _ = client.create_response("Prompt 1", "Input 1")
            # Теперь нужно подтвердить response
            client.confirm_response()
            assert client.last_response_id == "resp_001"

            # Второй запрос
            _, response_id2, _ = client.create_response("Prompt 2", "Input 2")
            client.confirm_response()

        # Проверка второго вызова create
        second_call_args = mock_client_instance.responses.with_raw_response.create.call_args_list[
            1
        ][1]
        assert second_call_args["previous_response_id"] == "resp_001"
        assert second_call_args["background"] is True
        assert client.last_response_id == "resp_002"

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.time.sleep")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_rate_limit_retry_with_headers(
        self, mock_openai_class, mock_get_encoding, mock_sleep, mock_probe, test_config
    ):
        """Проверка retry при rate limit с обновлением из headers в асинхронном режиме"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # RateLimitError для первого вызова
        rate_limit_error = openai.RateLimitError("Rate limit exceeded", response=Mock(), body={})
        rate_limit_error.response = Mock()
        rate_limit_error.response.headers = {
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": str(int(time.time() + 60)),
        }

        # Успешные ответы для второй попытки
        initial_response = Mock(id="resp_12345", status="queued")
        mock_raw_initial = MagicMock()
        mock_raw_initial.headers = {"x-ratelimit-remaining-tokens": "100000"}
        mock_raw_initial.parse.return_value = initial_response

        final_response = MockResponse("Success after retry")
        mock_raw_final = MagicMock()
        mock_raw_final.headers = {"x-ratelimit-remaining-tokens": "95000"}
        mock_raw_final.parse.return_value = final_response

        # Настраиваем последовательность
        mock_client_instance.responses.with_raw_response.create.side_effect = [
            rate_limit_error,
            mock_raw_initial,
        ]

        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_final

        client = OpenAIClient(test_config)

        # Настраиваем mock probe чтобы имитировать обновление bucket
        def update_bucket():
            client.tpm_bucket.update_from_headers(
                {
                    "x-ratelimit-remaining-tokens": "95000",
                    "x-ratelimit-reset-tokens": "60000ms",
                }
            )

        mock_probe.side_effect = update_bucket

        response_text, _, _ = client.create_response("Prompt", "Input")

        assert response_text == "Success after retry"
        assert mock_client_instance.responses.with_raw_response.create.call_count == 2
        mock_sleep.assert_called_once_with(20)  # Exponential backoff
        assert client.tpm_bucket.remaining_tokens == 95000

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_wait_before_request(self, mock_openai_class, mock_get_encoding, test_config):
        """Проверка ожидания перед запросом при низком лимите в асинхронном режиме"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(1000))  # 1000 токенов
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Настраиваем ответы
        initial_response = Mock(id="resp_12345", status="queued")
        mock_raw_initial = MagicMock()
        mock_raw_initial.headers = {}
        mock_raw_initial.parse.return_value = initial_response

        final_response = MockResponse("Response")
        mock_raw_final = MagicMock()
        mock_raw_final.headers = {}
        mock_raw_final.parse.return_value = final_response

        mock_client_instance.responses.with_raw_response.create.return_value = mock_raw_initial
        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_final

        client = OpenAIClient(test_config)

        # Устанавливаем низкий лимит
        current_time = 1000.0
        client.tpm_bucket.remaining_tokens = 2000  # Мало для 1000 input + 4096 max_completion
        client.tpm_bucket.reset_time = int(current_time + 5)

        with patch("src.utils.llm_client.time.sleep") as mock_sleep:
            with patch("src.utils.llm_client.time.time", return_value=current_time):
                client.create_response("Prompt", "Input")

                # Должен был подождать 5 + 0.1 секунд
                assert mock_sleep.called
                sleep_time = mock_sleep.call_args[0][0]
                assert abs(sleep_time - 5.1) < 0.01

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_repair_response(self, mock_openai_class, mock_get_encoding, mock_probe, test_config):
        """Проверка repair response в асинхронном режиме"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Первый запрос
        initial_response1 = Mock(id="resp_001", status="queued")
        mock_raw_initial1 = MagicMock()
        mock_raw_initial1.headers = {}
        mock_raw_initial1.parse.return_value = initial_response1

        final_response1 = MockResponse("First response")
        final_response1.id = "resp_001"
        mock_raw_final1 = MagicMock()
        mock_raw_final1.headers = {}
        mock_raw_final1.parse.return_value = final_response1

        # Repair запрос
        initial_response2 = Mock(id="resp_002", status="queued")
        mock_raw_initial2 = MagicMock()
        mock_raw_initial2.headers = {}
        mock_raw_initial2.parse.return_value = initial_response2

        final_response2 = MockResponse("Repaired response")
        final_response2.id = "resp_002"
        mock_raw_final2 = MagicMock()
        mock_raw_final2.headers = {}
        mock_raw_final2.parse.return_value = final_response2

        # Настраиваем последовательность
        mock_client_instance.responses.with_raw_response.create.side_effect = [
            mock_raw_initial1,
            mock_raw_initial2,
        ]

        mock_client_instance.responses.with_raw_response.retrieve.side_effect = [
            mock_raw_final1,
            mock_raw_final2,
        ]

        client = OpenAIClient(test_config)

        with patch("src.utils.llm_client.time.sleep"):
            # Первый запрос
            client.create_response("Original prompt", "Input")
            # Нужно подтвердить, чтобы repair мог использовать last_confirmed_response_id
            client.confirm_response()

            # Repair запрос без explicit previous_response_id (backward compatibility)
            response_text, _, _ = client.repair_response("Original prompt", "Input")

        # Проверка repair вызова - должен использовать last_confirmed_response_id
        repair_call_args = mock_client_instance.responses.with_raw_response.create.call_args_list[1]
        # repair_response вызывает create_response, который вызывает API
        # Проверяем kwargs в API вызове
        assert repair_call_args[1]["previous_response_id"] == "resp_001"
        assert response_text == "Repaired response"

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_repair_response_with_explicit_id(
        self, mock_openai_class, mock_get_encoding, mock_probe, test_config
    ):
        """Проверка repair response с явно заданным previous_response_id"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Настройка ответа для repair
        initial_response = Mock(id="resp_repair", status="queued")
        mock_raw_initial = MagicMock()
        mock_raw_initial.headers = {}
        mock_raw_initial.parse.return_value = initial_response

        final_response = MockResponse("Repaired with explicit ID")
        final_response.id = "resp_repair"
        mock_raw_final = MagicMock()
        mock_raw_final.headers = {}
        mock_raw_final.parse.return_value = final_response

        mock_client_instance.responses.with_raw_response.create.return_value = mock_raw_initial
        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_final

        client = OpenAIClient(test_config)
        client.last_response_id = "resp_last"  # Set different last_response_id

        with patch("src.utils.llm_client.time.sleep"):
            # Repair с явным previous_response_id
            response_text, _, _ = client.repair_response(
                "Instructions with JSON requirement",
                "Input data",
                previous_response_id="resp_explicit",
            )

        # Проверка: должен использовать явно переданный ID, а не last_response_id
        repair_call_args = mock_client_instance.responses.with_raw_response.create.call_args[1]
        assert repair_call_args["instructions"] == "Instructions with JSON requirement"
        assert repair_call_args["previous_response_id"] == "resp_explicit"  # Not "resp_last"!
        assert response_text == "Repaired with explicit ID"

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_refusal_handling(self, mock_openai_class, mock_get_encoding, test_config):
        """Проверка обработки отказа модели в асинхронном режиме"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Initial response
        initial_response = Mock(id="resp_12345", status="queued")
        mock_raw_initial = MagicMock()
        mock_raw_initial.headers = {}
        mock_raw_initial.parse.return_value = initial_response

        # Refusal response
        mock_refusal_content = Mock(type="refusal", refusal="Cannot process this request")
        mock_output = Mock(content=[mock_refusal_content], status="completed", type="message")
        mock_response = Mock(output=[mock_output], status="completed", id="resp_12345")

        mock_raw_final = MagicMock()
        mock_raw_final.headers = {}
        mock_raw_final.parse.return_value = mock_response

        # Настраиваем mock
        mock_client_instance.responses.with_raw_response.create.return_value = mock_raw_initial
        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_final

        client = OpenAIClient(test_config)

        with patch("src.utils.llm_client.time.sleep"):
            with pytest.raises(ValueError, match="Model refused to respond"):
                client.create_response("Prompt", "Input")

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.time.sleep")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_incomplete_response_handling(
        self, mock_openai_class, mock_get_encoding, mock_sleep, mock_probe, test_config
    ):
        """Проверка обработки incomplete статуса - теперь без retry (сразу exception)"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Первая попытка - incomplete
        initial_response1 = Mock(id="resp_001", status="queued")
        mock_raw_initial1 = MagicMock()
        mock_raw_initial1.headers = {}
        mock_raw_initial1.parse.return_value = initial_response1

        incomplete_response = Mock(
            id="resp_001",
            status="incomplete",
            incomplete_details=Mock(reason="max_output_tokens"),
            usage=Mock(output_tokens=4096),
        )
        mock_raw_incomplete = MagicMock()
        mock_raw_incomplete.headers = {}
        mock_raw_incomplete.parse.return_value = incomplete_response

        # Настраиваем последовательность
        mock_client_instance.responses.with_raw_response.create.return_value = mock_raw_initial1
        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_incomplete

        client = OpenAIClient(test_config)

        # Теперь IncompleteResponseError должна быть выброшена сразу без retry
        with pytest.raises(IncompleteResponseError) as exc_info:
            client.create_response("Prompt", "Input")

        # Проверки
        assert "max_output_tokens" in str(exc_info.value)
        assert "4096 tokens" in str(exc_info.value)
        assert mock_client_instance.responses.with_raw_response.create.call_count == 1  # Без retry!

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_timeout_cancellation(self, mock_openai_class, mock_get_encoding, test_config):
        """Проверка отмены запроса при превышении timeout"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Initial response
        initial_response = Mock(id="resp_12345", status="queued")
        mock_raw_initial = MagicMock()
        mock_raw_initial.headers = {}
        mock_raw_initial.parse.return_value = initial_response

        # Queued response (навсегда)
        queued_response = Mock(id="resp_12345", status="queued")
        mock_raw_queued = MagicMock()
        mock_raw_queued.headers = {}
        mock_raw_queued.parse.return_value = queued_response

        # Настраиваем mock
        mock_client_instance.responses.with_raw_response.create.return_value = mock_raw_initial
        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_queued
        mock_client_instance.responses.cancel.return_value = None  # Mock cancel метод

        client = OpenAIClient(test_config)
        client.config["timeout"] = 0.1  # Очень короткий timeout для теста

        with patch("src.utils.llm_client.time.sleep"):
            with pytest.raises(TimeoutError, match="Response generation exceeded"):
                client.create_response("Prompt", "Input")

        # Проверка что была попытка отмены
        mock_client_instance.responses.cancel.assert_called_once_with("resp_12345")

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.time.sleep")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_failed_status_handling(
        self, mock_openai_class, mock_get_encoding, mock_sleep, mock_probe, test_config
    ):
        """Проверка обработки failed статуса с retry"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Первая попытка - failed
        initial_response1 = Mock(id="resp_001", status="queued")
        mock_raw_initial1 = MagicMock()
        mock_raw_initial1.headers = {}
        mock_raw_initial1.parse.return_value = initial_response1

        failed_response = Mock(id="resp_001", status="failed", error=Mock(message="Internal error"))
        mock_raw_failed = MagicMock()
        mock_raw_failed.headers = {}
        mock_raw_failed.parse.return_value = failed_response

        # Вторая попытка - успех
        initial_response2 = Mock(id="resp_002", status="queued")
        mock_raw_initial2 = MagicMock()
        mock_raw_initial2.headers = {}
        mock_raw_initial2.parse.return_value = initial_response2

        final_response = MockResponse("Success after retry")
        final_response.id = "resp_002"
        mock_raw_final = MagicMock()
        mock_raw_final.headers = {}
        mock_raw_final.parse.return_value = final_response

        # Настраиваем последовательность
        mock_client_instance.responses.with_raw_response.create.side_effect = [
            mock_raw_initial1,
            mock_raw_initial2,
        ]

        mock_client_instance.responses.with_raw_response.retrieve.side_effect = [
            mock_raw_failed,
            mock_raw_final,
        ]

        client = OpenAIClient(test_config)

        with patch("src.utils.llm_client.time.sleep"):
            response_text, _, _ = client.create_response("Prompt", "Input")

        # Проверки
        assert response_text == "Success after retry"
        assert mock_client_instance.responses.with_raw_response.create.call_count == 2


# Тесты для логирования
class TestLogging:
    """Тесты для проверки логирования"""

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.time.sleep")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_multiple_retries_with_logging(
        self, mock_openai_class, mock_get_encoding, mock_sleep, mock_probe, caplog
    ):
        """Проверка логирования при множественных retry"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Три ошибки подряд
        errors = []
        for i in range(3):
            error = openai.RateLimitError(f"Rate limit {i+1}", response=Mock(), body={})
            error.response.headers = {
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": f"{(i+1)*500}ms",
            }
            errors.append(error)

        # Успешный ответ
        initial_response = Mock(id="resp_12345", status="queued")
        mock_raw_initial = MagicMock()
        mock_raw_initial.headers = {"x-ratelimit-remaining-tokens": "90000"}
        mock_raw_initial.parse.return_value = initial_response

        final_response = MockResponse("Success after retries")
        mock_raw_final = MagicMock()
        mock_raw_final.headers = {"x-ratelimit-remaining-tokens": "90000"}
        mock_raw_final.parse.return_value = final_response

        # Настраиваем последовательность
        mock_client_instance.responses.with_raw_response.create.side_effect = [
            errors[0],
            errors[1],
            errors[2],
            mock_raw_initial,
        ]

        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_final

        caplog.set_level(logging.INFO)

        test_config = {
            "api_key": "test-key",
            "model": "gpt-4o",
            "is_reasoning": False,  # Required parameter
            "tpm_limit": 100000,
            "tpm_safety_margin": 0.15,
            "max_completion": 4096,
            "timeout": 30,
            "max_retries": 3,
        }
        client = OpenAIClient(test_config)

        # Настраиваем mock probe
        mock_probe.return_value = None

        response_text, response_id, usage = client.create_response("Test prompt", "Test input")

        # Проверяем результат
        assert response_text == "Success after retries"
        assert response_id == "resp_12345"

        # Проверяем вызовы sleep
        assert mock_sleep.call_count == 3
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [20, 40, 80]  # Exponential backoff

        # Проверяем логи
        log_messages = [record.message for record in caplog.records]
        assert any("RateLimitError, retry 1/3 in 20s" in msg for msg in log_messages)
        assert any("RateLimitError, retry 2/3 in 40s" in msg for msg in log_messages)
        assert any("RateLimitError, retry 3/3 in 80s" in msg for msg in log_messages)

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_max_retries_exceeded_logging(
        self, mock_openai_class, mock_get_encoding, test_config, caplog
    ):
        """Проверка логирования при исчерпании retry"""
        # Setup с max_retries = 1
        test_config["max_retries"] = 1

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Ошибка
        error = openai.RateLimitError("Rate limit", response=Mock(), body={})
        error.response.headers = {
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "60000ms",
        }

        mock_client_instance.responses.with_raw_response.create.side_effect = error

        caplog.set_level(logging.ERROR)

        client = OpenAIClient(test_config)

        with patch("src.utils.llm_client.time.sleep"):
            with pytest.raises(openai.RateLimitError):
                client.create_response("Prompt", "Input")

        # Проверяем финальный лог ошибки
        log_messages = [record.message for record in caplog.records]
        assert any("Error after all retries" in msg for msg in log_messages)


# Дополнительные тесты для улучшенной функциональности TPMBucket
class TestTPMBucketEnhanced:
    """Тесты для улучшенной функциональности TPMBucket"""

    def test_update_from_headers_with_ms_format(self, caplog):
        """Проверка обработки времени в формате ms с детальным логированием"""
        bucket = TPMBucket(120000)

        # Включаем debug логирование
        caplog.set_level(logging.DEBUG)

        headers = {
            "x-ratelimit-remaining-tokens": "95000",
            "x-ratelimit-reset-tokens": "820ms",  # 820 миллисекунд
        }

        with patch("src.utils.llm_client.time.time", return_value=1000.0):
            bucket.update_from_headers(headers)

        # Проверяем состояние
        assert bucket.remaining_tokens == 95000
        assert bucket.reset_time == 1000  # 1000.0 + 820/1000 = 1000.82, округлено до 1000

        # Проверяем логи
        log_messages = [record.message for record in caplog.records]
        assert any("TPM consumed: 25000 tokens" in msg for msg in log_messages)
        assert any("TPM reset in 0.8s" in msg for msg in log_messages)

    def test_update_from_headers_reimbursement(self, caplog):
        """Проверка логирования при восстановлении токенов"""
        bucket = TPMBucket(120000)
        bucket.remaining_tokens = 50000  # Устанавливаем низкое значение

        caplog.set_level(logging.DEBUG)

        headers = {"x-ratelimit-remaining-tokens": "100000"}  # Больше чем было

        bucket.update_from_headers(headers)

        # Проверяем логи
        log_messages = [record.message for record in caplog.records]
        assert any("TPM reimbursed: 50000 tokens" in msg for msg in log_messages)

    def test_wait_if_needed_detailed_logging(self, caplog):
        """Проверка детального логирования при ожидании"""
        bucket = TPMBucket(120000)
        bucket.remaining_tokens = 1000

        caplog.set_level(logging.DEBUG)

        # Случай когда токенов достаточно
        bucket.wait_if_needed(500, safety_margin=0.15)

        log_messages = [record.message for record in caplog.records]
        assert any("TPM check: need 575 tokens" in msg for msg in log_messages)
        assert any("TPM check passed: sufficient tokens available" in msg for msg in log_messages)


class TestContextAccumulation:
    """Tests for context token accumulation with previous_response_id"""

    @patch("src.utils.llm_client.time.sleep")
    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_context_accumulation_first_request(
        self, mock_openai_class, mock_get_encoding, mock_probe, mock_sleep, test_config
    ):
        """Test first request without previous_response_id"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(1000))  # 1000 tokens
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Setup mock_probe to not actually call API
        mock_probe.return_value = None

        client = OpenAIClient(test_config)
        client.tpm_bucket.remaining_tokens = 100000

        # Mock successful response
        mock_response = MagicMock()
        mock_response.id = "resp_123"
        mock_response.status = "completed"
        mock_response.output = [{"message": {"content": "Test response"}}]
        mock_response.usage = {"input_tokens": 1005, "output_tokens": 50, "total_tokens": 1055}

        # Mock the create response
        mock_create_response = MagicMock()
        mock_create_response.id = "resp_123"
        mock_client_instance.responses.create.return_value = mock_create_response

        # Mock the retrieve chain for status checks
        mock_raw_response = MagicMock()
        mock_parsed_response = MagicMock()
        mock_parsed_response.status = "completed"
        mock_parsed_response.id = "resp_123"

        # Create proper output structure
        mock_message_output = MagicMock()
        mock_message_output.type = "message"
        mock_content_item = MagicMock()
        mock_content_item.type = "output_text"
        mock_content_item.text = "Test response"
        mock_message_output.content = [mock_content_item]

        mock_parsed_response.output = [mock_message_output]

        # Create proper usage structure
        mock_usage = MagicMock()
        mock_usage.input_tokens = 1005
        mock_usage.output_tokens = 50
        mock_usage.total_tokens = 1055
        mock_usage.output_tokens_details = None
        mock_parsed_response.usage = mock_usage
        mock_raw_response.parse.return_value = mock_parsed_response
        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_response

        # Act
        response, response_id, usage = client.create_response("You are helpful", "What is 2+2?")

        # Assert
        assert client.last_usage is not None
        assert client.last_usage.input_tokens == 1005
        # Without previous_response_id, should encode full prompt
        assert mock_encoder.encode.called

    @patch("src.utils.llm_client.time.sleep")
    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_context_accumulation_with_previous_response(
        self, mock_openai_class, mock_get_encoding, mock_probe, mock_sleep, test_config
    ):
        """Test request with previous_response_id accumulates context"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(500))  # 500 new tokens
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Setup mock_probe
        mock_probe.return_value = None

        # Add max_context_tokens to config
        test_config["max_context_tokens"] = 128000

        client = OpenAIClient(test_config)
        client.tpm_bucket.remaining_tokens = 200000

        # Simulate previous usage
        from src.utils.llm_client import ResponseUsage

        client.last_usage = ResponseUsage(
            input_tokens=50000,  # Previous context
            output_tokens=1000,
            total_tokens=51000,
            reasoning_tokens=0,
        )

        # Mock response
        mock_create_response = MagicMock()
        mock_create_response.id = "resp_456"
        mock_client_instance.responses.create.return_value = mock_create_response

        # Mock the retrieve chain
        mock_raw_response = MagicMock()
        mock_parsed_response = MagicMock()
        mock_parsed_response.status = "completed"
        mock_parsed_response.id = "resp_456"

        # Create proper output structure
        mock_message_output = MagicMock()
        mock_message_output.type = "message"
        mock_content_item = MagicMock()
        mock_content_item.type = "output_text"
        mock_content_item.text = "Another response"
        mock_message_output.content = [mock_content_item]

        mock_parsed_response.output = [mock_message_output]

        # Create proper usage structure
        mock_usage = MagicMock()
        mock_usage.input_tokens = 50500  # Old context + new
        mock_usage.output_tokens = 100
        mock_usage.total_tokens = 50600
        mock_usage.output_tokens_details = None
        mock_parsed_response.usage = mock_usage
        mock_raw_response.parse.return_value = mock_parsed_response
        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_response

        # Act
        response, response_id, usage = client.create_response(
            "Continue", "What about 3+3?", previous_response_id="resp_123"
        )

        # Assert
        # Should use accumulated context (50000 + 500 = 50500)
        assert usage.input_tokens == 50500

    @patch("src.utils.llm_client.time.sleep")
    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_context_limit_warning(
        self, mock_openai_class, mock_get_encoding, mock_probe, mock_sleep, test_config, caplog
    ):
        """Test warning when approaching context limit"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(10000))  # 10K new tokens
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Setup mock_probe
        mock_probe.return_value = None

        # Set lower context limit
        test_config["max_context_tokens"] = 130000

        client = OpenAIClient(test_config)
        client.tpm_bucket.remaining_tokens = 200000

        # Simulate large previous context
        from src.utils.llm_client import ResponseUsage

        client.last_usage = ResponseUsage(
            input_tokens=125000,  # Close to limit
            output_tokens=1000,
            total_tokens=126000,
            reasoning_tokens=0,
        )

        # Mock response
        mock_create_response = MagicMock()
        mock_create_response.id = "resp_789"
        mock_client_instance.responses.create.return_value = mock_create_response

        # Mock the retrieve chain
        mock_raw_response = MagicMock()
        mock_parsed_response = MagicMock()
        mock_parsed_response.status = "completed"
        mock_parsed_response.id = "resp_789"

        # Create proper output structure
        mock_message_output = MagicMock()
        mock_message_output.type = "message"
        mock_content_item = MagicMock()
        mock_content_item.type = "output_text"
        mock_content_item.text = "Response"
        mock_message_output.content = [mock_content_item]

        mock_parsed_response.output = [mock_message_output]

        # Create proper usage structure
        mock_usage = MagicMock()
        mock_usage.input_tokens = 130000  # Capped at limit
        mock_usage.output_tokens = 100
        mock_usage.total_tokens = 130100
        mock_usage.output_tokens_details = None
        mock_parsed_response.usage = mock_usage
        mock_raw_response.parse.return_value = mock_parsed_response
        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_response

        caplog.set_level(logging.WARNING)

        # Act
        response, response_id, usage = client.create_response(
            "Continue", "Long content here...", previous_response_id="resp_456"
        )

        # Assert
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("Context approaching limit" in msg for msg in warning_messages)
        assert any("135000 tokens" in msg for msg in warning_messages)  # 125K + 10K
        assert any("max: 130000" in msg for msg in warning_messages)

    @patch("src.utils.llm_client.time.sleep")
    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_different_max_context_tokens(
        self, mock_openai_class, mock_get_encoding, mock_probe, mock_sleep
    ):
        """Test with different max_context_tokens values"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(1000))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Setup mock_probe
        mock_probe.return_value = None

        # Test with 200K limit for o1/o4 models
        config_200k = {
            "api_key": "test-key",
            "model": "o4-mini",
            "is_reasoning": True,  # Required parameter
            "tpm_limit": 150000,
            "max_completion": 90000,
            "max_context_tokens": 200000,  # 200K for o1/o4
            "timeout": 360,
            "max_retries": 3,
        }

        client = OpenAIClient(config_200k)
        assert client.config["max_context_tokens"] == 200000

        # Test with 128K limit for gpt-4 models
        config_128k = {
            "api_key": "test-key",
            "model": "gpt-4",
            "is_reasoning": False,  # Required parameter
            "tpm_limit": 100000,
            "max_completion": 50000,
            "max_context_tokens": 128000,  # 128K for gpt-4
            "timeout": 300,
            "max_retries": 3,
        }

        client2 = OpenAIClient(config_128k)
        assert client2.config["max_context_tokens"] == 128000


# New tests for is_reasoning parameter
class TestIsReasoningParameter:
    """Tests for explicit is_reasoning parameter"""

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_is_reasoning_required(self, mock_openai_class, mock_get_encoding):
        """Test that is_reasoning parameter is required."""
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder

        config = {
            "api_key": "sk-test",
            "model": "gpt-5-mini",
            # is_reasoning отсутствует намеренно
            "tpm_limit": 100000,
            "max_completion": 1000,
        }
        with pytest.raises(ValueError, match="is_reasoning.*required"):
            OpenAIClient(config)

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_null_parameters_not_sent(self, mock_openai_class, mock_get_encoding, test_config):
        """Test that null parameters are not included in API request."""
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder

        config = {
            "api_key": "sk-test",
            "model": "gpt-5-mini",
            "is_reasoning": True,
            "temperature": None,  # В Python после загрузки из TOML
            "reasoning_effort": "medium",
            "reasoning_summary": None,
            "verbosity": None,
            "tpm_limit": 100000,
            "max_completion": 1000,
        }
        client = OpenAIClient(config)
        params = client._prepare_request_params("test", "test")

        assert "temperature" not in params
        assert "verbosity" not in params  # verbosity на верхнем уровне
        assert "reasoning" in params
        assert params["reasoning"]["effort"] == "medium"
        assert "summary" not in params["reasoning"]

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_reasoning_model_extraction(self, mock_openai_class, mock_get_encoding):
        """Test that reasoning models extract content from correct output index."""
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder

        config = {
            "api_key": "sk-test",
            "model": "o4-mini",
            "is_reasoning": True,
            "tpm_limit": 100000,
            "max_completion": 1000,
        }
        client = OpenAIClient(config)

        # Mock response with reasoning structure
        mock_response = Mock()
        mock_response.status = "completed"

        # Create reasoning output
        mock_reasoning_output = Mock()
        mock_reasoning_output.type = "reasoning"

        # Create message output
        mock_message_output = Mock()
        mock_message_output.type = "message"
        mock_content_item = Mock()
        mock_content_item.type = "output_text"
        mock_content_item.text = "Test response"
        mock_message_output.content = [mock_content_item]

        mock_response.output = [
            mock_reasoning_output,  # output[0] for reasoning models
            mock_message_output,  # output[1] for reasoning models
        ]

        result = client._extract_response_content(mock_response)
        assert result == "Test response"

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_non_reasoning_model_extraction(self, mock_openai_class, mock_get_encoding):
        """Test that non-reasoning models extract content from correct output index."""
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder

        config = {
            "api_key": "sk-test",
            "model": "gpt-4.1-mini",
            "is_reasoning": False,
            "tpm_limit": 100000,
            "max_completion": 1000,
        }
        client = OpenAIClient(config)

        # Mock response with regular structure
        mock_response = Mock()
        mock_response.status = "completed"

        # Create message output
        mock_message_output = Mock()
        mock_message_output.type = "message"
        mock_content_item = Mock()
        mock_content_item.type = "output_text"
        mock_content_item.text = "Test response"
        mock_message_output.content = [mock_content_item]

        mock_response.output = [mock_message_output]  # output[0] for regular models

        result = client._extract_response_content(mock_response)
        assert result == "Test response"

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_verbosity_parameter_handling(self, mock_openai_class, mock_get_encoding):
        """Test that verbosity parameter is handled correctly."""
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder

        # Test with verbosity set
        config_with_verbosity = {
            "api_key": "sk-test",
            "model": "gpt-5",
            "is_reasoning": False,
            "verbosity": "high",
            "tpm_limit": 100000,
            "max_completion": 1000,
        }
        client = OpenAIClient(config_with_verbosity)
        params = client._prepare_request_params("test", "test")

        assert "verbosity" in params
        assert params["verbosity"] == "high"

        # Test with verbosity as None
        config_without_verbosity = {
            "api_key": "sk-test",
            "model": "gpt-5",
            "is_reasoning": False,
            "verbosity": None,
            "tpm_limit": 100000,
            "max_completion": 1000,
        }
        client2 = OpenAIClient(config_without_verbosity)
        params2 = client2._prepare_request_params("test", "test")

        assert "verbosity" not in params2


class TestConfigHasTestParameters:
    """Тесты проверки корректности test параметров в конфигурации"""

    def test_config_has_test_parameters(self):
        """Проверка что конфиг содержит корректные тестовые параметры"""
        from src.utils.config import load_config

        config = load_config()

        # Проверка для itext2kg_concepts
        assert "model_test" in config["itext2kg_concepts"], "itext2kg_concepts должен содержать model_test"
        assert (
            "max_context_tokens_test" in config["itext2kg_concepts"]
        ), "itext2kg_concepts должен содержать max_context_tokens_test"
        assert "tpm_limit_test" in config["itext2kg_concepts"], "itext2kg_concepts должен содержать tpm_limit_test"
        assert (
            "max_completion_test" in config["itext2kg_concepts"]
        ), "itext2kg_concepts должен содержать max_completion_test"

        # Проверка для itext2kg_graph
        assert "model_test" in config["itext2kg_graph"], "itext2kg_graph должен содержать model_test"
        assert (
            "max_context_tokens_test" in config["itext2kg_graph"]
        ), "itext2kg_graph должен содержать max_context_tokens_test"
        assert "tpm_limit_test" in config["itext2kg_graph"], "itext2kg_graph должен содержать tpm_limit_test"
        assert (
            "max_completion_test" in config["itext2kg_graph"]
        ), "itext2kg_graph должен содержать max_completion_test"

        # Проверка для refiner
        assert "model_test" in config["refiner"], "refiner должен содержать model_test"
        assert (
            "max_context_tokens_test" in config["refiner"]
        ), "refiner должен содержать max_context_tokens_test"
        assert "tpm_limit_test" in config["refiner"], "refiner должен содержать tpm_limit_test"
        assert (
            "max_completion_test" in config["refiner"]
        ), "refiner должен содержать max_completion_test"

        # Проверка что test параметры отличаются от основных (обычно меньше)
        assert (
            config["itext2kg_concepts"]["tpm_limit_test"] <= config["itext2kg_concepts"]["tpm_limit"]
        ), "tpm_limit_test должен быть меньше или равен tpm_limit"
        assert (
            config["itext2kg_concepts"]["max_completion_test"] <= config["itext2kg_concepts"]["max_completion"]
        ), "max_completion_test должен быть меньше или равен max_completion"

        assert (
            config["refiner"]["tpm_limit_test"] <= config["refiner"]["tpm_limit"]
        ), "tpm_limit_test должен быть меньше или равен tpm_limit"
        assert (
            config["refiner"]["max_completion_test"] <= config["refiner"]["max_completion"]
        ), "max_completion_test должен быть меньше или равен max_completion"


# Новые тесты для управления response chain
class TestResponseChainManagement:
    """Тесты для управления цепочкой ответов с configurable depth"""

    @pytest.fixture
    def config_with_chain_depth(self):
        """Конфигурация с response_chain_depth"""
        return {
            "api_key": "test-key-123",
            "model": "gpt-4o",
            "is_reasoning": False,
            "tpm_limit": 120000,
            "tpm_safety_margin": 0.15,
            "max_completion": 4096,
            "timeout": 45,
            "max_retries": 3,
            "response_chain_depth": 2,  # Sliding window of 2
            "truncation": "auto",
        }

    @pytest.fixture
    def config_independent(self):
        """Конфигурация для независимых запросов"""
        return {
            "api_key": "test-key-123",
            "model": "gpt-4o",
            "is_reasoning": False,
            "tpm_limit": 120000,
            "tpm_safety_margin": 0.15,
            "max_completion": 4096,
            "timeout": 45,
            "max_retries": 3,
            "response_chain_depth": 0,  # Independent requests
        }

    @pytest.fixture
    def config_unlimited(self):
        """Конфигурация без ограничения цепочки"""
        return {
            "api_key": "test-key-123",
            "model": "gpt-4o",
            "is_reasoning": False,
            "tpm_limit": 120000,
            "tpm_safety_margin": 0.15,
            "max_completion": 4096,
            "timeout": 45,
            "max_retries": 3,
            # response_chain_depth not set - unlimited chain
        }

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_response_chain_depth_none(
        self, mock_openai_class, mock_get_encoding, config_unlimited
    ):
        """Проверка неограниченной цепочки (текущее поведение)"""
        # Setup
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        client = OpenAIClient(config_unlimited)

        # Проверяем, что цепочка не создается
        assert client.response_chain_depth is None
        assert client.response_chain is None
        # В неограниченном режиме response_chain не создается
        assert not hasattr(client, "response_chain") or client.response_chain is None

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_response_chain_depth_zero(
        self, mock_openai_class, mock_get_encoding, config_independent
    ):
        """Проверка независимых запросов"""
        # Setup
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        client = OpenAIClient(config_independent)

        # Проверяем конфигурацию независимых запросов
        assert client.response_chain_depth == 0
        assert client.response_chain is None

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_response_chain_depth_limited(
        self, mock_openai_class, mock_get_encoding, config_with_chain_depth
    ):
        """Проверка скользящего окна"""
        # Setup
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        client = OpenAIClient(config_with_chain_depth)

        # Проверяем создание deque
        assert client.response_chain_depth == 2
        assert client.response_chain is not None
        assert len(client.response_chain) == 0

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_response_deletion_success(
        self, mock_openai_class, mock_get_encoding, config_with_chain_depth
    ):
        """Проверка успешного удаления старых ответов"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Mock delete response
        mock_delete_result = Mock(deleted=True)
        mock_client_instance.responses.delete.return_value = mock_delete_result

        client = OpenAIClient(config_with_chain_depth)

        # Test deletion
        client._delete_response("resp_old_123")

        # Verify delete was called
        mock_client_instance.responses.delete.assert_called_once_with("resp_old_123")

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_response_deletion_failure(
        self, mock_openai_class, mock_get_encoding, config_with_chain_depth
    ):
        """Проверка обработки ошибки удаления"""
        # Setup
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Mock failed delete
        mock_client_instance.responses.delete.side_effect = Exception("API Error")

        client = OpenAIClient(config_with_chain_depth)

        # Test deletion failure
        with pytest.raises(ValueError, match="Failed to delete response"):
            client._delete_response("resp_old_123")

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_truncation_parameter(
        self, mock_openai_class, mock_get_encoding, config_with_chain_depth
    ):
        """Проверка передачи параметра truncation в API"""
        # Setup
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        client = OpenAIClient(config_with_chain_depth)

        # Test truncation parameter
        params = client._prepare_request_params("instructions", "input", "prev_id")

        assert "truncation" in params
        assert params["truncation"] == "auto"

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_incomplete_no_retry(
        self, mock_openai_class, mock_get_encoding, mock_probe, test_config
    ):
        """Проверка отсутствия retry при IncompleteResponseError"""
        # Setup
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        # Mock probe to avoid extra API call
        mock_probe.return_value = None

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Initial response
        initial_response = Mock(id="resp_12345", status="queued")
        mock_raw_initial = MagicMock()
        mock_raw_initial.headers = {}
        mock_raw_initial.parse.return_value = initial_response

        # Incomplete response
        mock_response = Mock(
            output=[],
            status="incomplete",
            id="resp_12345",
            incomplete_details=Mock(reason="max_output_tokens"),
            usage=Mock(output_tokens=1000),
        )

        mock_raw_final = MagicMock()
        mock_raw_final.headers = {}
        mock_raw_final.parse.return_value = mock_response

        # Configure mocks
        mock_client_instance.responses.with_raw_response.create.return_value = mock_raw_initial
        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_final

        client = OpenAIClient(test_config)

        with patch("src.utils.llm_client.time.sleep"):
            with pytest.raises(IncompleteResponseError) as exc_info:
                client.create_response("Prompt", "Input")

            # Verify no retry was attempted (create called only once after probe)
            assert mock_client_instance.responses.with_raw_response.create.call_count == 1
            assert "max_output_tokens" in str(exc_info.value)

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_confirm_response_workflow(
        self, mock_openai_class, mock_get_encoding, config_with_chain_depth
    ):
        """Test two-phase confirmation workflow"""
        # Setup encoder mock
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        # Setup OpenAI client mocks
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        # Create response using MockResponse helper
        from tests.test_llm_client import MockResponse

        # Initial response
        initial_response = Mock(id="test_response_id", status="queued")
        mock_raw_initial = MagicMock()
        mock_raw_initial.headers = {}
        mock_raw_initial.parse.return_value = initial_response

        # Final response
        final_response = MockResponse("test response")
        final_response.id = "test_response_id"
        mock_raw_final = MagicMock()
        mock_raw_final.headers = {}
        mock_raw_final.parse.return_value = final_response

        # Configure mocks
        mock_client_instance.responses.with_raw_response.create.return_value = mock_raw_initial
        mock_client_instance.responses.with_raw_response.retrieve.return_value = mock_raw_final

        client = OpenAIClient(config_with_chain_depth)

        # Create response - should not update chain yet
        with patch("src.utils.llm_client.time.sleep"):
            response_text, response_id, usage = client.create_response("test", "data")

        assert client.unconfirmed_response_id == "test_response_id"
        assert client.last_confirmed_response_id is None
        if client.response_chain is not None:
            assert len(client.response_chain) == 0

        # Confirm - should update chain
        client.confirm_response()
        assert client.unconfirmed_response_id is None
        assert client.last_confirmed_response_id == "test_response_id"
        if client.response_chain is not None:
            assert len(client.response_chain) == 1

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_repair_uses_confirmed_id(
        self, mock_openai_class, mock_get_encoding, mock_probe, config_with_chain_depth
    ):
        """Test that repair uses last confirmed ID, not unconfirmed"""
        # Setup encoder mock
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        # Setup OpenAI client mocks
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        from tests.test_llm_client import MockResponse

        # First response
        initial_response1 = Mock(id="resp_001", status="queued")
        mock_raw_initial1 = MagicMock()
        mock_raw_initial1.headers = {}
        mock_raw_initial1.parse.return_value = initial_response1

        final_response1 = MockResponse("First response")
        final_response1.id = "resp_001"
        mock_raw_final1 = MagicMock()
        mock_raw_final1.headers = {}
        mock_raw_final1.parse.return_value = final_response1

        # Second response (unconfirmed)
        initial_response2 = Mock(id="resp_002", status="queued")
        mock_raw_initial2 = MagicMock()
        mock_raw_initial2.headers = {}
        mock_raw_initial2.parse.return_value = initial_response2

        final_response2 = MockResponse("Second response")
        final_response2.id = "resp_002"
        mock_raw_final2 = MagicMock()
        mock_raw_final2.headers = {}
        mock_raw_final2.parse.return_value = final_response2

        # Repair response
        initial_response3 = Mock(id="resp_repair", status="queued")
        mock_raw_initial3 = MagicMock()
        mock_raw_initial3.headers = {}
        mock_raw_initial3.parse.return_value = initial_response3

        final_response3 = MockResponse("Repaired response")
        final_response3.id = "resp_repair"
        mock_raw_final3 = MagicMock()
        mock_raw_final3.headers = {}
        mock_raw_final3.parse.return_value = final_response3

        # Configure mocks
        mock_client_instance.responses.with_raw_response.create.side_effect = [
            mock_raw_initial1,
            mock_raw_initial2,
            mock_raw_initial3,
        ]

        mock_client_instance.responses.with_raw_response.retrieve.side_effect = [
            mock_raw_final1,
            mock_raw_final2,
            mock_raw_final3,
        ]

        client = OpenAIClient(config_with_chain_depth)

        # Create and confirm first response
        with patch("src.utils.llm_client.time.sleep"):
            client.create_response("test", "data1")
            client.confirm_response()

            # Create second response but don't confirm
            client.create_response("test", "data2", previous_response_id="resp_001")

            # Repair should use confirmed ID, not unconfirmed
            client.repair_response("repair", "data")

        # Check that repair used the confirmed ID
        # Call index 2 (0=first, 1=second, 2=repair)
        repair_call = mock_client_instance.responses.with_raw_response.create.call_args_list[2]
        assert repair_call[1]["previous_response_id"] == "resp_001"

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_auto_discard_unconfirmed(
        self, mock_openai_class, mock_get_encoding, mock_probe, config_with_chain_depth, caplog
    ):
        """Test automatic discard of unconfirmed response on next create"""
        # Setup encoder mock
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        # Setup OpenAI client mocks
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        from tests.test_llm_client import MockResponse

        # Two responses
        responses = []
        for i in range(2):
            initial_response = Mock(id=f"resp_{i:03d}", status="queued")
            mock_raw_initial = MagicMock()
            mock_raw_initial.headers = {}
            mock_raw_initial.parse.return_value = initial_response

            final_response = MockResponse(f"Response {i}")
            final_response.id = f"resp_{i:03d}"
            mock_raw_final = MagicMock()
            mock_raw_final.headers = {}
            mock_raw_final.parse.return_value = final_response

            responses.append((mock_raw_initial, mock_raw_final))

        # Configure mocks
        mock_client_instance.responses.with_raw_response.create.side_effect = [
            responses[0][0],
            responses[1][0],
        ]

        mock_client_instance.responses.with_raw_response.retrieve.side_effect = [
            responses[0][1],
            responses[1][1],
        ]

        client = OpenAIClient(config_with_chain_depth)

        # Create but don't confirm
        import logging

        caplog.set_level(logging.WARNING)

        with patch("src.utils.llm_client.time.sleep"):
            client.create_response("test", "data1")
            unconfirmed = client.unconfirmed_response_id

            # Create another - should discard previous unconfirmed
            client.create_response("test", "data2")

        # Check that discard warning was logged
        assert any("was never confirmed, discarding" in msg for msg in caplog.messages)

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_validation_failure_scenario(
        self, mock_openai_class, mock_get_encoding, mock_probe, test_config
    ):
        """Test the exact scenario that caused the production bug"""
        # Setup encoder mock
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_get_encoding.return_value = mock_encoder

        # Setup OpenAI client mocks
        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        from tests.test_llm_client import MockResponse

        config = test_config.copy()
        config["response_chain_depth"] = 1  # Critical: depth = 1

        # Three responses: first (confirmed), second (unconfirmed), repair
        responses = []
        for i in range(3):
            initial_response = Mock(id=f"resp_{i:03d}", status="queued")
            mock_raw_initial = MagicMock()
            mock_raw_initial.headers = {}
            mock_raw_initial.parse.return_value = initial_response

            final_response = MockResponse(f"Response {i}")
            final_response.id = f"resp_{i:03d}"
            mock_raw_final = MagicMock()
            mock_raw_final.headers = {}
            mock_raw_final.parse.return_value = final_response

            responses.append((mock_raw_initial, mock_raw_final))

        # Configure mocks
        mock_client_instance.responses.with_raw_response.create.side_effect = [
            responses[0][0],
            responses[1][0],
            responses[2][0],
        ]

        mock_client_instance.responses.with_raw_response.retrieve.side_effect = [
            responses[0][1],
            responses[1][1],
            responses[2][1],
        ]

        client = OpenAIClient(config)

        # Simulate the exact production scenario
        with patch("src.utils.llm_client.time.sleep"):
            # First successful response (slice_007)
            text1, id1, _ = client.create_response("test", "data1")
            client.confirm_response()  # Simulates successful validation

            # Second response (slice_008) - simulates validation failure
            text2, id2, _ = client.create_response("test2", "data2")
            # DON'T confirm - simulates validation failure

            # With the fix, repair should work without "Previous response not found" error
            repair_text, repair_id, _ = client.repair_response("repair", "data")

        # Verify repair used the confirmed ID (resp_000), not the unconfirmed one
        # Call index 2 (0=first, 1=second, 2=repair)
        repair_call = mock_client_instance.responses.with_raw_response.create.call_args_list[2]
        assert repair_call[1]["previous_response_id"] == "resp_000"


class TestErrorPaths:
    """Тесты для непокрытых error paths в llm_client.py"""

    def test_delete_response_error_handling(self):
        """Тест обработки ошибок при удалении ответа (покрытие строк в delete_response)."""
        config = {
            "api_key": "test-key",
            "model": "gpt-4o",
            "is_reasoning": False,
            "tpm_limit": 60000,
            "max_completion": 2048,
            "response_chain_depth": 10,
        }

        with patch("src.utils.llm_client.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Симулируем ошибку API при удалении (не 404)
            mock_client.responses.delete.side_effect = Exception("API Error during delete")

            client = OpenAIClient(config)

            # Метод _delete_response должен поднять ValueError для не-404 ошибок
            with pytest.raises(ValueError) as exc_info:
                client._delete_response("resp_to_delete")
            
            assert "Failed to delete response" in str(exc_info.value)

            # Проверяем, что delete был вызван
            mock_client.responses.delete.assert_called_once_with("resp_to_delete")
            
            # Теперь проверим, что 404 ошибки обрабатываются молча
            mock_client.responses.delete.side_effect = Exception("404 Not Found")
            # Не должно быть исключения для 404
            client._delete_response("resp_not_found")

    def test_tpm_bucket_edge_cases(self):
        """Тест граничных случаев TPM bucket."""
        bucket = TPMBucket(60000)

        # Тест с нулевыми токенами
        bucket.wait_if_needed(0)
        assert bucket.remaining_tokens == 60000  # Изначально полный

        # Тест обновления из заголовков и ожидания
        with patch("time.time") as mock_time:
            # Устанавливаем текущее время
            mock_time.return_value = 1000.0
            
            # Обновляем состояние из заголовков
            headers = {
                "x-ratelimit-remaining-tokens": "30000",
                "x-ratelimit-reset-tokens": "1000ms"
            }
            bucket.update_from_headers(headers)
            assert bucket.remaining_tokens == 30000
            # reset_time должен быть 1001 (текущее время 1000 + 1 секунда)
            assert bucket.reset_time == 1001.0
        
            # Тест с достаточным количеством токенов
            with patch("time.sleep") as mock_sleep:
                # 25000 * 1.15 = 28750, меньше чем 30000
                bucket.wait_if_needed(25000, safety_margin=0.15)
                # Не должен спать, так как хватает токенов
                mock_sleep.assert_not_called()
            
            # Тест с недостатком токенов
            with patch("time.sleep") as mock_sleep:
                # 30000 * 1.15 = 34500, больше чем 30000
                bucket.wait_if_needed(30000, safety_margin=0.15)
                # Должен подождать до reset_time (1001 - 1000 = 1 секунда)
                mock_sleep.assert_called_once()
                # Проверяем, что ожидание примерно 1.1 секунды (1 + 0.1 buffer)
                wait_time = mock_sleep.call_args[0][0]
                assert 1.0 <= wait_time <= 1.2

    @pytest.mark.skip(reason="Requires async polling mock sequence. Covered by integration tests")
    def test_repair_response_with_network_error(self):
        """Тест repair_response с сетевой ошибкой и retry."""
        config = {
            "api_key": "test-key",
            "model": "gpt-4o",
            "is_reasoning": False,
            "tpm_limit": 60000,
            "max_completion": 2048,
            "max_retries": 2,
            "timeout": 600,
            "prompt_caching": False,
        }

        with patch("src.utils.llm_client.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock для probe запросов  
            probe_response = Mock()
            probe_response.parse.return_value = Mock(id="probe_id")
            probe_response.headers = {
                "x-ratelimit-limit-tokens": "60000",
                "x-ratelimit-remaining-tokens": "50000",
                "x-ratelimit-reset-tokens": "1000ms"
            }
            
            # Первый успешный ответ
            first_response = MockResponse(text="response")
            first_response.id = "resp_001"
            
            # Repair ответ после retry
            repair_response = MockResponse(text="repaired")
            repair_response.id = "resp_002"
            
            # Создаем raw response обертки
            first_raw = Mock()
            first_raw.parse.return_value = first_response
            first_raw.headers = {}
            
            repair_raw = Mock()
            repair_raw.parse.return_value = repair_response
            repair_raw.headers = {}
            
            # Настраиваем side_effect для create
            create_calls = [
                probe_response,  # Первый probe при инициализации
                first_raw,  # Первый успешный create
                Exception("Network error"),  # Ошибка при repair
                repair_raw,  # Успешный retry после ошибки
            ]
            mock_client.responses.with_raw_response.create.side_effect = create_calls
            
            # Мокаем retrieve для polling
            mock_client.responses.with_raw_response.retrieve.side_effect = [
                first_raw,  # Polling для первого ответа
                repair_raw,  # Polling для repair ответа
            ]
            
            # Mock delete
            mock_client.responses.delete.return_value = None

            client = OpenAIClient(config)

            # Создаем первый ответ
            text1, id1, _ = client.create_response("test", "data")
            assert id1 == "resp_001"
            assert text1 == "Test response"  # MockResponse default text
            client.confirm_response()

            # Пытаемся repair с retry
            with patch("time.sleep"):  # Skip sleep during test
                text2, id2, _ = client.repair_response("repair", "data")

            assert id2 == "resp_002"
            assert text2 == "Test response"  # MockResponse default text

    # DELETED: test_create_response_with_rate_limit_error 
    # This test duplicates functionality already covered by test_rate_limit_retry_with_headers
    # Rate limit retry logic is fully tested in the existing test

    @pytest.mark.skip(reason="Requires async polling mock sequence. Covered by test_chain_management_with_confirmations")
    def test_response_chain_management_edge_cases(self):
        """Тест граничных случаев управления цепочкой ответов."""
        config = {
            "api_key": "test-key",
            "model": "gpt-4o",
            "is_reasoning": False,
            "tpm_limit": 60000,
            "max_completion": 2048,
            "response_chain_depth": 2,  # Очень короткая цепочка
            "max_retries": 2,
            "timeout": 600,
            "prompt_caching": False,
        }

        with patch("src.utils.llm_client.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock для probe запроса
            probe_response = Mock()
            probe_response.parse.return_value = Mock(id="probe_id")
            probe_response.headers = {
                "x-ratelimit-limit-tokens": "60000",
                "x-ratelimit-remaining-tokens": "50000",
                "x-ratelimit-reset-tokens": "1000ms"
            }
            
            # Мокаем ответы
            responses = []
            raw_responses = []
            for i in range(5):
                resp = MockResponse(text=f"response {i}")
                resp.id = f"resp_{i:03d}"
                raw = Mock()
                raw.parse.return_value = resp
                raw.headers = {}
                responses.append(resp)
                raw_responses.append(raw)

            # Настраиваем side_effect для create
            create_side_effect = [probe_response]  # Первый probe
            create_side_effect.extend(raw_responses[:4])  # 4 ответа
            mock_client.responses.with_raw_response.create.side_effect = create_side_effect
            
            # Мокаем retrieve для polling
            mock_client.responses.with_raw_response.retrieve.side_effect = raw_responses[:4]
            
            # Mock delete
            mock_client.responses.delete.return_value = None

            client = OpenAIClient(config)

            # Создаем цепочку ответов длиннее чем response_chain_depth
            for i in range(4):
                text, resp_id, _ = client.create_response(f"test {i}", f"data {i}")
                client.confirm_response()

            # Проверяем, что старые ответы были удалены
            # При depth=2 и 4 подтвержденных ответах, должны остаться только 2 последних
            assert len(client.response_chain) == 2
            assert "resp_002" in client.response_chain
            assert "resp_003" in client.response_chain
            
            # Проверяем что delete был вызван для старых ответов
            delete_calls = [call.responses.delete(id) for id in ["resp_000", "resp_001"]]
            for call_args in delete_calls:
                assert call_args in mock_client.mock_calls
    
    def test_update_tpm_probe_failure_with_fallback(self):
        """Test that probe failure triggers fallback chain"""
        config = {
            "api_key": "test-key",
            "model": "gpt-4o",
            "is_reasoning": False,
            "tpm_limit": 60000,
            "max_completion": 2048,
            "max_retries": 2,
            "timeout": 600,
            "prompt_caching": False,
            "probe_model": "non-existent-model"  # Custom probe model
        }
        
        with patch("src.utils.llm_client.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # First call fails (non-existent model), second succeeds (fallback)
            def mock_create(**kwargs):
                if kwargs.get("model") == "non-existent-model":
                    raise Exception("Model not found: non-existent-model")
                else:
                    # Return successful mock for fallback model
                    mock_resp = Mock()
                    mock_resp.parse.return_value = Mock(id="probe_success")
                    mock_resp.headers = {
                        "x-ratelimit-limit-tokens": "60000",
                        "x-ratelimit-remaining-tokens": "50000",
                        "x-ratelimit-reset-tokens": "1000ms"
                    }
                    return mock_resp
            
            mock_client.responses.with_raw_response.create = Mock(side_effect=mock_create)
            mock_client.responses.delete.return_value = None
            
            # Initialize client - should trigger probe with fallback
            client = OpenAIClient(config)
            
            # Call probe manually to test fallback
            client._update_tpm_via_probe()
            
            # Verify fallback was used (after manual call)
            assert client.probe_model != "non-existent-model"
            # At least 2 calls: initial probe + manual probe with fallback
            assert mock_client.responses.with_raw_response.create.call_count >= 2
    
    def test_update_tpm_all_probes_fail_uses_cache(self):
        """Test that cached limits are used when all probe models fail"""
        config = {
            "api_key": "test-key",
            "model": "gpt-4o",
            "is_reasoning": False,
            "tpm_limit": 60000,
            "max_completion": 2048,
            "max_retries": 2,
            "timeout": 600,
            "prompt_caching": False,
        }
        
        with patch("src.utils.llm_client.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # All probe attempts fail
            mock_client.responses.with_raw_response.create = Mock(
                side_effect=Exception("Network error")
            )
            
            # Initialize client - should use cached values
            client = OpenAIClient(config)
            
            # Set cached values
            client._cached_tpm_limit = 50000
            client._cached_tpm_remaining = 25000
            
            # Call update_tpm_via_probe manually
            client._update_tpm_via_probe()
            
            # Verify cached values were used
            assert client.tpm_bucket.initial_limit == 50000
            assert client.tpm_bucket.remaining_tokens == 25000
    
    def test_probe_model_configuration(self):
        """Test that probe_model is correctly configured from config"""
        config = {
            "api_key": "test-key",
            "model": "gpt-4o",
            "is_reasoning": False,
            "tpm_limit": 60000,
            "max_completion": 2048,
            "max_retries": 2,
            "timeout": 600,
            "prompt_caching": False,
            "probe_model": "my-custom-probe-model"
        }
        
        with patch("src.utils.llm_client.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Mock successful probe
            probe_resp = Mock()
            probe_resp.parse.return_value = Mock(id="probe_id")
            probe_resp.headers = {
                "x-ratelimit-limit-tokens": "60000",
                "x-ratelimit-remaining-tokens": "50000",
                "x-ratelimit-reset-tokens": "1000ms"
            }
            mock_client.responses.with_raw_response.create.return_value = probe_resp
            mock_client.responses.delete.return_value = None
            
            client = OpenAIClient(config)
            
            # Verify probe_model was set from config
            assert client.probe_model == "my-custom-probe-model"
            
            # Verify fallback chain includes main model
            assert client.model in client.probe_fallback_models
