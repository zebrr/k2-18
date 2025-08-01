"""
Тесты для LLM Client с OpenAI Responses API в асинхронном режиме
"""

import logging
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import openai
import pytest

from src.utils.llm_client import OpenAIClient, TPMBucket


# Fixtures для конфигурации
@pytest.fixture
def test_config():
    """Тестовая конфигурация"""
    return {
        "api_key": "test-key-123",
        "model": "gpt-4o",
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

    def __init__(
        self, text="Test response", reasoning_tokens=0, headers=None, status="completed"
    ):
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
    def test_init_regular_model(
        self, mock_openai_class, mock_get_encoding, test_config
    ):
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
    def test_init_reasoning_model(
        self, mock_openai_class, mock_get_encoding, reasoning_config
    ):
        """Проверка инициализации с reasoning моделью"""
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder

        client = OpenAIClient(reasoning_config)

        assert client.is_reasoning_model is True

    @patch("src.utils.llm_client.OpenAIClient._update_tpm_via_probe")
    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_successful_response(
        self, mock_openai_class, mock_get_encoding, mock_probe, test_config
    ):
        """Проверка успешного вызова API с асинхронной логикой"""
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
        initial_response = Mock(id="resp_12345", status="queued")
        mock_raw_initial = MagicMock()
        mock_raw_initial.headers = initial_headers
        mock_raw_initial.parse.return_value = initial_response

        # Final response
        final_response = MockResponse("Generated text response", headers=status_headers)

        # Создаем sequence для retrieve
        retrieve_responses = create_async_response_sequence(final_response)

        # Настраиваем mock для create
        mock_client_instance.responses.with_raw_response.create.return_value = (
            mock_raw_initial
        )

        # Настраиваем mock для retrieve с последовательностью
        mock_raw_responses = []
        for resp in retrieve_responses:
            mock_raw = MagicMock()
            mock_raw.headers = status_headers
            mock_raw.parse.return_value = resp
            mock_raw_responses.append(mock_raw)

        mock_client_instance.responses.with_raw_response.retrieve.side_effect = (
            mock_raw_responses
        )

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
        create_call_args = (
            mock_client_instance.responses.with_raw_response.create.call_args[1]
        )
        assert create_call_args["background"] is True

        # Проверка что был polling через retrieve
        assert (
            mock_client_instance.responses.with_raw_response.retrieve.call_count
            == len(retrieve_responses)
        )

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_reasoning_model_response(
        self, mock_openai_class, mock_get_encoding, reasoning_config
    ):
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
        final_response = MockReasoningResponse(
            "Reasoning response", reasoning_tokens=500
        )

        # Настраиваем mock
        mock_client_instance.responses.with_raw_response.create.return_value = (
            mock_raw_initial
        )

        # Для retrieve - сразу completed
        mock_raw_final = MagicMock()
        mock_raw_final.headers = headers
        mock_raw_final.parse.return_value = final_response
        mock_client_instance.responses.with_raw_response.retrieve.return_value = (
            mock_raw_final
        )

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
            assert client.last_response_id == "resp_001"

            # Второй запрос
            _, response_id2, _ = client.create_response("Prompt 2", "Input 2")

        # Проверка второго вызова create
        second_call_args = (
            mock_client_instance.responses.with_raw_response.create.call_args_list[1][1]
        )
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
        rate_limit_error = openai.RateLimitError(
            "Rate limit exceeded", response=Mock(), body={}
        )
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

        mock_client_instance.responses.with_raw_response.retrieve.return_value = (
            mock_raw_final
        )

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
    def test_wait_before_request(
        self, mock_openai_class, mock_get_encoding, test_config
    ):
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

        mock_client_instance.responses.with_raw_response.create.return_value = (
            mock_raw_initial
        )
        mock_client_instance.responses.with_raw_response.retrieve.return_value = (
            mock_raw_final
        )

        client = OpenAIClient(test_config)

        # Устанавливаем низкий лимит
        current_time = 1000.0
        client.tpm_bucket.remaining_tokens = (
            2000  # Мало для 1000 input + 4096 max_completion
        )
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
    def test_repair_response(
        self, mock_openai_class, mock_get_encoding, mock_probe, test_config
    ):
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

            # Repair запрос
            response_text, _, _ = client.repair_response("Original prompt", "Input")

        # Проверка repair вызова
        repair_call_args = (
            mock_client_instance.responses.with_raw_response.create.call_args_list[1][1]
        )
        assert "Return **valid JSON** only" in repair_call_args["instructions"]
        assert repair_call_args["previous_response_id"] == "resp_001"
        assert response_text == "Repaired response"

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
        mock_refusal_content = Mock(
            type="refusal", refusal="Cannot process this request"
        )
        mock_output = Mock(
            content=[mock_refusal_content], status="completed", type="message"
        )
        mock_response = Mock(output=[mock_output], status="completed", id="resp_12345")

        mock_raw_final = MagicMock()
        mock_raw_final.headers = {}
        mock_raw_final.parse.return_value = mock_response

        # Настраиваем mock
        mock_client_instance.responses.with_raw_response.create.return_value = (
            mock_raw_initial
        )
        mock_client_instance.responses.with_raw_response.retrieve.return_value = (
            mock_raw_final
        )

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
        """Проверка обработки incomplete статуса с автоматическим увеличением токенов"""
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

        # Вторая попытка - успех
        initial_response2 = Mock(id="resp_002", status="queued")
        mock_raw_initial2 = MagicMock()
        mock_raw_initial2.headers = {}
        mock_raw_initial2.parse.return_value = initial_response2

        final_response = MockResponse("Success with more tokens")
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
            mock_raw_incomplete,
            mock_raw_final,
        ]

        client = OpenAIClient(test_config)

        with patch("src.utils.llm_client.time.sleep"):
            response_text, _, _ = client.create_response("Prompt", "Input")

        # Проверки
        assert response_text == "Success with more tokens"
        assert mock_client_instance.responses.with_raw_response.create.call_count == 2

        # Проверка что токены увеличились во втором вызове
        second_call_args = (
            mock_client_instance.responses.with_raw_response.create.call_args_list[1][1]
        )
        assert second_call_args["max_output_tokens"] == int(
            4096 * 1.5
        )  # Увеличено в 1.5 раза

    @patch("src.utils.llm_client.tiktoken.get_encoding")
    @patch("src.utils.llm_client.OpenAI")
    def test_timeout_cancellation(
        self, mock_openai_class, mock_get_encoding, test_config
    ):
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
        mock_client_instance.responses.with_raw_response.create.return_value = (
            mock_raw_initial
        )
        mock_client_instance.responses.with_raw_response.retrieve.return_value = (
            mock_raw_queued
        )
        mock_client_instance.responses.cancel.return_value = None  # Mock cancel метод

        client = OpenAIClient(test_config)
        client.config["timeout"] = 0.1  # Очень короткий timeout для теста

        with patch("src.utils.llm_client.time.sleep") as mock_sleep:
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

        failed_response = Mock(
            id="resp_001", status="failed", error=Mock(message="Internal error")
        )
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

        mock_client_instance.responses.with_raw_response.retrieve.return_value = (
            mock_raw_final
        )

        caplog.set_level(logging.INFO)

        test_config = {
            "api_key": "test-key",
            "model": "gpt-4o",
            "tpm_limit": 100000,
            "tpm_safety_margin": 0.15,
            "max_completion": 4096,
            "timeout": 30,
            "max_retries": 3,
        }
        client = OpenAIClient(test_config)

        # Настраиваем mock probe
        mock_probe.return_value = None

        response_text, response_id, usage = client.create_response(
            "Test prompt", "Test input"
        )

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
        assert (
            bucket.reset_time == 1000
        )  # 1000.0 + 820/1000 = 1000.82, округлено до 1000

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
        assert any(
            "TPM check passed: sufficient tokens available" in msg
            for msg in log_messages
        )
