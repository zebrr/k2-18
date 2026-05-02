"""
Тесты для модуля llm_embeddings.
"""

import sys
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from utils.llm_embeddings import EmbeddingsClient, cosine_similarity_batch, get_embeddings


class TestEmbeddingsClient:
    """Тесты для класса EmbeddingsClient."""

    @pytest.fixture
    def mock_config(self):
        """Тестовая конфигурация."""
        return {
            "embedding_api_key": "test-key",
            "embedding_model": "text-embedding-3-small",
            "embedding_tpm_limit": 1000000,
            "max_retries": 3,
            "max_batch_tokens": 100000,
            "max_texts_per_batch": 2048,
            "truncate_tokens": 8000,
        }

    @pytest.fixture
    def mock_openai_response(self):
        """Mock ответа OpenAI API."""
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
        ]
        return mock_response

    def test_init_with_embedding_api_key(self, mock_config):
        """Тест инициализации с embedding_api_key."""
        with patch("utils.llm_embeddings.OpenAI"):
            client = EmbeddingsClient(mock_config)
            assert client.model == "text-embedding-3-small"
            assert client.tpm_limit == 1000000
            assert client.max_retries == 3
            assert client.max_batch_tokens == 100000
            assert client.max_texts_per_batch == 2048
            assert client.truncate_tokens == 8000

    def test_init_fallback_to_api_key(self):
        """Тест fallback на основной api_key."""
        config = {"api_key": "main-key", "embedding_model": "text-embedding-3-small"}
        with patch("utils.llm_embeddings.OpenAI"):
            client = EmbeddingsClient(config)
            assert client.model == "text-embedding-3-small"

    def test_init_no_api_key(self):
        """Тест ошибки при отсутствии API ключа."""
        config = {"embedding_model": "text-embedding-3-small"}
        with pytest.raises(ValueError, match="API key not found"):
            EmbeddingsClient(config)

    def test_count_tokens(self, mock_config):
        """Тест подсчета токенов."""
        with patch("utils.llm_embeddings.OpenAI"):
            client = EmbeddingsClient(mock_config)
            # Простой текст для теста
            text = "Hello world"
            tokens = client._count_tokens(text)
            assert isinstance(tokens, int)
            assert tokens > 0

    def test_truncate_text(self, mock_config):
        """Тест обрезки текста."""
        with patch("utils.llm_embeddings.OpenAI"):
            client = EmbeddingsClient(mock_config)

            # Короткий текст - не обрезается
            short_text = "Short text"
            assert client._truncate_text(short_text, 100) == short_text

            # Длинный текст - обрезается
            long_text = "Very " * 5000  # Очень длинный текст
            truncated = client._truncate_text(long_text, 100)
            tokens = client._count_tokens(truncated)
            assert tokens <= 100

    def test_batch_texts(self, mock_config):
        """Тест разбиения текстов на батчи."""
        with patch("utils.llm_embeddings.OpenAI"):
            client = EmbeddingsClient(mock_config)

            # Маленький батч
            texts = ["text1", "text2", "text3"]
            batches = client._batch_texts(texts)
            assert len(batches) == 1
            assert len(batches[0]) == 3

            # Большой батч (превышает лимит)
            many_texts = ["text"] * 3000
            batches = client._batch_texts(many_texts)
            assert len(batches) > 1
            assert all(len(batch) <= client.max_texts_per_request for batch in batches)

    def test_batch_texts_with_long_text(self, mock_config):
        """Тест обработки длинных текстов при батчинге."""
        with patch("utils.llm_embeddings.OpenAI"):
            client = EmbeddingsClient(mock_config)

            # Mock подсчета токенов
            with patch.object(client, "_count_tokens") as mock_count:
                # Первый текст слишком длинный
                mock_count.side_effect = [10000, 100, 100]

                texts = ["very long text", "normal text", "another text"]
                with patch.object(client, "_truncate_text", return_value="truncated"):
                    _ = client._batch_texts(texts)

                    # Должен вызвать truncate для длинного текста
                    client._truncate_text.assert_called_once()

    def test_update_tpm_state_with_headers(self, mock_config):
        """Тест обновления TPM состояния из headers."""
        with patch("utils.llm_embeddings.OpenAI"):
            client = EmbeddingsClient(mock_config)

            headers = {
                "x-ratelimit-remaining-tokens": "300000",
                "x-ratelimit-reset-tokens": "5000ms",
            }

            client._update_tpm_state(1000, headers)
            assert client.remaining_tokens == 300000
            # reset_time должен быть примерно через 5 секунд
            assert client.reset_time > time.time()

    def test_update_tpm_state_without_headers(self, mock_config):
        """Тест обновления TPM состояния без headers."""
        with patch("utils.llm_embeddings.OpenAI"):
            client = EmbeddingsClient(mock_config)
            initial_tokens = client.remaining_tokens

            client._update_tpm_state(1000)
            assert client.remaining_tokens == initial_tokens - 1000

    def test_wait_for_tokens(self, mock_config):
        """Тест ожидания доступности токенов."""
        with patch("utils.llm_embeddings.OpenAI"):
            client = EmbeddingsClient(mock_config)

            # Устанавливаем низкий остаток токенов
            client.remaining_tokens = 100
            client.reset_time = time.time() + 0.1  # Сброс через 0.1 секунды

            with patch("time.sleep") as mock_sleep:
                client._wait_for_tokens(10000)
                # Должен был вызвать sleep
                mock_sleep.assert_called_once()
                assert client.remaining_tokens == client.tpm_limit

    def test_get_embeddings_success(self, mock_config, mock_openai_response):
        """Тест успешного получения эмбеддингов."""
        with patch("utils.llm_embeddings.OpenAI") as mock_openai:
            # Настройка mock для raw response
            mock_raw = Mock()
            mock_raw.headers = {
                "x-ratelimit-remaining-tokens": "900000",
                "x-ratelimit-reset-tokens": "5000ms",
            }
            mock_raw.parse.return_value = mock_openai_response

            # Настройка mock client
            mock_client = Mock()
            mock_client.embeddings.with_raw_response.create.return_value = mock_raw
            mock_openai.return_value = mock_client

            client = EmbeddingsClient(mock_config)
            texts = ["text1", "text2"]

            embeddings = client.get_embeddings(texts)

            # Проверки
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (2, 1536)
            mock_client.embeddings.with_raw_response.create.assert_called_once()

    def test_get_embeddings_empty_input(self, mock_config):
        """Тест обработки пустого входа."""
        with patch("utils.llm_embeddings.OpenAI"):
            client = EmbeddingsClient(mock_config)
            embeddings = client.get_embeddings([])
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (0,)

    def test_get_embeddings_rate_limit_retry(self, mock_config):
        """Тест retry при rate limit."""
        with patch("utils.llm_embeddings.OpenAI") as mock_openai:
            # Настройка mock для raw response
            mock_raw_success = Mock()
            mock_raw_success.headers = {
                "x-ratelimit-remaining-tokens": "900000",
                "x-ratelimit-reset-tokens": "5000ms",
            }
            mock_raw_success.parse.return_value = Mock(data=[Mock(embedding=[0.1] * 1536)])

            # Настройка mock client
            mock_client = Mock()
            # Первый вызов - rate limit error, второй - успех
            mock_client.embeddings.with_raw_response.create.side_effect = [
                Exception("rate_limit exceeded"),
                mock_raw_success,
            ]
            mock_openai.return_value = mock_client

            client = EmbeddingsClient(mock_config)

            with patch("time.sleep"):  # Не ждем в тестах
                embeddings = client.get_embeddings(["text"])

            assert embeddings.shape == (1, 1536)
            assert mock_client.embeddings.with_raw_response.create.call_count == 2

    def test_get_embeddings_max_retries_exceeded(self, mock_config):
        """Тест исчерпания попыток retry."""
        with patch("utils.llm_embeddings.OpenAI") as mock_openai:
            # Настройка mock
            mock_client = Mock()
            mock_client.embeddings.with_raw_response.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            client = EmbeddingsClient(mock_config)
            client.max_retries = 2  # Уменьшаем для теста

            with patch("time.sleep"):  # Не ждем в тестах
                with pytest.raises(Exception, match="API Error"):
                    client.get_embeddings(["text"])

            # Должен попытаться max_retries раз
            assert mock_client.embeddings.with_raw_response.create.call_count == 2

    def test_config_parameters_loading(self):
        """Тест загрузки новых параметров из конфига."""
        config = {
            "embedding_api_key": "test-key",
            "max_batch_tokens": 50000,
            "max_texts_per_batch": 1000,
            "truncate_tokens": 7000,
        }

        with patch("utils.llm_embeddings.OpenAI"):
            client = EmbeddingsClient(config)

            # Проверяем что параметры применились
            assert client.max_batch_tokens == 50000
            assert client.max_texts_per_batch == 1000
            assert client.truncate_tokens == 7000
            assert client.max_texts_per_request == 1000  # Должен быть равен max_texts_per_batch

    def test_with_raw_response_handling(self, mock_config):
        """Тест обработки raw response с headers."""
        with patch("utils.llm_embeddings.OpenAI") as mock_openai:
            # Создать mock raw response
            mock_raw = Mock()
            mock_raw.headers = {
                "x-ratelimit-remaining-tokens": "900000",
                "x-ratelimit-reset-tokens": "5000ms",
            }
            mock_raw.parse.return_value = Mock(data=[Mock(embedding=[0.1] * 1536)])

            # Настроить mock client
            mock_client = Mock()
            mock_client.embeddings.with_raw_response.create.return_value = mock_raw
            mock_openai.return_value = mock_client

            client = EmbeddingsClient(mock_config)

            # Вызвать get_embeddings
            _ = client.get_embeddings(["test"])

            # Проверить извлечение headers
            assert client.remaining_tokens == 900000
            assert client.reset_time > time.time()

            # Проверить что вызов был с with_raw_response
            mock_client.embeddings.with_raw_response.create.assert_called_once()

    def test_fallback_when_headers_not_available(self, mock_config):
        """Тест fallback на простое вычитание токенов когда headers недоступны."""
        with patch("utils.llm_embeddings.OpenAI") as mock_openai:
            # Создать mock raw response без headers
            mock_raw = Mock()
            mock_raw.headers = {}  # Пустые headers
            mock_raw.parse.return_value = Mock(data=[Mock(embedding=[0.1] * 1536)])

            # Настроить mock client
            mock_client = Mock()
            mock_client.embeddings.with_raw_response.create.return_value = mock_raw
            mock_openai.return_value = mock_client

            client = EmbeddingsClient(mock_config)
            initial_tokens = client.remaining_tokens

            # Вызвать get_embeddings
            with patch.object(client, "_count_tokens", return_value=100):
                _ = client.get_embeddings(["test"])

            # Проверить что использовался fallback (простое вычитание)
            assert client.remaining_tokens == initial_tokens - 100


class TestHelperFunctions:
    """Тесты для вспомогательных функций."""

    def test_get_embeddings_wrapper(self):
        """Тест функции-обертки get_embeddings."""
        config = {"embedding_api_key": "test-key"}
        texts = ["text1", "text2"]

        mock_embeddings = np.array([[0.1] * 1536, [0.2] * 1536])

        with patch("utils.llm_embeddings.EmbeddingsClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get_embeddings.return_value = mock_embeddings
            mock_client_class.return_value = mock_client

            result = get_embeddings(texts, config)

            assert np.array_equal(result, mock_embeddings)
            mock_client.get_embeddings.assert_called_once_with(texts)

    def test_cosine_similarity_batch(self):
        """Тест вычисления косинусной близости."""
        # Создаем нормализованные векторы
        emb1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        emb2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        similarity = cosine_similarity_batch(emb1, emb2)

        # Проверяем размерность
        assert similarity.shape == (2, 3)

        # Проверяем значения
        # emb1[0] идентичен emb2[0], косинус = 1
        assert similarity[0, 0] == 1.0
        # emb1[0] ортогонален emb2[1], косинус = 0
        assert similarity[0, 1] == 0.0
        # emb1[1] идентичен emb2[1], косинус = 1
        assert similarity[1, 1] == 1.0

    def test_cosine_similarity_batch_normalized(self):
        """Тест косинусной близости с реальными нормализованными векторами."""
        # Симулируем нормализованные векторы от OpenAI
        vec1 = np.array([0.6, 0.8])  # норма = 1
        vec2 = np.array([0.8, 0.6])  # норма = 1

        emb1 = np.array([vec1])
        emb2 = np.array([vec2])

        similarity = cosine_similarity_batch(emb1, emb2)

        # Косинус между векторами
        expected = 0.6 * 0.8 + 0.8 * 0.6  # = 0.96
        assert np.isclose(similarity[0, 0], expected)

    def test_get_embeddings_with_mixed_empty_texts(self):
        """Тест обработки смешанных пустых и непустых текстов (покрытие строк 319-330)."""
        texts = [
            "First non-empty text",
            "",  # Empty text
            "   ",  # Whitespace only
            "Second non-empty text",
            "",  # Another empty
            "Third non-empty text",
        ]

        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
            Mock(embedding=[0.3] * 1536),
        ]

        config = {
            "embedding_api_key": "test-key",
            "embedding_model": "text-embedding-3-small",
            "embedding_tpm_limit": 1000000,
            "batch_size": 100,
        }

        with patch("utils.llm_embeddings.OpenAI") as mock_openai:
            # Создать mock raw response
            mock_raw = Mock()
            mock_raw.headers = {
                "x-ratelimit-remaining-tokens": "900000",
                "x-ratelimit-reset-tokens": "5000ms",
            }
            mock_raw.parse.return_value = mock_response

            mock_client = Mock()
            mock_client.embeddings.with_raw_response.create.return_value = mock_raw
            mock_openai.return_value = mock_client

            with patch("builtins.print"):  # Suppress print output
                result = get_embeddings(texts, config)

        # Проверяем, что результат имеет правильную форму
        assert result.shape == (6, 1536)

        # Проверяем, что пустые тексты получили нулевые векторы
        assert np.allclose(result[1], 0)  # Empty text
        assert np.allclose(result[2], 0)  # Whitespace only
        assert np.allclose(result[4], 0)  # Another empty

        # Проверяем, что непустые тексты получили ненулевые векторы
        assert not np.allclose(result[0], 0)  # First non-empty
        assert not np.allclose(result[3], 0)  # Second non-empty
        assert not np.allclose(result[5], 0)  # Third non-empty

    def test_get_embeddings_batch_processing_with_retries(self):
        """Тест batch processing с retry при ошибках API."""
        texts = ["text" + str(i) for i in range(250)]  # Больше чем один batch

        # Создаем 250 embeddings для всех текстов
        # Первые 100 - [0.1] * 1536, следующие 100 - [0.2] * 1536, последние 50 - [0.3] * 1536
        all_embeddings = []
        all_embeddings.extend([Mock(embedding=[0.1] * 1536) for _ in range(100)])
        all_embeddings.extend([Mock(embedding=[0.2] * 1536) for _ in range(100)])
        all_embeddings.extend([Mock(embedding=[0.3] * 1536) for _ in range(50)])

        # Один Mock response со всеми 250 embeddings
        mock_response = Mock(data=all_embeddings)

        config = {
            "embedding_api_key": "test-key",
            "embedding_model": "text-embedding-3-small",
            "embedding_tpm_limit": 1000000,
            "batch_size": 250,  # Process all at once
            "max_retries": 3,
        }

        with patch("utils.llm_embeddings.OpenAI") as mock_openai:
            # Создать mock raw response для успешного вызова
            mock_raw_success = Mock()
            mock_raw_success.headers = {
                "x-ratelimit-remaining-tokens": "900000",
                "x-ratelimit-reset-tokens": "5000ms",
            }
            mock_raw_success.parse.return_value = mock_response

            mock_client = Mock()
            # Первая попытка вызывает ошибку, вторая успешная
            mock_client.embeddings.with_raw_response.create.side_effect = [
                Exception("Rate limit error"),
                mock_raw_success,
            ]
            mock_openai.return_value = mock_client

            with patch("utils.llm_embeddings.time.sleep"):  # Skip sleep during test
                with patch("utils.llm_embeddings.logger"):  # Suppress log output
                    result = get_embeddings(texts, config)

        # Проверяем, что все тексты получили embeddings несмотря на ошибку
        assert result.shape == (250, 1536)
        assert not np.allclose(result, 0)  # Все векторы должны быть ненулевыми

    def test_embeddings_client_retry_with_final_failure(self):
        """Тест исчерпания попыток retry."""
        config = {
            "embedding_api_key": "test-key",
            "embedding_model": "text-embedding-3-small",
            "embedding_tpm_limit": 1000000,
            "max_retries": 2,
        }

        with patch("utils.llm_embeddings.OpenAI") as mock_openai:
            mock_client = Mock()
            # Все попытки вызывают ошибку
            mock_client.embeddings.with_raw_response.create.side_effect = Exception(
                "Persistent API error"
            )
            mock_openai.return_value = mock_client

            client = EmbeddingsClient(config)

            with patch("utils.llm_embeddings.time.sleep"):  # Skip sleep
                with pytest.raises(Exception) as exc_info:
                    client.get_embeddings(["test text"])

                assert "Persistent API error" in str(exc_info.value)
