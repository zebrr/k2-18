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

from utils.llm_embeddings import (EmbeddingsClient, cosine_similarity_batch,
                                  get_embeddings)


class TestEmbeddingsClient:
    """Тесты для класса EmbeddingsClient."""

    @pytest.fixture
    def mock_config(self):
        """Тестовая конфигурация."""
        return {
            "embedding_api_key": "test-key",
            "embedding_model": "text-embedding-3-small",
            "embedding_tpm_limit": 350000,
            "max_retries": 3,
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
            assert client.tpm_limit == 350000
            assert client.max_retries == 3

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
                    batches = client._batch_texts(texts)

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
            # Настройка mock
            mock_client = Mock()
            mock_client.embeddings.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client

            client = EmbeddingsClient(mock_config)
            texts = ["text1", "text2"]

            embeddings = client.get_embeddings(texts)

            # Проверки
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (2, 1536)
            mock_client.embeddings.create.assert_called_once()

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
            # Настройка mock
            mock_client = Mock()
            # Первый вызов - rate limit error, второй - успех
            mock_client.embeddings.create.side_effect = [
                Exception("rate_limit exceeded"),
                Mock(data=[Mock(embedding=[0.1] * 1536)]),
            ]
            mock_openai.return_value = mock_client

            client = EmbeddingsClient(mock_config)

            with patch("time.sleep"):  # Не ждем в тестах
                embeddings = client.get_embeddings(["text"])

            assert embeddings.shape == (1, 1536)
            assert mock_client.embeddings.create.call_count == 2

    def test_get_embeddings_max_retries_exceeded(self, mock_config):
        """Тест исчерпания попыток retry."""
        with patch("utils.llm_embeddings.OpenAI") as mock_openai:
            # Настройка mock
            mock_client = Mock()
            mock_client.embeddings.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            client = EmbeddingsClient(mock_config)
            client.max_retries = 2  # Уменьшаем для теста

            with patch("time.sleep"):  # Не ждем в тестах
                with pytest.raises(Exception, match="API Error"):
                    client.get_embeddings(["text"])

            # Должен попытаться max_retries раз
            assert mock_client.embeddings.create.call_count == 2


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
