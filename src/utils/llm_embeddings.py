"""
Модуль для работы с OpenAI Embeddings API.

Предоставляет функции для получения векторных представлений текста
и вычисления косинусной близости между векторами.
"""

import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timezone
from openai import OpenAI
import tiktoken

# Настройка логирования
logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """
    Клиент для работы с OpenAI Embeddings API.
    
    Поддерживает:
    - Батчевую обработку (до 2048 текстов за запрос)
    - Контроль TPM лимитов (350,000 токенов/мин)
    - Обработку длинных текстов (до 8192 токенов)
    - Retry логику с exponential backoff
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация клиента.
        
        Args:
            config: Словарь конфигурации с ключами:
                - embedding_api_key: API ключ (опционально, fallback на api_key)
                - api_key: Основной API ключ
                - embedding_model: Модель для эмбеддингов
                - embedding_tpm_limit: TPM лимит (по умолчанию 350000)
                - max_retries: Количество повторных попыток (по умолчанию 3)
        """
        # API ключ - сначала пробуем embedding_api_key, потом api_key
        api_key = config.get('embedding_api_key') or config.get('api_key')
        if not api_key:
            raise ValueError("API key not found in config (embedding_api_key or api_key)")
            
        self.client = OpenAI(api_key=api_key)
        self.model = config.get('embedding_model', 'text-embedding-3-small')
        
        # TPM контроль
        self.tpm_limit = config.get('embedding_tpm_limit', 350000)
        self.remaining_tokens = self.tpm_limit
        self.reset_time = time.time()
        
        # Retry параметры
        self.max_retries = config.get('max_retries', 3)
        
        # Токенизер для подсчета токенов (cl100k_base для embedding моделей)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Лимиты API
        self.max_texts_per_request = 2048
        self.max_tokens_per_text = 8192
        
        logger.info(f"EmbeddingsClient initialized with model={self.model}, tpm_limit={self.tpm_limit}")
    
    def _count_tokens(self, text: str) -> int:
        """Подсчет токенов в тексте."""
        return len(self.encoding.encode(text))
    
    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """Обрезка текста до указанного количества токенов."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Обрезаем и декодируем обратно
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def _update_tpm_state(self, tokens_used: int, headers: Optional[Dict[str, str]] = None):
        """
        Обновление состояния TPM bucket.
        
        Args:
            tokens_used: Количество использованных токенов
            headers: Response headers от API (если есть)
        """
        current_time = time.time()
        
        # Если есть headers от API - используем их
        if headers:
            remaining = headers.get('x-ratelimit-remaining-tokens')
            reset_str = headers.get('x-ratelimit-reset-tokens')
            
            if remaining is not None:
                self.remaining_tokens = int(remaining)
                
            if reset_str:
                # Формат: "XXXms" - конвертируем в секунды
                reset_ms = int(reset_str.rstrip('ms'))
                self.reset_time = current_time + (reset_ms / 1000)
                logger.debug(f"TPM state from headers: remaining={self.remaining_tokens}, reset_in={reset_ms/1000:.1f}s")
        else:
            # Простое вычитание токенов
            if current_time >= self.reset_time:
                # Минута прошла - восстанавливаем лимит
                self.remaining_tokens = self.tpm_limit
                self.reset_time = current_time + 60
            
            self.remaining_tokens -= tokens_used
            logger.debug(f"TPM state updated: remaining={self.remaining_tokens}, used={tokens_used}")
    
    def _wait_for_tokens(self, required_tokens: int, safety_margin: float = 0.15):
        """
        Ожидание доступности токенов.
        
        Args:
            required_tokens: Требуемое количество токенов
            safety_margin: Запас безопасности
        """
        required_with_margin = int(required_tokens * (1 + safety_margin))
        
        if self.remaining_tokens < required_with_margin:
            wait_time = max(0, self.reset_time - time.time())
            if wait_time > 0:
                logger.info(f"Waiting {wait_time:.1f}s for TPM limit reset (need {required_with_margin} tokens)")
                utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(f"[{utc3_time}] EMBEDDINGS | ⏳ Waiting {wait_time:.1f}s for TPM limit reset...")
                time.sleep(wait_time + 0.1)  # +0.1s для надежности
                self.remaining_tokens = self.tpm_limit
                self.reset_time = time.time() + 60
                utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(f"[{utc3_time}] EMBEDDINGS | ✅ TPM limit reset, continuing...")
    
    def _batch_texts(self, texts: List[str]) -> List[List[str]]:
        """
        Разбиение текстов на батчи с учетом лимитов API.
        
        Лимиты:
        - Максимум 2048 текстов в батче
        - Рекомендуется не более ~100K токенов в батче
        
        Returns:
            Список батчей
        """
        batches = []
        current_batch = []
        current_tokens = 0
        max_batch_tokens = 100000  # Soft limit для производительности
        
        for text in texts:
            # Проверяем длину текста
            text_tokens = self._count_tokens(text)
            
            # Если текст слишком длинный - обрезаем
            if text_tokens > self.max_tokens_per_text:
                logger.warning(f"Text with {text_tokens} tokens exceeds limit, truncating to 8000")
                utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(f"[{utc3_time}] EMBEDDINGS | ⚠️ Text truncated: {text_tokens} → 8000 tokens")
                text = self._truncate_text(text, 8000)
                text_tokens = 8000
            
            # Проверяем, поместится ли в текущий батч
            if (len(current_batch) >= self.max_texts_per_request or 
                (current_tokens + text_tokens > max_batch_tokens and current_batch)):
                # Сохраняем текущий батч и начинаем новый
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens
        
        # Добавляем последний батч
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Split {len(texts)} texts into {len(batches)} batches")
        return batches
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Получение эмбеддингов для списка текстов.
        
        Args:
            texts: Список текстов для обработки
            
        Returns:
            numpy array shape (n_texts, 1536) с нормализованными векторами
            
        Raises:
            Exception: При ошибках API после всех retry попыток
        """
        if not texts:
            return np.array([])
        
        # Сохраняем индексы непустых текстов
        non_empty_indices = []
        non_empty_texts = []
        
        for i, text in enumerate(texts):
            if text.strip():  # Непустой текст
                non_empty_indices.append(i)
                non_empty_texts.append(text)
        
        # Если все тексты пустые - возвращаем нулевые векторы
        if not non_empty_texts:
            logger.warning(f"All {len(texts)} texts are empty, returning zero vectors")
            return np.zeros((len(texts), 1536), dtype=np.float32)
        
        # Разбиваем на батчи только непустые тексты
        batches = self._batch_texts(non_empty_texts)

        if len(batches) > 1:
            utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{utc3_time}] EMBEDDINGS | Processing {len(non_empty_texts)} texts in {len(batches)} batches...")

        all_embeddings = []
        
        for batch_idx, batch in enumerate(batches):
            batch_tokens = sum(self._count_tokens(text) for text in batch)
            
            # Ждем доступности токенов
            self._wait_for_tokens(batch_tokens)
            
            # Retry логика
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"Processing batch {batch_idx + 1}/{len(batches)}, {len(batch)} texts")
                    
                    # Вызов API
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model,
                        encoding_format="float"
                    )
                    
                    # Обновляем TPM состояние из headers (если SDK предоставляет)
                    # В новом SDK headers могут быть недоступны напрямую
                    self._update_tpm_state(batch_tokens)
                    
                    # Извлекаем векторы
                    embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(embeddings)
                    
                    logger.debug(f"Batch {batch_idx + 1} processed successfully")

                    if len(batches) > 1:
                        utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                        print(f"[{utc3_time}] EMBEDDINGS | ✅ Batch {batch_idx + 1}/{len(batches)} completed")

                    break
                    
                except Exception as e:
                    last_error = e
                    error_type = type(e).__name__
                    
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        # Rate limit - используем exponential backoff
                        wait_time = (2 ** attempt) * 10
                        logger.warning(f"Rate limit hit, attempt {attempt + 1}/{self.max_retries}, waiting {wait_time}s")
                        utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                        print(f"[{utc3_time}] EMBEDDINGS | ⏳ Rate limit hit, retry {attempt + 1}/{self.max_retries} in {wait_time}s...")
                        time.sleep(wait_time)
                        
                        # Сбрасываем TPM состояние
                        self.remaining_tokens = 0
                        self.reset_time = time.time() + wait_time
                    else:
                        # Другие ошибки - тоже retry с backoff
                        wait_time = (2 ** attempt) * 5
                        logger.warning(f"API error: {error_type}, attempt {attempt + 1}/{self.max_retries}, waiting {wait_time}s")
                        time.sleep(wait_time)
            else:
                # Все попытки исчерпаны
                logger.error(f"Failed to get embeddings after {self.max_retries} attempts")
                raise last_error
        
        # Конвертируем в numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # Восстанавливаем исходный порядок с нулевыми векторами для пустых текстов
        if len(non_empty_indices) < len(texts):
            result = np.zeros((len(texts), 1536), dtype=np.float32)
            for i, idx in enumerate(non_empty_indices):
                result[idx] = embeddings_array[i]
            logger.info(f"Processed {len(non_empty_texts)} non-empty texts out of {len(texts)} total")
            if len(batches) > 1:
                utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(f"[{utc3_time}] EMBEDDINGS | ✅ Completed: {len(non_empty_texts)} texts processed")
            return result
        
        if len(batches) > 1:
            utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{utc3_time}] EMBEDDINGS | ✅ Completed: {len(texts)} texts processed")
        
        return embeddings_array

def get_embeddings(texts: List[str], config: Dict[str, Any]) -> np.ndarray:
    """
    Простая обертка для получения эмбеддингов.
    
    Args:
        texts: Список текстов
        config: Конфигурация (должна содержать embedding_api_key или api_key)
        
    Returns:
        numpy array с эмбеддингами
    """
    client = EmbeddingsClient(config)
    return client.get_embeddings(texts)


def cosine_similarity_batch(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Вычисление косинусной близости между двумя наборами векторов.
    
    Args:
        embeddings1: Массив векторов shape (n1, dim)
        embeddings2: Массив векторов shape (n2, dim)
        
    Returns:
        Массив косинусных близостей shape (n1, n2)
        
    Note:
        Векторы должны быть нормализованы (OpenAI API возвращает нормализованные)
    """
    # Для нормализованных векторов косинусная близость = скалярное произведение
    return np.dot(embeddings1, embeddings2.T)
