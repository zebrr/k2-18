# util_llm_embeddings.md

## Status: READY

Модуль для работы с OpenAI Embeddings API. Предоставляет функции для получения векторных представлений текста и вычисления косинусной близости между векторами.

## Public API

### EmbeddingsClient
Клиент для работы с OpenAI Embeddings API с поддержкой батчевой обработки и контроля TPM лимитов.

#### EmbeddingsClient.__init__(config: Dict[str, Any])
Инициализация клиента.
- **Input**: config - словарь конфигурации с ключами:
  - embedding_api_key (str, optional) - API ключ для embeddings (fallback на api_key)
  - api_key (str) - основной API ключ
  - embedding_model (str) - модель для эмбеддингов (по умолчанию "text-embedding-3-small")
  - embedding_tpm_limit (int) - TPM лимит (по умолчанию 350000)
  - max_retries (int) - количество повторных попыток (по умолчанию 3)
- **Raises**: ValueError - если API ключ не найден

#### EmbeddingsClient.get_embeddings(texts: List[str]) -> np.ndarray
Получение эмбеддингов для списка текстов.
- **Input**: texts - список текстов для обработки
- **Returns**: numpy array shape (n_texts, 1536) с нормализованными векторами
- **Terminal Output**: При обработке нескольких батчей показывает прогресс и статус
- **Features**: 
  - Автоматическое разбиение на батчи (до 2048 текстов за запрос)
  - Обрезка текстов длиннее 8192 токенов
  - Контроль TPM лимитов с ожиданием
  - Retry логика с exponential backoff
  - **Обработка пустых строк**: пустые строки (после strip()) не отправляются в API, для них возвращаются нулевые векторы
  - **Сохранение порядка**: исходный порядок текстов сохраняется в результате
- **Raises**: Exception - при ошибках API после всех retry попыток

### get_embeddings(texts: List[str], config: Dict[str, Any]) -> np.ndarray
Простая обертка для получения эмбеддингов.
- **Input**: 
  - texts - список текстов
  - config - конфигурация (должна содержать embedding_api_key или api_key)
- **Returns**: numpy array с эмбеддингами

### cosine_similarity_batch(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray
Вычисление косинусной близости между двумя наборами векторов.
- **Input**: 
  - embeddings1 - массив векторов shape (n1, dim)
  - embeddings2 - массив векторов shape (n2, dim)
- **Returns**: массив косинусных близостей shape (n1, n2)
- **Note**: Векторы должны быть нормализованы (OpenAI API возвращает нормализованные)

## Internal Methods

#### _count_tokens(text: str) -> int
Подсчет токенов в тексте с использованием cl100k_base токенизера.

#### _truncate_text(text: str, max_tokens: int = 8000) -> str
Обрезка текста до указанного количества токенов.

#### _update_tpm_state(tokens_used: int, headers: Optional[Dict[str, str]] = None)
Обновление состояния TPM bucket из response headers или простым вычитанием.

#### _wait_for_tokens(required_tokens: int, safety_margin: float = 0.15)
Ожидание доступности токенов с учетом safety margin.
- **Terminal Output**: Информирует пользователя об ожидании и восстановлении лимита

#### _batch_texts(texts: List[str]) -> List[List[str]]
Разбиение текстов на батчи с учетом лимитов API (2048 текстов, ~100K токенов).
- **Terminal Output**: Предупреждает об обрезке текстов длиннее 8192 токенов

## Terminal Output

Модуль выводит информацию о прогрессе обработки в терминал с форматом `[HH:MM:SS] EMBEDDINGS | сообщение`:

### Обработка батчей
- Информация о начале обработки (при >1 батча):
  ```
  [10:30:00] EMBEDDINGS | Processing 457 texts in 5 batches...
  ```

- Прогресс по батчам (при >1 батча):
  ```
  [10:30:05] EMBEDDINGS | ✅ Batch 1/5 completed
  [10:30:10] EMBEDDINGS | ✅ Batch 2/5 completed
  ```

- Финальное сообщение (при >1 батча):
  ```
  [10:31:08] EMBEDDINGS | ✅ Completed: 457 texts processed
  ```

### Обработка текстов
- Предупреждение об обрезке длинных текстов:
  ```
  [10:30:01] EMBEDDINGS | ⚠️ Text truncated: 9234 → 8000 tokens
  ```

### TPM контроль
- Ожидание восстановления лимита:
  ```
  [10:30:12] EMBEDDINGS | ⏳ Waiting 42.3s for TPM limit reset...
  [10:30:55] EMBEDDINGS | ✅ TPM limit reset, continuing...
  ```

### Обработка ошибок
- Rate limit с retry:
  ```
  [10:31:20] EMBEDDINGS | ⏳ Rate limit hit, retry 1/3 in 10s...
  ```

**Примечание**: Вывод в терминал происходит только для операций с несколькими батчами. При обработке небольшого количества текстов (1 батч) модуль работает "тихо".

## Test Coverage

### Unit Tests (test_llm_embeddings.py)
- **TestEmbeddingsClient**: 14 тестов
  - test_init_with_embedding_api_key
  - test_init_fallback_to_api_key
  - test_init_no_api_key
  - test_count_tokens
  - test_truncate_text
  - test_batch_texts
  - test_batch_texts_with_long_text
  - test_update_tpm_state_with_headers
  - test_update_tpm_state_without_headers
  - test_wait_for_tokens
  - test_get_embeddings_success
  - test_get_embeddings_empty_input
  - test_get_embeddings_rate_limit_retry
  - test_get_embeddings_max_retries_exceeded

- **TestHelperFunctions**: 3 теста
  - test_get_embeddings_wrapper
  - test_cosine_similarity_batch
  - test_cosine_similarity_batch_normalized

### Integration Tests (test_llm_embeddings_integration.py)
- **TestSingleEmbedding**: 2 теста (одиночные тексты)
- **TestBatchProcessing**: 3 теста (батчи разных размеров)
- **TestLongTexts**: 3 теста (обработка длинных текстов)
- **TestCosineSimilarity**: 4 теста (проверка similarity)
- **TestEdgeCases**: 5 тестов (пустые строки, спецсимволы)
- **TestTPMLimits**: 2 теста (контроль лимитов)
- **TestErrorHandling**: 2 теста (обработка ошибок)
- **TestVectorProperties**: 2 теста (свойства векторов)
- **TestPerformance**: 1 тест (производительность)

## Dependencies

- **Standard Library**: time, logging
- **External**: openai, tiktoken, numpy
- **Internal**: None

## Configuration

Модуль использует следующие параметры из конфигурации:
- `embedding_api_key` - API ключ для embeddings (опционально)
- `api_key` - основной API ключ (fallback)
- `embedding_model` - модель OpenAI для эмбеддингов
- `embedding_tpm_limit` - лимит токенов в минуту (350000 для embedding моделей)
- `max_retries` - количество повторных попыток при ошибках

## Performance Notes

- **Токенизер**: Использует cl100k_base (НЕ o200k_base!) согласно требованиям OpenAI Embeddings API
- **Батчинг**: Автоматическое разбиение на батчи для оптимальной производительности
- **TPM контроль**: Встроенный контроль лимитов с ожиданием восстановления
- **Размерность**: text-embedding-3-small возвращает векторы размерности 1536
- **Нормализация**: Векторы автоматически нормализованы (норма = 1)
- **Пустые строки**: OpenAI API не принимает пустые строки, модуль автоматически фильтрует их и возвращает нулевые векторы

## Usage Examples

```python
from utils import load_config, get_embeddings, cosine_similarity_batch

# Загрузка конфигурации
config = load_config()

# Получение эмбеддингов для текстов
texts = ["Python programming", "Machine learning", "Data science"]
embeddings = get_embeddings(texts, config['dedup'])

# Вычисление попарной косинусной близости
similarity_matrix = cosine_similarity_batch(embeddings, embeddings)
print(f"Similarity between text 0 and 1: {similarity_matrix[0, 1]:.3f}")

# Использование с классом напрямую для более тонкого контроля
from utils.llm_embeddings import EmbeddingsClient

client = EmbeddingsClient(config['dedup'])
# Обработка большого количества текстов
large_text_list = ["text"] * 5000  # Будет автоматически разбито на батчи
embeddings = client.get_embeddings(large_text_list)

# Обработка текстов с пустыми строками
mixed_texts = ["Hello", "", "World", "   ", "!"]
embeddings = client.get_embeddings(mixed_texts)
# Результат: shape (5, 1536), где embeddings[1] и embeddings[3] - нулевые векторы
```

## Error Handling

```python
try:
    embeddings = get_embeddings(texts, config)
except ValueError as e:
    # Ошибка конфигурации (нет API ключа)
    print(f"Config error: {e}")
except Exception as e:
    # API ошибки после всех retry попыток
    if "rate_limit" in str(e).lower():
        print("Rate limit exceeded, try again later")
    else:
        print(f"API error: {e}")
```

## Edge Cases

- **Пустые строки**: Автоматически заменяются на нулевые векторы без вызова API
- **Слишком длинные тексты**: Обрезаются до 8000 токенов с логированием предупреждения
- **Пустой входной массив**: Возвращает пустой numpy array shape (0,)
- **Смешанные языки**: Поддерживаются, модель multilingual
- **Специальные символы и эмодзи**: Обрабатываются корректно