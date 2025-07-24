# util_llm_client.md

## Status: READY

Клиент для работы с OpenAI Responses API с поддержкой асинхронного режима, контроля лимитов токенов, цепочек ответов и reasoning моделей.

## Public API

### ResponseUsage
Dataclass для учета использованных токенов.
- **Fields**: 
  - input_tokens (int) - токены во входных данных
  - output_tokens (int) - токены в ответе
  - total_tokens (int) - общее количество (input + output)
  - reasoning_tokens (int) - токены на reasoning (для o* моделей)

### TPMBucket
Класс для контроля лимитов токенов в минуту через response headers.

#### TPMBucket.__init__(initial_limit: int)
Инициализация bucket с начальным лимитом.
- **Input**: initial_limit - начальный лимит токенов в минуту
- **Attributes**: 
  - initial_limit - сохраненный начальный лимит
  - remaining_tokens - текущий остаток токенов
  - reset_time - Unix timestamp сброса лимита

#### TPMBucket.update_from_headers(headers: Dict[str, str]) -> None
Обновление состояния из response headers.
- **Input**: headers - словарь с headers от API (x-ratelimit-remaining-tokens, x-ratelimit-reset-tokens)
- **Logic**: Парсит remaining tokens и reset time (формат "XXXms" или "Xs")

#### TPMBucket.wait_if_needed(required_tokens: int, safety_margin: float = 0.15) -> None
Проверка достаточности токенов и ожидание при необходимости.
- **Input**: required_tokens - необходимое количество, safety_margin - запас безопасности
- **Logic**: Ждет до reset_time если токенов недостаточно
- **Terminal Output**: При ожидании выводит информационное сообщение о восстановлении лимита

### IncompleteResponseError
Exception для обработки incomplete статуса ответа (превышен лимит токенов).

### OpenAIClient
Основной клиент для работы с OpenAI Responses API.

#### OpenAIClient.__init__(config: Dict[str, Any])
Инициализация клиента с конфигурацией.
- **Input**: config - словарь с параметрами:
  - api_key (str) - ключ OpenAI API
  - model (str) - модель (gpt-4o, o4-mini и т.д.)
  - tpm_limit (int) - лимит токенов в минуту
  - tpm_safety_margin (float) - запас безопасности (default 0.15)
  - max_completion (int) - максимум токенов для генерации
  - timeout (int) - таймаут запроса в секундах
  - max_retries (int) - количество повторных попыток
  - temperature (float, optional) - для обычных моделей
  - reasoning_effort (str, optional) - для reasoning моделей
  - reasoning_summary (str, optional) - тип summary для reasoning
  - poll_interval (int) - интервал polling в секундах (default 5)

#### OpenAIClient.create_response(instructions: str, input_data: str, previous_response_id: Optional[str] = None) -> Tuple[str, str, ResponseUsage]
Создание response через OpenAI Responses API (публичный интерфейс).
- **Input**: 
  - instructions - системный промпт
  - input_data - пользовательские данные
  - previous_response_id - ID предыдущего ответа (опционально)
- **Returns**: (response_text, response_id, usage_info)
- **Raises**: 
  - TimeoutError - при превышении timeout
  - IncompleteResponseError - при incomplete статусе
  - ValueError - при failed статусе или отказе модели
  - openai.RateLimitError - при превышении rate limit

#### OpenAIClient.repair_response(instructions: str, input_data: str) -> Tuple[str, str, ResponseUsage]
Repair запрос с тем же previous_response_id для исправления невалидного JSON.
- **Input**: instructions - оригинальный промпт, input_data - оригинальные данные
- **Returns**: (response_text, response_id, usage_info)
- **Logic**: Добавляет требование вернуть только валидный JSON

## Internal Methods

### OpenAIClient._update_tpm_via_probe() -> None
Получение актуальных rate limit данных через probe запрос.
- **Logic**: Синхронный запрос к gpt-4.1-nano с "2+2=?" для получения headers
- **Note**: Необходим т.к. в background режиме OpenAI не возвращает rate limit headers

### OpenAIClient._create_response_async(instructions, input_data, previous_response_id) -> Tuple[str, str, ResponseUsage]
Основная логика создания response в асинхронном режиме.
- **Steps**:
  1. TPM probe для актуализации лимитов
  2. Создание background запроса (background=true)
  3. Polling loop с проверкой статуса
  4. Обработка статусов: completed, incomplete, failed, cancelled, queued
  5. Автоматическое увеличение токенов при incomplete (×1.5, ×2.0)
  6. Отмена при timeout с попыткой cancellation
- **Progress Display**: 
  - Статус queued показывается при первом обнаружении
  - Прогресс отображается каждые 3 polling итерации с временем ожидания

### OpenAIClient._prepare_request_params(instructions, input_data, previous_response_id) -> Dict[str, Any]
Подготовка параметров для Responses API.
- **Logic**: 
  - Базовые параметры + store=true
  - Для reasoning моделей: reasoning параметры, без temperature
  - Для обычных: temperature, без reasoning

### OpenAIClient._extract_response_content(response) -> str
Извлечение текста из response объекта.
- **Logic**:
  - Reasoning модели: output[0]=reasoning, output[1]=message
  - Обычные модели: output[0]=message
  - Обработка refusal и incomplete статусов
  - Поддержка как completed, так и incomplete статусов

### OpenAIClient._extract_usage_info(response) -> ResponseUsage
Извлечение информации о токенах из response.

### OpenAIClient._clean_json_response(response_text: str) -> str
Очистка ответа от markdown оберток.
- **Logic**: Удаляет ```json...``` и ```...``` обертки

## Terminal Output

Модуль использует структурированный вывод в терминал с форматом `[HH:MM:SS] TAG | сообщение`:

### Асинхронная генерация ответов
- **QUEUE** - ожидание в очереди OpenAI (показывается при первом обнаружении)
  ```
  [10:30:01] QUEUE    | ⏳ Response 6871a162606c... waiting in queue...
  ```

- **PROGRESS** - прогресс ожидания в очереди (каждые 3 polling итерации)
  ```
  [10:30:08] PROGRESS | ⏳ Still queued (7s elapsed)...
  [10:30:15] PROGRESS | ⏳ Still queued (14s elapsed)...
  [10:30:22] PROGRESS | ⏳ Still queued (21s elapsed)...
  [10:31:08] PROGRESS | ⏳ Still queued (1m 7s elapsed)...
  ```

- **ERROR** - критические ошибки
  ```
  [10:30:15] ERROR    | ❌ Response incomplete: max_output_tokens
  [10:30:15] ERROR    |    Generated only 4000 tokens
  [10:30:20] ERROR    | ❌ Response generation failed: Server error
  ```

- **HINT** - полезные подсказки для reasoning моделей
  ```
  [10:30:16] HINT     | 💡 Reasoning model needs more tokens. Current limit: 10000
  ```

- **RETRY** - информация о повторных попытках
  ```
  [10:30:25] RETRY    | 🔄 Increasing token limit: 10000 → 15000
  [10:30:30] RETRY    | ⏳ Waiting 40s before retry 2/3...
  ```

### TPM контроль
- **INFO** - восстановление лимита после ожидания
  ```
  [10:31:15] INFO     | ✅ TPM limit reset, continuing...
  ```

## Test Coverage

- **test_llm_client**: 23 теста
  - test_initialization
  - test_tpm_bucket_*
  - test_prepare_request_params_*
  - test_extract_response_content_*
  - test_create_response_async_*
  - test_retry_logic_*
  - test_incomplete_handling
  - test_timeout_handling
  
- **test_llm_client_integration**: 13 тестов
  - test_simple_response
  - test_json_response
  - test_response_chain
  - test_tpm_limiting
  - test_error_handling
  - test_reasoning_model
  - test_headers_update
  - test_background_mode_verification
  - test_incomplete_response_handling
  - test_timeout_cancellation
  - test_console_progress_output
  - test_incomplete_with_reasoning_model
  - test_tpm_probe_mechanism

## Dependencies
- **Standard Library**: time, logging, json, typing, dataclasses, datetime
- **External**: openai>=1.0.0, tiktoken
- **Internal**: None

## Performance Notes

### Асинхронный режим
- Все запросы выполняются в background режиме для контроля над timeout
- Polling с адаптивным интервалом:
  - Первые 3 проверки: каждые 2 секунды (быстрая реакция)
  - Далее: используется poll_interval из конфига
- При timeout выполняется попытка отмены запроса

### Отображение прогресса
- **Оптимизировано для читаемости**: прогресс показывается каждые 3 polling итерации
- **Форматирование времени**: до 60 секунд показывается как "42s", после - как "2m 15s"
- **Единственный промежуточный статус**: только queued, нет промежуточной информации о токенах

### TPM Probe механизм
- **ВАЖНО**: OpenAI не возвращает rate limit headers в background режиме
- Probe запрос к дешевой модели gpt-4.1-nano перед основным запросом
- Overhead: ~20 токенов (0.02% от типичного лимита)
- В коде закомментированы попытки обновления TPM из async headers

### Обработка incomplete
- Автоматическое увеличение max_output_tokens при retry:
  - 1-й retry: ×1.5 от исходного
  - 2-й retry: ×2.0 от исходного
  - 3-й retry: критическая ошибка
- **Исправлен баг**: переменная old_limit теперь корректно определяется перед использованием

### Консольный вывод прогресса
- [QUEUE] ⏳ - запрос в очереди (только первый раз)
- [PROGRESS] ⏳ - прогресс ожидания с временем (каждые 3 проверки)
- [ERROR] ❌ - ошибки генерации
- [RETRY] ⏳ - ожидание перед повторной попыткой
- [RETRY] 🔄 - увеличение лимита токенов при incomplete
- [HINT] 💡 - подсказки для reasoning моделей
- [INFO] ✅ - восстановление TPM лимита

## Usage Examples

### Базовое использование
```python
from src.utils.llm_client import OpenAIClient

config = {
    'api_key': 'sk-...',
    'model': 'gpt-4.1-mini-2025-04-14',
    'tpm_limit': 120000,
    'tpm_safety_margin': 0.15,
    'max_completion': 4096,
    'timeout': 45,
    'max_retries': 6,
    'temperature': 0.7,
    'poll_interval': 7
}

client = OpenAIClient(config)

# Простой запрос
response_text, response_id, usage = client.create_response(
    "You are a helpful assistant",
    "What is the capital of France?"
)
print(f"Response: {response_text}")
print(f"Tokens used: {usage.total_tokens}")
```

### Цепочка запросов с контекстом
```python
# Первый запрос
text1, id1, usage1 = client.create_response(
    "You are a math tutor",
    "My name is Alice. What is 5 + 3?"
)

# Второй запрос помнит контекст
text2, id2, usage2 = client.create_response(
    "Continue being a math tutor",
    "What was my name?",
    previous_response_id=id1  # Явная передача контекста
)
# Ответ будет содержать "Alice"
```

### Работа с reasoning моделями
```python
config = {
    'api_key': 'sk-...',
    'model': 'o4-mini-2025-04-16',  # Reasoning модель
    'tpm_limit': 100000,
    'max_completion': 25000,  # Больше токенов для reasoning
    'reasoning_effort': 'medium',
    'reasoning_summary': 'auto',
    # temperature не указываем для reasoning моделей!
}

client = OpenAIClient(config)
response_text, _, usage = client.create_response(
    "Solve step by step",
    "What is 123 * 456?"
)
print(f"Reasoning tokens: {usage.reasoning_tokens}")
```

### Обработка JSON ответов с repair
```python
import json

try:
    response_text, _, _ = client.create_response(
        "Return a JSON object",
        "Create user object with name and age"
    )
    data = json.loads(response_text)
except json.JSONDecodeError:
    # Пробуем repair с тем же контекстом
    response_text, _, _ = client.repair_response(
        "Return a JSON object",
        "Create user object with name and age"
    )
    data = json.loads(response_text)
```

### Обработка ошибок
```python
try:
    response_text, response_id, usage = client.create_response(
        instructions="Complex task",
        input_data="Generate very long text..."
    )
except IncompleteResponseError as e:
    print(f"Response was incomplete: {e}")
    # Можно повторить с большим max_completion
except TimeoutError as e:
    print(f"Request timed out: {e}")
except openai.RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
```

### Мониторинг TPM
```python
# Проверка текущего состояния TPM
print(f"TPM remaining: {client.tpm_bucket.remaining_tokens}")
print(f"TPM reset at: {client.tpm_bucket.reset_time}")

# После запроса
response_text, _, usage = client.create_response(...)
print(f"Tokens used: {usage.total_tokens}")
print(f"TPM remaining after: {client.tpm_bucket.remaining_tokens}")
```