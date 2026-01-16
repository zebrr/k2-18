# OpenAI Responses API - Полный референс для K2-18 v2

## 1. CREATE Response Endpoint

### Endpoint

`POST /v1/responses`

### Заголовки аутентификации

```bash
Authorization: Bearer OPENAI_API_KEY
OpenAI-Organization: YOUR_ORG_ID (опционально)
OpenAI-Project: PROJECT_ID (опционально)
```

### Request Body Parameters

#### Обязательные параметры

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | ID модели для использования (например `o3-2025-04-16`)|
| `input` | string/array | Входной контент - строка или массив input объектов |

#### Основные параметры

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instructions` | string | null | Системные инструкции для модели |
| `tools` | array | [] | Список инструментов, доступных модели |
| `tool_choice` | string/object | "auto" | Управление выбором инструментов ("auto", "none", "required" или конкретный инструмент) |
| `parallel_tool_calls` | boolean | true | Разрешить параллельные вызовы инструментов |
| `temperature` | float | 1.0 | Управление случайностью (0.0 - 2.0) - игнорируется reasoning моделями |
| `top_p` | float | 1.0 | Nucleus sampling параметр - игнорируется reasoning моделями (лучше использовать только `temperature`) |
| `max_output_tokens` | integer | null | Максимальное количество токенов для генерации (рекомендуется минимум 25,000 для reasoning моделей) |
| `max_tool_calls` | integer | null | Максимальное количество вызовов встроенных инструментов |
| `reasoning` | object | null | **Конфигурация рассуждений для o-серии моделей** |
| `reasoning_effort` | string | null | **Уровень усилий рассуждения** (устаревшая альтернатива reasoning.effort) |
| `text` | object | null | Конфигурация формата текста |
| `background` | boolean | false | Запуск в фоновом режиме |
| `prompt` | object | null | Reference to a prompt template and its variables |
| `stream` | boolean | false | Потоковая передача ответа |
| `store` | boolean | true | Сохранять ли запрос для обучения модели: лучше `true`, `false` используется для режима zero‑retention |
| `service_tier` | string | "auto" | **Тип обработки**: `auto` наследует конфигурацию из проекта, `default` — стандартная цена и производительность, `flex` — гибкий режим, `priority` — приоритетный режим. Возвращённое значение может отличаться от запрошенного, если сервер решил использовать другой режим |
| `previous_response_id` | string | null | ID предыдущего ответа для продолжения диалога |
| `metadata` | object | {} | Пользовательские метаданные (максимум 16 пар ключ-значение) |
| `include` | array | null | Дополнительные поля для включения в ответ |
| `top_logprobs` | integer | null | Количество наиболее вероятных токенов для возврата (0-20) |
| `truncation` | string | "disabled" | Стратегия обрезки: `disabled` (по умолчанию; переполнение контекста приводит к ошибке 400), `auto` — в случае переполнения обрезает **ранние** входы в начале беседы, чтобы поместиться в окно контекста |
| `verbosity` | string | null | Constrains the verbosity of the model's response: "low", "medium", "high" |
| `conversation` | string/object | null | Идентификатор или объект беседы; входные и выходные элементы этого ответа автоматически добавляются к ней |
| `prompt_cache_key` | string | null | Ключ для кеширования запросов; помогает идентифицировать одинаковые промпты и **заменяет** параметр `user` |
| `prompt_cache_retention` | integer | 1 | Время хранения кеша (в часах, 1 или 24) для заданного `prompt_cache_key` |
| `safety_identifier` | string | null | Уникальный идентификатор пользователя, который рекомендуется хешировать; помогает обнаруживать злоупотребления |
| `stream_options` | object | null | Настройки потоковой передачи; применяются только когда `stream = true` |
| `user` | string | null | **Deprecated**: устаревший параметр. Вместо него используйте `prompt_cache_key` или `safety_identifier` |

#### Конфигурация Reasoning (для GPT-5 и o-серии моделей)

```json
{
  "reasoning": {
    // Уровень усилий рассуждения - контролирует глубину и количество токенов
    // "minimal" - очень мало reasoning токенов для максимально быстрого ответа
    // "low" - быстрые ответы с минимальными рассуждениями
    // "medium" - сбалансированная глубина (по умолчанию) 
    // "high" - глубокие рассуждения с детальными объяснениями
    "effort": "minimal|low|medium|high",
    
    // Формат резюме рассуждений
    // "auto" - система выбирает формат автоматически
    // "concise" - краткое резюме основных пунктов
    // "detailed" - подробное резюме всех шагов рассуждения
    // null - без резюме (по умолчанию)
    "summary": "auto|concise|detailed|null"
  }
}
```

#### **`verbosity`** - параметр ТОЛЬКО для GPT-5 серии

```json
{
  "text": {
    "format": {
      "type": "text"
    },
    // НОВЫЙ параметр verbosity для GPT-5, gpt-5-mini, gpt-5-nano
    "verbosity": "low|medium|high"   // по умолчанию null
  }
}
```

**Описание verbosity:**

- **"low"** - краткие, сжатые ответы, минимум многословности
- **"medium"** - сбалансированная детализация (по умолчанию)
- **"high"** - подробные ответы, больше объяснений, хорошо для аудитов и обучения

**Важно:** Параметр `verbosity` поддерживается только моделями **GPT-5 серии** (gpt-5, gpt-5-mini, gpt-5-nano).

#### Конфигурация Text Format

```json
{
  "text": {
    "format": {
      "type": "text"  // Единственное допустимое значение на данный момент (по умолчанию)
    }
  }
}
```

#### Конфигурация `prompt`

Ссылка на шаблон промпта и его переменные. Позволяет использовать переиспользуемые промпты из дашборда.
```json
{
  "prompt": {
    "id": "pmpt_abc123", // The unique identifier of the prompt template to use.
    "version": "2", // Optional version of the prompt template.
    "variables": { // Optional map of values to substitute in for variables in your prompt. The substitution values can either be strings, or other Response input types like images or files.
      "customer_name": "Jane Doe",
      "product": "40oz juice box"
    }
  }
}

```

#### Example Request

```python
import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.responses.create(
    model="gpt-4o",
    instructions="You are a coding assistant that talks like a pirate.",
    input="How do I check if a Python object is an instance of a class?",
)

print(response.output_text)
```

### Input Object Structure

Когда `input` является массивом, каждый объект может иметь:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Тип входа ("message", "input_text", "input_image", "input_file", "function_call_output", "computer_call_output", "mcp_approval_response") |
| `role` | string | Роль для типа message ("user", "assistant", "system", "developer") |
| `content` | string/array | Содержимое сообщения |
| `text` | string | Текстовое содержимое для типа input_text |
| `image_url` | string | URL изображения или base64 данные для типа input_image |
| `filename` | string | Имя файла для типа input_file |
| `file_data` | string | Base64 закодированные данные файла |
| `call_id` | string | ID вызова для выходов function/computer вызовов |
| `output` | string/object | Выходные данные для function вызовов |

### Tools Configuration

#### Function Tool (пользовательские функции)

```json
{
  "type": "function",
  "function": {
    "name": "function_name",
    "description": "Описание функции",
    "parameters": {
      "type": "object",
      "properties": {
        // JSON Schema для параметров
      },
      "required": ["param1", "param2"]
    }
  }
}
```

## 2. GET Response Endpoint

### Endpoint

`GET /v1/responses/{response_id}`

### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response_id` | string | Yes | ID ответа для получения |

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `include` | array | No | Дополнительные поля для включения в ответ |
| `include_obfuscation` | boolean | No | **Новое**: если `true`, в каждое delta‑событие при потоковой передаче вставляются случайные символы для защиты от атак побочного канала |
| `starting_after` | integer | No | **Новое**: номер последовательности, после которого возвращаются элементы; заменяет устаревший `after` |
| `stream` | boolean | No | Потоковая передача данных ответа |

### Дополнительные Endpoints

- `GET /v1/responses/{response_id}/input_items` — список входных элементов. Допустимые параметры:
   * `after` (string) — ID элемента, после которого начинать выдачу
   * `include` (array) — дополнительные поля для включения
   * `limit` (integer, 1–100, по умолчанию 20) — максимальное количество возвращаемых элементов
   * `order` (string: `asc` или `desc`, по умолчанию `desc`) — направление сортировки элементов
- `DELETE /v1/responses/{response_id}` - Удаление ответа
- `POST /v1/responses/{response_id}/cancel` - Отмена ответа
- `POST /v1/responses/input_tokens` — возвращает количество входных токенов для будущего запроса без запуска модели. В теле запроса принимаются те же поля, что и для `POST /v1/responses` (`conversation`, `input`, `instructions`, `tools` и др.). Ответ содержит поле `input_tokens`

## 3. Response Object Structure

### Основной объект ответа

```json
{
  "id": "string",
  "object": "response",
  "created_at": "integer (unix timestamp)",
  "model": "string",
  "status": "completed|in_progress|queued|failed|cancelled|incomplete",
  "output": ["array of output objects"],
  "usage": {
    "total_tokens": "integer",
    "input_tokens": "integer", 
    "output_tokens": "integer",
    "input_tokens_details": "object",
    "output_tokens_details": {
      "reasoning_tokens": "integer"
    },
    "completion_tokens": "integer|null",
    "prompt_tokens": "integer|null",
    "completion_tokens_details": "object|null",
    "prompt_tokens_details": "object|null"
  },
  "error": {
    "code": "string",
    "message": "string",
    "type": "string",
    "param": "string|null"
  },
  "incomplete_details": {
    "reason": "string"
  },
  "metadata": "object",
  "prompt_cache_key": "string|null",    // **Новое**: ключ, использованный для кеширования запросов
  "prompt_cache_retention": "integer|null",    // **Новое**: период хранения кеша (1 или 24 часа)
  "safety_identifier": "string|null",  // **Новое**: хешированный идентификатор пользователя, используемый для проверки политики
  "output_text": "string|null",  // **Новое**: удобное поле для чтения финального текста ответа (дубликат текста из первого сообщения)
  "instructions": "string|null",
  "temperature": "float",
  "top_p": "float",
  "max_output_tokens": "integer|null",
  "max_tool_calls": "integer|null",
  "tool_choice": "string|object|null",
  "tools": ["array"],
  "parallel_tool_calls": "boolean|null",
  "previous_response_id": "string|null",
  "reasoning": "object|null",
  "reasoning_effort": "string|null",
  "text": "object|null",
  "truncation": "string|null",
  "service_tier": "string|null",
  "background": "boolean|null",
  "user": "string|null" // для обратной совместимости, считается устаревшим; следует использовать prompt_cache_key или safety_identifier
}
```

### Status Codes (детально)

- `"completed"` - Генерация ответа завершена успешно
- `"failed"` - Генерация ответа завершилась ошибкой
- `"in_progress"` - Ответ генерируется
- `"cancelled"` - Генерация ответа была отменена
- `"queued"` - Ответ находится в очереди на обработку  
- `"incomplete"` - Ответ завершен, но неполный

### Output Object Types

#### Message Output

```json
{
  "id": "string",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "output_text|output_image",
      "text": "string",
      "annotations": ["array"],
      "logprobs": "object|null"
    }
  ],
  "status": "string|null"
}
```

#### Function Call Output

```json
{
  "id": "string", 
  "type": "function_call",
  "name": "string",
  "arguments": "string (JSON)",
  "call_id": "string"
}
```

#### Computer Call Output

```json
{
  "id": "string",
  "type": "computer_call", 
  "call_id": "string",
  "action": {
    "type": "string",
    "parameters": "object"
  },
  "pending_safety_checks": ["array"]
}
```

#### Web Search Output

```json
{
  "id": "string",
  "type": "web_search_call",
  "status": "string"
}
```

#### Reasoning Output

```json
{
  "id": "string",
  "type": "reasoning",
  "summary": ["array of reasoning summary objects"]
}
```

## 4. Delete a model response

Deletes a model response with the given ID:
`delete https://api.openai.com/v1/responses/{response_id}`

Path parameters: `response_id` | string Required | The ID of the response to delete.
```python
from openai import OpenAI
client = OpenAI()

response = client.responses.delete("resp_123")
print(response)
```

Returns a success message.
```json
{
  "id": "resp_6786a1bec27481909a17d673315b29f6",
  "object": "response",
  "deleted": true
}
```

### Response Deletion Example

Представим цепочку разговора:
`Ответ A → Ответ B → Ответ C → Ответ D`

Если удалить `Ответ B`, то:
1. `Ответ C` больше не может получить доступ к контексту `Ответа A`
2. При вызове `/v1/responses/{response_C_id}/input_items` система сможет восстановить историю только начиная с `Ответа C`
3. Весь предыдущий контекст теряется для всех последующих ответов в цепочке
4. Что показывает API при разорванной цепочке

```python
# До удаления - полная цепочка доступна
response_d_history = client.responses.input_items.list("resp_D")
# Возвращает: [Input_A, Response_A, Input_B, Response_B, Input_C, Response_C, Input_D]

# После удаления Response_B
response_d_history = client.responses.input_items.list("resp_D") 
# Возвращает только: [Input_C, Response_C, Input_D]
# Input_A, Response_A, Input_B - ПОТЕРЯНЫ!
```


## 5. Rate Limiting Check

* `client.responses.create(...)` (or any other .create) returns a pre-parsed `Response` object; SDK discards the low-level `httpx.Response`
* Headers still come from the server (e.g. `x-ratelimit-remaining-requests`, `x-ratelimit-reset-tokens`, etc.)
* You need to prefix `with_raw_response` before `create` to get a `raw` object - this is `LegacyAPIResponse`, it has a `.headers` attribute (type `httpx.Headers`)
* After that, you need to call `parsed = raw.parse()` and work with the usual `Response` object

```Python
from openai import OpenAI
client = OpenAI()

# 1) Делаем обычный запрос, но через with_raw_response
raw = client.responses.with_raw_response.create(
        model="gpt-4o",
        input="ping"
)

# 2) Читаем лимиты
headers = raw.headers
tpm_limit  = int(headers.get("x-ratelimit-limit-tokens", 0))
tpm_left   = int(headers.get("x-ratelimit-remaining-tokens", 0))
reset_str  = headers.get("x-ratelimit-reset-tokens", "0ms")

# 3) При необходимости превращаем '820ms' → секунд
reset_sec = int(reset_str.rstrip("ms")) / 1000

# 4) Когда нужны обычные данные ответа
parsed = raw.parse()      # ← тот же объект, что вернул бы .create()
print(parsed.output_text)
```

**Note**:
* `headers` - это `httpx.Headers`, поэтому `.get()` работает так же, как у `dict`
* TPM считается по `max(input_tokens, max_tokens)`
* 429 до лимита - бывают под-секундные ограничения; смотрите `x-request-time` и текущие остатки

### Rate Limiting Headers

```
x-ratelimit-limit-requests - Лимит запросов (в минуту)
x-ratelimit-limit-tokens - Лимит токенов (в минуту)
x-ratelimit-remaining-requests - Оставшиеся запросы до лимита
x-ratelimit-remaining-tokens - Оставшиеся токены до лимита
x-ratelimit-reset-requests - Время сброса лимита запросов, для RPM/TPM время приходит в мс; просто делите на 1000, ждём reset sec и шлём снова
x-ratelimit-reset-tokens - Время сброса лимита токенов, для RPM/TPM время приходит в мс; просто делите на 1000, ждём reset sec и шлём снова
```

### Troubleshooting Headers

```
x-request-id // ID запроса для диагностики лежит в `raw.headers["x-request-id"]` или `response._request_id`
```

## 6. Usage Examples

### Базовый запрос для iText2KG

```json
{
  "model": "gpt-4.1-mini",
  "input": "Проанализируй этот учебный текст и извлеки концепции...",
  "instructions": "Ты агент для извлечения концепций из образовательных текстов...",
  "temperature": 0.1,
  "max_output_tokens": 4096,
  "store": false
}
```

### Запрос с reasoning моделью

```json
{
  "model": "o4-mini",
  "input": "Сложный анализ текста с выявлением связей между концепциями...",
  "reasoning": {
    "effort": "medium",
    "summary": "concise"
  },
  "max_output_tokens": 25000,
  "truncation": "auto",
  "store": false
}
```

### Многоэтапный диалог с context

```json
{
  "model": "gpt-4.1",
  "input": "Продолжи анализ предыдущего фрагмента...",
  "previous_response_id": "resp_abc123",
  "instructions": "Учитывай контекст предыдущего анализа...",
  "max_output_tokens": 4096
}
```

### Запрос с пользовательскими функциями

```json
{
  "model": "gpt-4.1-mini",
  "input": "Извлеки концепции и проверь их в базе знаний",
  "tools": [
    {
      "type": "function", 
      "function": {
        "name": "search_existing_concepts",
        "description": "Поиск существующих концепций в базе знаний",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"}
          },
          "required": ["query"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": false
}
```

### Обычный ответ

**Note**: Аlways true for regular models:
* output[0] = message
* output[0].content[0].text = text

```json
{
  "id": "resp_6871a162606c819b82af1121ff8e005605da467959369fa7",
  "object": "response",
  "created_at": 1752277346,
  "status": "completed",
  "background": false,
  "error": null,
  "incomplete_details": null,
  "instructions": "Answer briefly",
  "max_output_tokens": 100,
  "max_tool_calls": null,
  "model": "gpt-4.1-nano-2025-04-14",
  "output": [
    {
      "id": "msg_6871a16313a8819b98d9530d6b6452f605da467959369fa7",
      "type": "message",
      "status": "completed",
      "content": [
        {
          "type": "output_text",
          "annotations": [],
          "logprobs": [],
          "text": "2 + 2 = 4"
        }
      ],
      "role": "assistant"
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "service_tier": "default",
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_logprobs": 0,
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 20,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 8,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 28
  },
  "user": null,
  "metadata": {}
}
```

### Ответ с reasoning моделью

**Note**: Аlways true for reasoning models:
* output[0] = reasoning
* output[1] = message
* output[1].content[0].text = text

```json
{
  "id": "resp_6871a163a628819a8add54ce1417a43009b959bef9cc7999",
  "object": "response",
  "created_at": 1752277347,
  "status": "completed",
  "background": false,
  "error": null,
  "incomplete_details": null,
  "instructions": "Answer briefly",
  "max_output_tokens": 100,
  "max_tool_calls": null,
  "model": "o4-mini-2025-04-16",
  "output": [
    {
      "id": "rs_6871a16405c4819abd5950adb5fb00a609b959bef9cc7999",
      "type": "reasoning",
      "summary": []
    },
    {
      "id": "msg_6871a164331c819aaf2948d7e36d17c609b959bef9cc7999",
      "type": "message",
      "status": "completed",
      "content": [
        {
          "type": "output_text",
          "annotations": [],
          "logprobs": [],
          "text": "2+2 = 4."
        }
      ],
      "role": "assistant"
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": "low",
    "summary": null
  },
  "service_tier": "default",
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_logprobs": 0,
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 19,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 13,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 32
  },
  "user": null,
  "metadata": {}
}
```

## 7. Best Practices

- **Резервируйте минимум 25,000 токенов** для reasoning моделей
- **Мониторьте usage объект** в ответах
- **Логируйте request ID** для диагностики: `x-request-id` header
- **Отслеживайте rate limits** через headers
- **Используйте metadata** для маркировки запросов по типу задач (если надо)

### Обработка ошибок

```python
def handle_response(response):
    if response.status == "failed":
        print(f"Error: {response.error.message}")
        return None
    elif response.status == "incomplete":
        print(f"Incomplete: {response.incomplete_details.reason}")
        # Возможно, retry с большим max_output_tokens
    return response
```

### Конфигурация для разных задач

```python
# Для извлечения концепций
extraction_config = {
    "model": "gpt-4.1-mini",
    "temperature": 0.1,
    "max_output_tokens": 4096,
    "store": False
}

# Для анализа связей (более сложная задача)
relation_config = {
    "model": "o4-mini", 
    "reasoning": {"effort": "medium"},
    "max_output_tokens": 15000,
    "store": False
}
```

## 8. Notes for LLM Context Usage

* **Context**: Сохраняйте контекст через `previous_response_id`  
* **Parameter Validation**: Всегда проверяйте обязательные параметры (`model`, `input`) перед запросами
* **Tool Selection**: Используйте `tool_choice` для управления инструментами - "auto" позволяет модели решать, "none" отключает инструменты
* **Model Selection**: Выбирайте модели в зависимости от сложности задач и требований к рассуждениям
* **Input Flexibility**: API принимает как простые строковые входы, так и сложные мультимодальные входы
* **Error Handling**: Проверяйте поле `status` и объект `error` для неудачных ответов
* **Token Management**: Отслеживайте объект `usage` для контроля потребления токенов
* **Conversation Context**: Используйте `previous_response_id` для поддержания непрерывности диалога
* **Zero Data Retention**: `store` по умолчанию `true`; значение `false` отключает запись вашего запроса и ответа для последующего обучения моделей

### Managing the context window

It's important to ensure there's enough space in the context window for reasoning tokens when creating responses. Depending on the problem's complexity, the models may generate anywhere from a few hundred to tens of thousands of reasoning tokens. The exact number of reasoning tokens used is visible in the usage object of the response object, under `output_tokens_details`:

```json
{
  "usage": {
    "input_tokens": 75,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 1186,
    "output_tokens_details": {
      "reasoning_tokens": 1024
    },
    "total_tokens": 1261
  }
}
```

### Controlling costs

If you're managing context manually across model turns, you can discard older reasoning items unless you're responding to a function call, in which case you must include all reasoning items between the function call and the last user message.

To manage costs with reasoning models, you can limit the total number of tokens the model generates (including both reasoning and final output tokens) by using the `max_output_tokens parameter`.

### Allocating space for reasoning

If the generated tokens reach the context window limit or the `max_output_tokens` value you've set, you'll receive a response with a `status` of `incomplete` and `incomplete_details` with `reason` set to `max_output_tokens`. This might occur before any visible output tokens are produced, meaning you could incur costs for input and reasoning tokens without receiving a visible response.

To prevent this, ensure there's sufficient space in the context window or adjust the max_output_tokens value to a higher number. OpenAI recommends reserving at least 25,000 tokens for reasoning and outputs when you start experimenting with these models. As you become familiar with the number of reasoning tokens your prompts require, you can adjust this buffer accordingly.

#### Handling incomplete responses

```python
from openai import OpenAI

client = OpenAI()

prompt = """
Write a bash script that takes a matrix represented as a string with 
format '[1,2],[3,4],[5,6]' and prints the transpose in the same format.
"""

response = client.responses.create(
    model="o4-mini",
    reasoning={"effort": "medium"},
    input=[
        {
            "role": "user", 
            "content": prompt
        }
    ],
    max_output_tokens=300,
)

if response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens":
    print("Ran out of tokens")
    if response.output_text:
        print("Partial output:", response.output_text)
    else: 
        print("Ran out of tokens during reasoning")
```

## 9. Notes for Prompt caching

Model prompts often contain repetitive content, like system prompts and common instructions. OpenAI routes API requests to servers that recently processed the same prompt, making it cheaper and faster than processing a prompt from scratch. This can reduce latency by up to 80% and cost by up to 75%. Prompt Caching works automatically on all your API requests (no code changes required) and has no additional fees associated with it.

* Structure prompts with static or repeated content at the beginning and dynamic, user-specific content at the end.
* Use `prompt_cache_key` and `safety_identifier` if needed across requests that share common prompts.
* **Monitor your cache performance metrics**, including cache hit rates, latency, and the proportion of tokens cached, to refine your strategy.
* **Maintain a steady stream of requests** with identical prompt prefixes to minimize cache evictions and maximize caching benefits.

### Requirements

Caching is available for prompts containing 1024 tokens or more, with cache hits occurring in increments of 128 tokens. Therefore, the number of cached tokens in a request will always fall within the following sequence: 1024, 1152, 1280, 1408, and so on, depending on the prompt's length.

All requests, including those with fewer than 1024 tokens, will display a `cached_tokens` field of the `usage.prompt_tokens_details` For requests under 1024 tokens, `cached_tokens` will be zero.

```json
"usage": {
  "prompt_tokens": 2006,
  "completion_tokens": 300,
  "total_tokens": 2306,
  "prompt_tokens_details": {
    "cached_tokens": 1920
  },
  "completion_tokens_details": {
    "reasoning_tokens": 0,
    "accepted_prediction_tokens": 0,
    "rejected_prediction_tokens": 0
  }
}
```

---

**Референс оптимизирован для проекта iText2KG и содержит все необходимые детали для работы с OpenAI Responses API.**