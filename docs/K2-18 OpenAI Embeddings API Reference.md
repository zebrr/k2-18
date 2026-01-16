# OpenAI Embeddings API Schema - Полный референс для K2-18

Документация по OpenAI Embeddings API для создания векторных представлений текста, которые могут быть легко обработаны алгоритмами машинного обучения.

## Аутентификация

API использует Bearer токены для аутентификации:

```
Authorization: Bearer YOUR_OPENAI_API_KEY
```

Дополнительные заголовки (опционально):

```
OpenAI-Organization: YOUR_ORG_ID
OpenAI-Project: YOUR_PROJECT_ID
```

## Эндпоинт для создания embeddings

### POST https://api.openai.com/v1/embeddings

Создает embedding для входного текста.

## Параметры Request Body

### Обязательные параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `input` | string или array | Текст для создания embedding. Может быть строкой или массивом строк. Максимум 8192 токена для каждого input. |
| `model` | string | ID модели для использования. Доступные модели: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |

### Опциональные параметры

| Параметр | Тип | Описание | Значение по умолчанию |
|----------|-----|----------|----------------------|
| `encoding_format` | string | Формат возвращаемых embeddings. Может быть `float` или `base64` | `float` |
| `dimensions` | integer | Количество измерений в результирующих векторах. Поддерживается только для `text-embedding-3` и более новых моделей | 1536 для text-embedding-3-small, 3072 для text-embedding-3-large |
| `user` | string | Уникальный идентификатор конечного пользователя. Используется для кэширования и предотвращения злоупотреблений | null |

## Доступные модели

| Модель | Размерность | Макс. токенов | Производительность MTEB |
|--------|-------------|---------------|-------------------------|
| `text-embedding-3-small` | 1536 | 8192 | 62.3% |
| `text-embedding-3-large` | 3072 | 8192 | 64.6% |
| `text-embedding-ada-002` | 1536 | 8192 | 61.0% |

## Примеры запросов

### Python

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-small",
    encoding_format="float"
)

print(response.data[0].embedding)
```

### Python с dimensions

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-large",
    dimensions=1024,
    encoding_format="float"
)

print(response.data[0].embedding)
```

### Обработка массива текстов

```python
from openai import OpenAI

client = OpenAI()

texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "OpenAI provides powerful language models"
]

response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

for i, embedding in enumerate(response.data):
    print(f"Embedding {i}: {embedding.embedding[:5]}...")  # Первые 5 значений
```

## Структура ответа

### Основной объект ответа

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        -0.006929283495992422,
        -0.005336422007530928,
        -4.547132266452536e-05,
        -0.024047505110502243,
        ...
      ]
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

### Описание полей ответа

#### Корневой объект

| Поле | Тип | Описание |
|------|-----|----------|
| `object` | string | Тип объекта, всегда `"list"` |
| `data` | array | Массив объектов embedding |
| `model` | string | Модель, использованная для создания embeddings |
| `usage` | object | Информация об использовании токенов |

#### Объект embedding

| Поле | Тип | Описание |
|------|-----|----------|
| `object` | string | Тип объекта, всегда `"embedding"` |
| `index` | integer | Индекс embedding в списке входных данных |
| `embedding` | array | Массив чисел с плавающей точкой, представляющий векторное представление |

#### Объект usage

| Поле | Тип | Описание |
|------|-----|----------|
| `prompt_tokens` | integer | Количество токенов во входном тексте |
| `total_tokens` | integer | Общее количество использованных токенов |

## Практические примеры

### Семантический поиск

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Создание embeddings для документов
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms to learn patterns",
    "OpenAI creates artificial intelligence models"
]

doc_embeddings = [get_embedding(doc) for doc in documents]

# Поиск по запросу
query = "programming languages"
query_embedding = get_embedding(query)

similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
best_match_idx = np.argmax(similarities)

print(f"Наилучшее совпадение: {documents[best_match_idx]}")
print(f"Сходство: {similarities[best_match_idx]:.4f}")
```

### Кластеризация текстов

```python
from sklearn.cluster import KMeans
import numpy as np

# Создание embeddings для текстов
texts = [
    "Python programming tutorial",
    "Machine learning basics",
    "Cooking recipe for pasta",
    "JavaScript web development",
    "Data science methods",
    "Italian cuisine guide"
]

embeddings = [get_embedding(text) for text in texts]
embeddings_matrix = np.array(embeddings)

# Кластеризация
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings_matrix)

# Результаты кластеризации
for i, (text, cluster) in enumerate(zip(texts, clusters)):
    print(f"Кластер {cluster}: {text}")
```

### Уменьшение размерности

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def normalize_l2(x):
    x = np.array(x)
    norm = np.linalg.norm(x)
    return x / norm if norm != 0 else x

# Создание полноразмерного embedding
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Testing dimensionality reduction"
)

full_embedding = response.data[0].embedding
print(f"Полная размерность: {len(full_embedding)}")

# Уменьшение до 256 измерений
reduced_embedding = full_embedding[:256]
normalized_embedding = normalize_l2(reduced_embedding)

print(f"Уменьшенная размерность: {len(normalized_embedding)}")
```

### Обработка больших текстов

```python
import tiktoken

def count_tokens(text, model="text-embedding-3-small"):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def chunk_text(text, max_tokens=8000):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

# Обработка длинного текста
long_text = "..." * 10000  # Ваш длинный текст

if count_tokens(long_text) > 8192:
    chunks = chunk_text(long_text)
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
    
    # Агрегация embeddings (например, усреднение)
    avg_embedding = np.mean(chunk_embeddings, axis=0)
    print(f"Создан усредненный embedding для {len(chunks)} частей")
else:
    embedding = get_embedding(long_text)
    print("Текст помещается в одном запросе")
```

## HTTP статус коды

| Код | Описание |
|-----|----------|
| 200 | Успешный запрос |
| 400 | Неверный запрос (неправильные параметры) |
| 401 | Неавторизованный доступ (неверный API ключ) |
| 429 | Превышен лимит запросов |
| 500 | Внутренняя ошибка сервера |

## Заголовки ответа

| Заголовок | Описание |
|-----------|----------|
| `openai-organization` | Организация, связанная с запросом |
| `openai-processing-ms` | Время обработки запроса в миллисекундах |
| `openai-version` | Версия API |
| `x-request-id` | Уникальный идентификатор запроса |
| `x-ratelimit-limit-requests` | Лимит запросов |
| `x-ratelimit-remaining-requests` | Оставшиеся запросы |
| `x-ratelimit-limit-tokens` | Лимит токенов |
| `x-ratelimit-remaining-tokens` | Оставшиеся токены |

## Ограничения

### Лимиты на вход

- **Максимальная длина текста**: 8192 токена на один input
- **Максимальный размер массива**: 2048 элементов в одном запросе
- **Максимальный TPM**: 350,000 токенов в минуту для embedding моделей третьего поколения

### Рекомендации по использованию

1. **Функция расстояния**: Рекомендуется использовать косинусное сходство
2. **Нормализация**: OpenAI embeddings нормализованы по длине = 1
3. **Кэширование**: Используйте параметр `user` для улучшения кэширования
4. **Размерность**: Используйте параметр `dimensions` для оптимизации затрат

## Лучшие практики

### Оптимизация производительности

```python
# Батчевая обработка
def batch_embeddings(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        embeddings.extend([item.embedding for item in response.data])
    return embeddings

# Кэширование embeddings
import hashlib
import pickle

def cached_embedding(text, cache_dir="embeddings_cache"):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_path = f"{cache_dir}/{text_hash}.pkl"
    
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        embedding = get_embedding(text)
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
        return embedding
```

### Обработка ошибок

```python
import time
import random

def get_embedding_with_retry(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            ).data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            raise e
```

For third-generation embedding models like `text-embedding-3-small`, use the `cl100k_base` encoding.