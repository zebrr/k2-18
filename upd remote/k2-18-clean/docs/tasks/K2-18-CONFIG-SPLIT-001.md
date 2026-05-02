# K2-18-CONFIG-SPLIT-001: Разделение [itext2kg] на независимые секции для concepts и graph

## References

Перед началом работы ОБЯЗАТЕЛЬНО изучи:

1. `docs/specs/util_config.md` — текущая спецификация модуля конфигурации
2. `docs/specs/cli_itext2kg_concepts.md` — спецификация утилиты concepts
3. `docs/specs/cli_itext2kg_graph.md` — спецификация утилиты graph
4. `src/utils/config.py` — текущая реализация
5. `src/config.toml` — текущий конфиг
6. `tests/test_config.py` — тесты конфига

## Context

### Текущее состояние

Утилиты `itext2kg_concepts.py` и `itext2kg_graph.py` используют общую секцию `[itext2kg]` в конфиге. Это не позволяет независимо настраивать параметры для каждой утилиты (разные модели, TPM лимиты, timeout, max_retries и т.д.).

### Цель

Разделить `[itext2kg]` на две независимые секции:
- `[itext2kg_concepts]` — для `itext2kg_concepts.py`
- `[itext2kg_graph]` — для `itext2kg_graph.py`

Каждая секция содержит полный набор параметров (полное дублирование, без наследования).

### Критерии успеха

1. Обе утилиты работают независимо с разными настройками
2. Все существующие тесты проходят (после адаптации)
3. Валидация конфига проверяет обе новые секции
4. API keys инжектируются в обе секции
5. Проверка `is_reasoning` работает для обеих секций

## Steps

### Шаг 1: Обновить `src/config.toml`

Заменить секцию `[itext2kg]` на две секции: `[itext2kg_concepts]` и `[itext2kg_graph]`.

**АЛГОРИТМ:**

1. Скопируй ПОЛНОСТЬЮ текущую секцию `[itext2kg]` со ВСЕМИ комментариями
2. Вставь копию сразу после оригинала
3. Переименуй первую копию в `[itext2kg_concepts]`
4. Переименуй вторую копию в `[itext2kg_graph]`
5. Удали оригинальную секцию `[itext2kg]`
6. В секции `[itext2kg_concepts]` — УДАЛИ параметр `auto_mentions_weight` (он специфичен только для graph)
7. В секции `[itext2kg_graph]` — оставь `auto_mentions_weight` как есть

**ВАЖНО**: 
- СОХРАНИ ВСЕ комментарии из оригинальной секции (там много полезных пояснений про модели, параметры и т.д.)
- Обе секции должны быть ИДЕНТИЧНЫ по структуре и комментариям, кроме `auto_mentions_weight`
- Порядок секций в файле: `[slicer]`, `[itext2kg_concepts]`, `[itext2kg_graph]`, `[dedup]`, `[refiner]`

### Шаг 2: Обновить `src/utils/config.py`

#### 2.1. Функция `_inject_env_api_keys()`

Добавить инжекцию API keys для новых секций:

```python
# itext2kg_concepts.api_key
if env_api_key:
    if "itext2kg_concepts" in config:
        current_key = config["itext2kg_concepts"].get("api_key", "")
        if not current_key or current_key.startswith("sk-..."):
            config["itext2kg_concepts"]["api_key"] = env_api_key

# itext2kg_graph.api_key
if env_api_key:
    if "itext2kg_graph" in config:
        current_key = config["itext2kg_graph"].get("api_key", "")
        if not current_key or current_key.startswith("sk-..."):
            config["itext2kg_graph"]["api_key"] = env_api_key
```

**УДАЛИТЬ** старый блок для `itext2kg`.

#### 2.2. Функция `_validate_config()`

Изменить `required_sections`:

```python
required_sections = ["slicer", "itext2kg_concepts", "itext2kg_graph", "dedup", "refiner"]
```

Изменить вызовы валидации:

```python
_validate_itext2kg_concepts_section(config["itext2kg_concepts"])
_validate_itext2kg_graph_section(config["itext2kg_graph"])
```

#### 2.3. Создать функцию `_validate_itext2kg_concepts_section()`

Скопировать `_validate_itext2kg_section()` и переименовать. Изменить все `"itext2kg."` на `"itext2kg_concepts."` в сообщениях об ошибках.

#### 2.4. Создать функцию `_validate_itext2kg_graph_section()`

Скопировать `_validate_itext2kg_section()` и переименовать. Изменить все `"itext2kg."` на `"itext2kg_graph."` в сообщениях об ошибках.

**ДОПОЛНИТЕЛЬНО** добавить валидацию `auto_mentions_weight` (опциональный параметр):

```python
# Validate optional auto_mentions_weight (only for graph)
if "auto_mentions_weight" in section:
    weight = section["auto_mentions_weight"]
    if not isinstance(weight, (int, float)) or not (0.0 <= weight <= 1.0):
        raise ConfigValidationError(
            "itext2kg_graph.auto_mentions_weight must be between 0.0 and 1.0"
        )
```

#### 2.5. Удалить старую функцию `_validate_itext2kg_section()`

После создания двух новых функций — удалить старую.

#### 2.6. Обновить проверку `is_reasoning` в `load_config()`

Заменить:

```python
if "itext2kg" in config:
    if "is_reasoning" not in config["itext2kg"]:
        raise ConfigValidationError(
            "Parameter 'is_reasoning' is required in [itext2kg] section"
        )
```

На:

```python
if "itext2kg_concepts" in config:
    if "is_reasoning" not in config["itext2kg_concepts"]:
        raise ConfigValidationError(
            "Parameter 'is_reasoning' is required in [itext2kg_concepts] section"
        )

if "itext2kg_graph" in config:
    if "is_reasoning" not in config["itext2kg_graph"]:
        raise ConfigValidationError(
            "Parameter 'is_reasoning' is required in [itext2kg_graph] section"
        )
```

#### 2.7. Обновить consistency warnings в `load_config()`

Заменить:

```python
for section in ["itext2kg", "refiner"]:
```

На:

```python
for section in ["itext2kg_concepts", "itext2kg_graph", "refiner"]:
```

### Шаг 3: Обновить `src/itext2kg_concepts.py`

Найти строку (около 96):

```python
self.config = config["itext2kg"]
```

Заменить на:

```python
self.config = config["itext2kg_concepts"]
```

### Шаг 4: Обновить `src/itext2kg_graph.py`

Найти строку (около 96):

```python
self.config = config["itext2kg"]
```

Заменить на:

```python
self.config = config["itext2kg_graph"]
```

### Шаг 5: Обновить тесты

#### 5.1. `tests/test_config.py`

**КРИТИЧНО**: Этот файл содержит много тестовых конфигов. Нужно:

1. В функции `get_minimal_valid_config()` — заменить `[itext2kg]` на ДВЕ секции `[itext2kg_concepts]` и `[itext2kg_graph]`

2. В классе `TestConfigLoading`, метод `test_load_valid_config()` — обновить тестовый конфиг

3. В классе `TestSlicerValidation` — все тестовые конфиги должны содержать обе новые секции

4. Класс `TestItext2kgValidation` — переименовать в `TestItext2kgConceptsValidation` или создать два класса. Обновить все тестовые конфиги и expected error messages.

5. Во ВСЕХ тестовых конфигах по всему файлу — заменить `[itext2kg]` на обе секции

6. В функции `get_config_with_override()` — обновить логику для работы с новыми именами секций

7. Параметризованные тесты `TestParametrizedValidation` — обновить параметры с `"itext2kg"` на `"itext2kg_concepts"` или `"itext2kg_graph"`

**ПОДХОД**: Используй поиск и замену, но АККУРАТНО проверяй контекст. Где тестируется конкретно itext2kg — выбирай одну из секций (concepts или graph).

#### 5.2. `tests/test_itext2kg_concepts.py`

В фикстуре `mock_config`:

```python
return {
    "itext2kg_concepts": {  # было "itext2kg"
        "model": "test-model",
        ...
    }
}
```

#### 5.3. `tests/test_itext2kg_graph.py`

В фикстуре `sample_config`:

```python
config = {
    "itext2kg_graph": {  # было "itext2kg"
        "model": "test-model",
        ...
    }
}
```

#### 5.4. `tests/test_itext2kg_graph_deduplication.py`

В фикстуре `processor`:

```python
config = {
    "itext2kg_graph": {  # было "itext2kg"
        ...
    },
    ...
}
```

#### 5.5. `tests/test_itext2kg_graph_postprocessing.py`

В фикстуре `mock_config`:

```python
return {
    "itext2kg_graph": {  # было "itext2kg"
        ...
    }
}
```

#### 5.6. `tests/test_itext2kg_graph_timeout.py`

В фикстуре `mock_config`:

```python
return {
    "itext2kg_graph": {  # было "itext2kg"
        ...
    },
    ...
}
```

### Шаг 6: Обновить спецификации

#### 6.1. `docs/specs/util_config.md`

- Обновить описание required_sections
- Добавить описания секций `[itext2kg_concepts]` и `[itext2kg_graph]`
- Удалить описание старой секции `[itext2kg]`
- Обновить примеры использования

#### 6.2. `docs/specs/cli_itext2kg_concepts.md`

В разделе "Configuration" заменить `[itext2kg]` на `[itext2kg_concepts]`

#### 6.3. `docs/specs/cli_itext2kg_graph.md`

В разделе "Configuration" заменить `[itext2kg]` на `[itext2kg_graph]`

## Testing

### Перед запуском тестов

```bash
cd C:\Users\Aski\Documents\AI\projects\semantic_graphs\k2-18
.venv\Scripts\activate
```

### Проверка качества кода (ВЫПОЛНИТЬ ПЕРВЫМ)

```bash
# Форматирование
black src/utils/config.py src/itext2kg_concepts.py src/itext2kg_graph.py
black tests/test_config.py tests/test_itext2kg_concepts.py tests/test_itext2kg_graph.py
black tests/test_itext2kg_graph_deduplication.py tests/test_itext2kg_graph_postprocessing.py
black tests/test_itext2kg_graph_timeout.py

# Сортировка импортов
isort src/utils/config.py src/itext2kg_concepts.py src/itext2kg_graph.py
isort tests/test_config.py tests/test_itext2kg_concepts.py tests/test_itext2kg_graph.py
isort tests/test_itext2kg_graph_deduplication.py tests/test_itext2kg_graph_postprocessing.py
isort tests/test_itext2kg_graph_timeout.py

# Линтинг
flake8 src/utils/config.py src/itext2kg_concepts.py src/itext2kg_graph.py
flake8 tests/test_config.py tests/test_itext2kg_concepts.py tests/test_itext2kg_graph.py
flake8 tests/test_itext2kg_graph_deduplication.py tests/test_itext2kg_graph_postprocessing.py
flake8 tests/test_itext2kg_graph_timeout.py

# Проверка типов
mypy src/utils/config.py --ignore-missing-imports
```

### Запуск тестов (ВЫПОЛНИТЬ ПОСЛЕ проверки качества)

```bash
# Тесты конфига (основные)
pytest tests/test_config.py -v

# Тесты concepts
pytest tests/test_itext2kg_concepts.py -v

# Тесты graph
pytest tests/test_itext2kg_graph.py -v
pytest tests/test_itext2kg_graph_deduplication.py -v
pytest tests/test_itext2kg_graph_postprocessing.py -v
pytest tests/test_itext2kg_graph_timeout.py -v

# Все тесты разом
pytest tests/test_config.py tests/test_itext2kg_concepts.py tests/test_itext2kg_graph.py tests/test_itext2kg_graph_deduplication.py tests/test_itext2kg_graph_postprocessing.py tests/test_itext2kg_graph_timeout.py -v
```

### Ожидаемый результат

- Все тесты PASSED
- Нет ошибок flake8
- Нет ошибок mypy (или только незначительные warnings)

### Ручная проверка загрузки конфига

```bash
python -c "from src.utils.config import load_config; c = load_config(); print('concepts model:', c['itext2kg_concepts']['model']); print('graph model:', c['itext2kg_graph']['model'])"
```

Должен вывести модели из обеих секций без ошибок.

## Deliverables

1. **Изменённые файлы**:
   - `src/config.toml` — новая структура с двумя секциями
   - `src/utils/config.py` — обновлённая валидация
   - `src/itext2kg_concepts.py` — использует `itext2kg_concepts`
   - `src/itext2kg_graph.py` — использует `itext2kg_graph`
   - `tests/test_config.py` — адаптированные тесты
   - `tests/test_itext2kg_concepts.py` — обновлённая фикстура
   - `tests/test_itext2kg_graph.py` — обновлённая фикстура
   - `tests/test_itext2kg_graph_deduplication.py` — обновлённая фикстура
   - `tests/test_itext2kg_graph_postprocessing.py` — обновлённая фикстура
   - `tests/test_itext2kg_graph_timeout.py` — обновлённая фикстура
   - `docs/specs/util_config.md` — обновлённая спецификация
   - `docs/specs/cli_itext2kg_concepts.md` — обновлённая секция Configuration
   - `docs/specs/cli_itext2kg_graph.md` — обновлённая секция Configuration

2. **Отчёт**: `docs/tasks/K2-18-CONFIG-SPLIT-001_REPORT.md` с:
   - Списком всех изменённых файлов
   - Результатами тестов (вывод pytest)
   - Результатами проверки качества кода
   - Любыми проблемами или отклонениями от задания

## Важные замечания

1. **НЕ МЕНЯЙ** логику работы утилит — только источник конфига
2. **НЕ УДАЛЯЙ** параметры — полное дублирование в обеих секциях
3. **СОХРАНЯЙ** комментарии в config.toml
4. **ПРОВЕРЯЙ** каждый тестовый конфиг — их много и они разбросаны по файлу
5. При ошибках в тестах — внимательно читай traceback, скорее всего где-то осталась старая секция `[itext2kg]`
