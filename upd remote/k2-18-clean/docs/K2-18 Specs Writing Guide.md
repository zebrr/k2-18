# ИНСТРУКЦИЯ ПО СОЗДАНИЮ ТЕХНИЧЕСКИХ СПЕК МОДУЛЕЙ K2-18 v2.0

Все спеки создаются на английском языке в UTF-8.

## Назначение

Технические спеки - это единый источник правды о реализации модулей. Они экономят контекст диалога, ускоряют понимание кода и служат справочником для разработки. Спека должна полностью заменить необходимость читать исходный код для понимания возможностей модуля.

## Принципы

1. **Полнота** - вся информация для использования модуля без чтения кода
2. **Практичность** - примеры использования, частые сценарии, граничные случаи
3. **Структурированность** - единообразная структура для быстрой навигации
4. **Актуальность** - спека обновляется вместе с кодом
5. **Без украшений** - технический стиль без эмодзи и лишних слов

---

## Классификация и именование

Спецификации хранятся в `docs/specs/`. Префикс отражает тип модуля:

| Тип | Модули | Префикс | Пример |
|-----|--------|---------|--------|
| **CLI** | `src/*.py` | `cli_` | `slicer.py` → `cli_slicer.md` |
| **Utils** | `src/utils/*.py` | `util_` | `llm_client.py` → `util_llm_client.md` |
| **Viz** | `viz/*.py` | `viz_` | `graph2metrics.py` → `viz_graph2metrics.md` |
| **Viz Frontend** | `viz/static/*.js` | `viz_` | `graph_core.js` → `viz_graph_core.md` |

---

## Базовая структура спеки

### 1. Заголовок и статус
```markdown
# module_name.md

## Status: READY|IN_PROGRESS|DRAFT

Краткое описание назначения модуля (1-3 предложения).
```

### 2. Обязательные разделы для всех модулей

#### Public API
Полное описание публичного интерфейса:
```markdown
### FunctionName(param1: type, param2: type = default) -> return_type
Описание функции.
- **Input**: 
  - param1 - описание параметра
  - param2 - описание с указанием default значения
- **Returns**: описание возвращаемого значения
- **Raises**: ExceptionType - условия возникновения

### ClassName
Описание класса и его назначения.

#### ClassName.__init__(params) -> None
Конструктор класса.
- **Input**: детальное описание параметров
- **Attributes**: список создаваемых атрибутов

#### ClassName.method(params) -> return_type
Описание метода.
```

#### Dependencies
```markdown
## Dependencies
- **Standard Library**: os, sys, json, pathlib
- **External**: openai>=1.0.0, numpy, faiss-cpu
- **Internal**: utils.config, utils.validation (или None)
```

#### Test Coverage
```markdown
## Test Coverage

- **test_module_name**: X tests
  - test_function_basic
  - test_function_edge_cases
  - test_error_handling
  
- **test_integration**: Y tests
  - test_full_workflow
  - test_performance
```

#### Usage Examples
```markdown
## Usage Examples

### Basic Usage
```python
from module import function

# Простой пример
result = function(data)
```

### Advanced Usage
```python
# Сложный сценарий с обработкой ошибок
try:
    result = complex_operation()
except SpecificError as e:
    handle_error(e)
```
```

### 3. Обязательные разделы для CLI утилит

#### CLI Interface
```markdown
## CLI Interface

### Usage
```bash
python -m src.module_name [options]
```

### Input Directory/Files
- **Source**: путь и описание входных данных
- **Formats**: поддерживаемые форматы

### Output Directory/Files  
- **Target**: путь и описание выходных данных
- **Naming**: правила именования файлов
```

#### Terminal Output
```markdown
## Terminal Output

### Output Format
Utility uses structured output with format:
```
[HH:MM:SS] TAG      | Message
```

### Progress Messages
```
[10:30:00] START    | Processing 157 files
[10:30:05] PROGRESS | ✅ 42/157 completed
[10:30:10] SUCCESS  | All files processed
```

### Error Messages
```
[10:30:00] ERROR    | ❌ File not found: data.txt
[10:30:00] WARNING  | ⚠️ Skipping empty file
```
```

#### Core Algorithm
```markdown
## Core Algorithm

Высокоуровневое описание алгоритма работы:
1. Загрузка данных
2. Валидация
3. Обработка
4. Сохранение результатов

Детали реализации в соответствующих методах.
```

#### Error Handling & Exit Codes
```markdown
## Error Handling & Exit Codes

### Exit Codes
- **0 (SUCCESS)** - successful execution
- **1 (CONFIG_ERROR)** - configuration errors
- **2 (INPUT_ERROR)** - input data errors
- **3 (RUNTIME_ERROR)** - processing errors
- **4 (API_LIMIT_ERROR)** - API limits exceeded
- **5 (IO_ERROR)** - file system errors

### Error Types
- **ConfigError** - неверные параметры конфигурации
- **InputError** - проблемы с входными данными
- **ProcessingError** - ошибки обработки

### Boundary Cases
- Пустые файлы → INPUT_ERROR с сообщением
- Превышение лимитов → ожидание и retry
- Частичный успех → сохранение промежуточных результатов
```

### 4. Дополнительные разделы (по необходимости)

#### Configuration (для модулей с конфигурацией)
```markdown
## Configuration

Section `[module]` in config.toml:

### Required Parameters
- **param_name** (type, constraints) - описание параметра
- **api_key** (str, non-empty) - API ключ

### Optional Parameters  
- **timeout** (int, >0, default=30) - таймаут в секундах

### Validation Rules
- Специальные правила валидации
- Взаимозависимости параметров
```

#### Internal Methods (для сложных модулей)
```markdown
## Internal Methods

### _important_internal_method(params) -> return_type
Описание важного внутреннего метода.
- **Purpose**: зачем нужен этот метод
- **Algorithm**: краткое описание алгоритма
- **Side effects**: побочные эффекты
```

#### Output Format (для модулей, создающих файлы)
```markdown
## Output Format

### JSON Structure
```json
{
  "field": "description",
  "nested": {
    "structure": "with examples"
  }
}
```

### CSV Format
```csv
column1,column2,column3
example,data,here
```
```

#### Performance Notes
```markdown
## Performance Notes

- **Memory**: ~2GB для 10K элементов
- **Speed**: ~1000 элементов/минуту
- **Complexity**: O(n log n) для основного алгоритма
- **Bottlenecks**: API запросы (до 100/мин)
- **Optimization**: батчевая обработка, кеширование
```

---

## Специфика для разных типов модулей

### Utils модули (util_*.md)
**Фокус**: API, алгоритмы, производительность

**Структура**:
1. Status и описание
2. Public API (детально)
3. Internal Methods (если критичны для понимания)
4. Configuration (если есть)
5. Test Coverage
6. Dependencies
7. Performance Notes
8. Usage Examples

### CLI утилиты (cli_*.md)
**Фокус**: пользовательский интерфейс, workflow, обработка ошибок

**Структура**:
1. Status и описание
2. CLI Interface
3. Terminal Output
4. Core Algorithm
5. Public Functions/Classes
6. Internal Methods (основные)
7. Output Format
8. Configuration
9. Error Handling & Exit Codes
10. Boundary Cases
11. Test Coverage
12. Dependencies
13. Performance Notes
14. Usage Examples

### Специализированные модули (так же util_*.md)
- **API клиенты**: детальное описание retry логики, rate limiting
- **Парсеры**: примеры входных/выходных данных, обработка ошибок
- **Валидаторы**: полный список правил валидации с примерами

---

## Требования к оформлению

### Язык и стиль
- Технический английский
- Настоящее время для описаний
- Краткие, но полные предложения
- Активный залог где возможно

### Форматирование
- **Bold** для важных терминов и названий параметров
- `code` для значений, имен функций, параметров
- ```python для блоков кода
- Таблицы для структурированных данных

### Описания API
```markdown
### function_name(param1: Type, param2: Optional[Type] = None) -> ReturnType
Краткое описание (1-2 предложения).
- **Input**: 
  - param1 (Type) - описание
  - param2 (Optional[Type]) - описание, default: None
- **Returns**: ReturnType - что возвращает
- **Raises**: 
  - ValueError - когда возникает
  - APIError - условия
- **Side effects**: изменения состояния (если есть)
- **Note**: важные замечания
```

---

## Алгоритм создания спеки

1. **Анализ кода**:
   - Изучить основной файл модуля
   - Выявить публичное API
   - Понять основные алгоритмы
   - Найти граничные случаи

2. **Анализ тестов**:
   - Понять сценарии использования
   - Выявить граничные случаи
   - Подсчитать покрытие

3. **Анализ зависимостей**:
   - От каких модулей зависит
   - Какие модули от него зависят
   - Внешние библиотеки

4. **Структурирование**:
   - Выбрать нужные разделы
   - Расположить в правильном порядке
   - Добавить специфичные разделы

5. **Написание**:
   - Начать с краткого описания
   - Детально описать API
   - Добавить примеры использования
   - Проверить полноту

6. **Валидация**:
   - Можно ли использовать модуль только по спеке?
   - Все ли граничные случаи описаны?
   - Актуальны ли примеры?

---

## Примеры образцовых спек

### Простой util модуль
`util_exit_codes.md` - минималистичная спека с полным покрытием

### Сложный util модуль  
`util_llm_client.md` - детальное описание сложного API с retry логикой

### CLI утилита базовая
`cli_slicer.md` - хороший баланс деталей и читаемости

### CLI утилита сложная
`cli_itext2kg.md` - полное описание сложного workflow с множеством состояний

### Модуль с конфигурацией
`util_config.md` - детальное описание параметров и валидации

---

## Контроль качества

### Хорошая спека

- Можно использовать модуль без чтения кода
- Понятна архитектура и основные решения
- Есть примеры для всех основных сценариев
- Описаны граничные случаи и ошибки
- Terminal output показывает реальный UX
- Ясно, какие параметры конфигурации нужны

### Плохая спека

- Требуется смотреть в код для понимания
- Нет примеров или они не работают
- Пропущены важные методы или параметры
- Не описаны типичные ошибки
- Неясно, как выглядит работа утилиты
- Дублирует код вместо объяснения концепций

---

## Поддержка актуальности

1. **При изменении кода** - сразу обновлять спеку
2. **При добавлении функций** - дополнять соответствующие разделы
3. **При исправлении багов** - обновлять граничные случаи
4. **При рефакторинге** - проверять актуальность описаний

