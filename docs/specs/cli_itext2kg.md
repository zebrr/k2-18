# cli_itext2kg.md

## Status: READY

CLI утилита для инкрементального построения графа знаний из образовательных текстов. Обрабатывает слайсы последовательно, отправляет в LLM с сохранением контекста через previous_response_id. Включает механизмы восстановления после ошибок и сохранение промежуточных результатов.

## CLI Interface

**Запуск:**
```bash
python -m src.itext2kg
```

**Входные данные:**
- `/data/staging/*.slice.json` - слайсы от slicer.py

**Выходные данные:**
- `/data/out/ConceptDictionary.json` - словарь концептов
- `/data/out/LearningChunkGraph_raw.json` - граф знаний
- `/logs/itext2kg_YYYY-MM-DD_HH-MM-SS.log` - детальные логи
- `/logs/{slice_id}_bad.json` - проблемные ответы LLM (при ошибках)
- `/logs/*_temp_*.json` - временные дампы (при критических ошибках)

**Коды завершения:**
- 0 (SUCCESS) - успешная обработка
- 1 (CONFIG_ERROR) - ошибки конфигурации
- 2 (INPUT_ERROR) - нет слайсов в staging
- 3 (RUNTIME_ERROR) - все слайсы failed или критическая ошибка
- 4 (API_LIMIT_ERROR) - исчерпаны лимиты API
- 5 (IO_ERROR) - ошибки записи файлов

## Core Algorithm

1. **Загрузка слайсов** из staging в лексикографическом порядке
2. **Последовательная обработка** с сохранением previous_response_id:
   - Форматирование входных данных (ConceptDictionary + Slice)
   - Вызов LLM через Responses API
   - Валидация и парсинг ответа
   - Repair-reprompt при ошибках (1 попытка)
   - Инкрементальное обновление структур данных
   - **Промежуточная валидация** после каждого слайса
   - Автоматическое добавление MENTIONS edges для всех обработанных Chunks
3. **Обработка сбоев** с graceful degradation:
   - Продолжение при частичных сбоях
   - Сохранение временных дампов при критических ошибках
4. **Финальная валидация** с использованием промежуточной валидации (допускает дубликаты концептов)
5. **Сохранение результатов** в output

## Terminal Output

Утилита использует структурированный вывод прогресса с унифицированным форматом:
```
[HH:MM:SS] TAG      | Данные
```

### Формат вывода прогресса

**START - начало обработки:**
```
[10:30:00] START    | 157 slices | model=o4-mini-2025-04-16 | tpm=100k
```

**SLICE - успешная обработка слайса:**
```
[10:30:05] SLICE    | ✅ 001/157 | tokens_used=12.35k | tokens_current=1.23k | 5s | concepts=23 | nodes=156 | edges=287
[10:30:12] SLICE    | ✅ 002/157 | tokens_used=112.34k | tokens_current=11.23k incl. reasoning=567 | 8s | concepts=25 | nodes=163 | edges=301
```

**REPAIR - попытка исправления ошибок валидации:**
```
[10:30:45] REPAIR   | 🔧 Attempting to fix JSON validation error...
[10:30:45] REPAIR   | 📝 Adding clarification to prompt and retrying...
[10:30:50] REPAIR   | ✅ JSON validation fixed successfully!
```

**ERROR - ошибки обработки:**
```
[10:30:45] ERROR    | ❌ 042/157 | slice_042 | JSON validation failed after repair
[10:30:45] ERROR    | ❌ Incremental validation failed for slice_042
[10:30:45] ERROR    | 📋 Error: Дублированный ID узла (Assessment): algo101:q:1234:0...
[10:31:02] ERROR    | ⚠️ RateLimitError | waiting for retry...
[10:31:15] ERROR    | ⚠️ APIError | slice slice_055
```

**FAILED - критические ошибки:**
```
[10:45:30] FAILED   | ❌ All slices failed processing
[10:45:30] FAILED   | ❌ Critical error: Connection timeout...
[10:45:30] FAILED   | ❌ Validation failed: Invalid graph structure...
```

**SAVING - сохранение временных файлов:**
```
[10:45:30] SAVING   | 💾 Attempting to save empty structures...
[10:45:30] SAVING   | 💾 Emergency dump of current state...
[10:45:30] SAVING   | 💾 Attempting to save partial results...
```

**INFO - информационные сообщения:**
```
[10:45:31] INFO     | Check /logs/ for temporary files and diagnostics
```

**SUCCESS - успешное завершение:**
```
[10:45:30] SUCCESS  | ✅ Results saved to /data/out/
                    | - ConceptDictionary.json
                    | - LearningChunkGraph_raw.json
```

**END - завершение работы:**
```
[10:45:30] END      | Done | slices=157 | time=15m 30s
```

### Логирование в файлы

В файлах логов используется JSON Lines формат для структурированного анализа:
- **INFO уровень**: основные события обработки
- **DEBUG уровень**: полные промпты и ответы LLM (при log_level=debug)
- **ERROR уровень**: ошибки валидации и API

Ошибки также выводятся в консоль через стандартный logger:
```
[10:30:00] ERROR    | No slice files found in staging directory
```

## Public Classes

### ProcessingStats
Статистика обработки слайсов.
- **Attributes**: total_slices, processed_slices, failed_slices, total_concepts, total_nodes, total_edges, total_tokens_used, start_time

### SliceData
Данные одного слайса.
- **Attributes**: id, order, source_file, slug, text, slice_token_start, slice_token_end

### SliceProcessor
Основной класс обработки.
- **__init__(config)** - инициализация с конфигурацией
- **run()** - основной метод запуска обработки

## Internal Methods

### SliceProcessor._format_tokens(tokens)
Форматирование количества токенов в читаемый вид.
- **Input**: tokens - количество токенов
- **Returns**: строка вида "123", "45.61k", "1.22M"
- **Features**:
  - Числа < 1000: без изменений ("123")
  - Тысячи (1K-999K): форматируются как "45.61k" с двумя знаками после запятой
  - Миллионы (1M+): форматируются как "1.22M" с двумя знаками после запятой

### SliceProcessor._process_chunk_nodes(new_nodes)
Обработка узлов типа Chunk и Assessment с проверкой дубликатов.
- **Input**: new_nodes - список новых узлов из патча
- **Returns**: список узлов для добавления в граф
- **Features**:
  - Для Chunk: сравнение длины текста, обновление если новый длиннее
  - Для Assessment: игнорирование дубликатов с логированием предупреждения
  - Для остальных типов: добавление без изменений

### SliceProcessor._validate_edges(edges)
Валидация рёбер с проверкой существования узлов и фильтрацией дубликатов.
- **Input**: edges - список рёбер для проверки
- **Returns**: список валидных рёбер
- **Features**:
  - Проверка существования source/target узлов (включая концепты из ConceptDictionary)
  - Отбрасывание PREREQUISITE self-loops
  - Проверка весов в диапазоне [0,1]
  - **Фильтрация дублированных рёбер**:
    - Проверка против существующих рёбер в графе
    - Проверка дубликатов внутри текущего патча
    - Дубликаты определяются по комбинации (source, target, type)
    - Вес игнорируется при определении дубликата
  - Битые рёбра отбрасываются с WARNING логированием
  - Дублированные рёбра отбрасываются с INFO логированием

### SliceProcessor._save_bad_response(slice_id, original_response, error, repair_response=None)
Сохранение некорректного ответа LLM для анализа.
- **Input**: slice_id, оригинальный ответ, описание ошибки, repair ответ (если был)
- **Output**: файл `/logs/{slice_id}_bad.json` с полной информацией

### SliceProcessor._save_temp_dumps(reason)
Сохранение временных дампов при критических ошибках.
- **Input**: reason - причина сохранения (interrupted, validation_failed, io_error, all_failed, critical_error, validation_error_slice_{id})
- **Output**: 
  - ConceptDictionary_temp_{reason}_{timestamp}.json
  - LearningChunkGraph_temp_{reason}_{timestamp}.json
  - processing_stats_{reason}_{timestamp}.json

### SliceProcessor._process_single_slice(slice_file)
Обработка одного слайса с полной обработкой ошибок.
- **Returns**: True при успехе, False при неудаче
- **Features**: 
  - repair-reprompt при невалидном JSON
  - сохранение bad responses
  - промежуточная валидация после применения патча
  - graceful error handling

### SliceProcessor._add_mentions_edges(chunk_nodes)
Автоматически добавляет MENTIONS edges от Chunks к Concepts на основе текстового поиска.
- **Input**: chunk_nodes - список узлов типа Chunk для обработки
- **Returns**: количество добавленных MENTIONS edges
- **Algorithm**:
  - Для каждого Chunk ищет упоминания всех концептов из ConceptDictionary
  - Поиск выполняется по term.primary и всем term.aliases
  - **Правила поиска**:
    - Full word matches only (использует regex `\b` границы слов)
    - Case-insensitive (сравнение в нижнем регистре)
    - Exact forms only (без морфологии, "стеки" ≠ "стек")
  - Избегает дублирования существующих MENTIONS edges
  - Все MENTIONS edges имеют weight=1.0

### SliceProcessor._process_llm_response(response_text, slice_id)
Обработка и валидация ответа LLM с предварительной очисткой известных проблем.
- **Input**: response_text - сырой ответ от LLM, slice_id - ID текущего слайса
- **Returns**: (success, parsed_data) - успех и распарсенные данные или None
- **Features**:
  - **Предварительная очистка HTML атрибутов**:
    - Исправляет паттерны типа `href='\"url\"'` → `href="url"`
    - Исправляет паттерны типа `src="'url'"` → `src="url"` 
    - Применяется ко всем ответам перед парсингом JSON
    - Обрабатывает атрибуты: href, src, target, action, name, frameborder, width, height, align
    - Использует регулярные выражения для замены
  - Парсинг очищенного JSON
  - Валидация структуры ответа (наличие concepts_added и chunk_graph_patch)
  - Базовая валидация по схемам ConceptDictionary и LearningChunkGraph
  - При ошибке логирует детали для отладки

## Key Features

### Управление контекстом
- Автоматическое управление previous_response_id
- Сохранение контекста между слайсами до 128K токенов
- При retry и repair используется тот же previous_response_id

### Инкрементальное обновление ConceptDictionary
- Новые концепты добавляются целиком с автоматической очисткой case-insensitive дубликатов в aliases
- Для существующих концептов:
  - Обновляются только aliases с проверкой case-insensitive уникальности
  - Primary термин и definition сохраняются от первого появления
  - Новые aliases добавляются только если их lowercase версии еще нет в списке
- Создание узлов типа Concept для новых концептов
- **Case-insensitive логика**:
  - При добавлении нового концепта: удаляются дубликаты aliases (например, ["Stack", "stack"] → ["Stack"])
  - При обновлении существующего: новые aliases проверяются case-insensitive против существующих
  - Сохраняется первое вхождение каждого уникального alias с его оригинальным регистром

**Примечание:** Система автоматически обеспечивает case-insensitive уникальность aliases внутри каждого концепта, что предотвращает ошибки валидации при инкрементальной обработке. LLM может возвращать дубликаты типа ["Brute Force", "brute force"], но система сохранит только первый вариант.

### Обработка дубликатов узлов
- **Chunk узлы**: при одинаковых ID сохраняется более длинная версия текста
- **Assessment узлы**: дубликаты игнорируются с предупреждением
- **Concept узлы**: дубликаты НЕ предотвращаются (семантическая дедупликация в dedup.py)
- Все изменения и предупреждения логируются

### Промежуточная валидация
- Выполняется после обработки каждого слайса
- Использует `validate_graph_invariants_intermediate()`:
  - Проверяет уникальность ID для Chunk и Assessment
  - НЕ проверяет уникальность ID для Concept (допустимы дубликаты)
  - Проверяет все остальные инварианты графа
- При ошибке валидации:
  - Слайс помечается как failed
  - Сохраняется временное состояние для отладки
  - Обработка продолжается со следующего слайса

### Валидация рёбер
- Проверка существования source/target узлов
- Отбрасывание PREREQUISITE self-loops
- Проверка весов в диапазоне [0,1]
- **Фильтрация дублированных рёбер** (добавлено для решения проблемы с previous_response_id):
  - LLM может повторно создавать MENTIONS для узлов из предыдущих слайсов
  - Все дубликаты автоматически отфильтровываются
  - Логирование на уровне INFO для отслеживания
- Поддержка ссылок на концепты из ConceptDictionary

### Автоматическое добавление MENTIONS edges
- После применения каждого патча автоматически ищутся упоминания концептов
- Обрабатываются как новые Chunk узлы, так и обновленные существующие
- Поиск по всем термам из ConceptDictionary (primary + aliases)
- Гарантирует полноту графа даже если LLM пропустил очевидные связи
- Пример:
  ```
  Концепт: {"primary": "Стек", "aliases": ["stack", "LIFO"]}
  Chunk text: "Используем стек для хранения. Stack - это LIFO структура."
  Результат: 1 MENTIONS edge (все три упоминания ведут к одному концепту)
  ```

### Error Recovery
- **Repair-reprompt**: при невалидном JSON делается повторный запрос с уточнением
  - Метод `repair_response()` автоматически использует сохранённый previous_response_id
  - В repair промпт добавляется явное указание на ошибку и требование валидного JSON
- **HTML attributes cleanup**: перед парсингом JSON автоматически исправляются известные проблемы с кавычками в HTML атрибутах, которые LLM иногда генерирует при копировании ссылок из слайсов
- **Graceful degradation**: процесс продолжается при частичных сбоях
- **Temporary dumps**: сохранение состояния при критических ошибках
- **Interrupt handling**: корректная обработка Ctrl+C с сохранением результатов

## Configuration

Секция `[itext2kg]` в config.toml:
- **model** - модель LLM (o4-mini-2025-04-16)
- **tpm_limit** - лимит токенов в минуту
- **tpm_safety_margin** - запас безопасности для TPM (0.15)
- **max_completion** - максимум токенов на генерацию
- **log_level** - уровень логирования (debug/info)
- **temperature** - для обычных моделей
- **reasoning_effort** - для reasoning моделей
- **reasoning_summary** - формат резюме для reasoning моделей
- **timeout** - таймаут запроса в секундах
- **max_retries** - количество повторов при ошибках API

## Error Handling & Exit Codes

### Recoverable Errors
- **JSON validation errors** → repair-reprompt (1 попытка) → bad response сохраняется
- **Incremental validation errors** → слайс помечается failed → временный дамп → продолжение
- **API errors** → exponential backoff через llm_client (20s → 40s → 80s...)
- **Rate limits** → автоматическое ожидание через TPMBucket с восстановлением

### Non-recoverable Errors
- **All slices failed** → временные дампы → EXIT_RUNTIME_ERROR (3)
- **Configuration errors** → EXIT_CONFIG_ERROR (1)
- **I/O errors** → временные дампы → EXIT_IO_ERROR (5)

### Partial Failures
- Процесс продолжается если хотя бы некоторые слайсы успешны
- Предупреждение при failure rate > 50%
- Статистика сохраняется в логах и временных дампах

## Boundary Cases

- **Пустой staging** → EXIT_INPUT_ERROR (2)
- **Битый slice.json** → логирование, пропуск слайса, продолжение
- **Невалидный LLM ответ после repair** → сохранение в logs/{slice_id}_bad.json
- **Прерывание Ctrl+C** → сохранение временных дампов → EXIT_RUNTIME_ERROR
- **Validation failed (финальная)** → временные дампы с префиксом validation_failed
- **Validation failed (промежуточная)** → временный дамп для слайса → слайс failed
- **Высокий failure rate** → предупреждение, но продолжение работы
- **Дубликат ID концепта** → разрешен, семантическая дедупликация в dedup.py
- **Дублированные рёбра от LLM** → автоматически фильтруются в _validate_edges → INFO логирование
- **Некорректные HTML атрибуты в ответе LLM** → автоматическая очистка паттернов `='\"...\"'` → успешный парсинг

## Output Validation

Финальная валидация использует:
- `validate_json()` - проверка по схемам ConceptDictionary и LearningChunkGraph
- `validate_concept_dictionary_invariants()` - проверка инвариантов словаря
- `validate_graph_invariants_intermediate()` - промежуточная валидация графа (разрешает дубликаты концептов)

**Важно:** Финальная валидация НЕ использует полную `validate_graph_invariants()`, так как на этом этапе могут существовать дубликаты концептов, которые будут обработаны в dedup.py.

## Output Format

**ConceptDictionary.json:**
```json
{
  "concepts": [
    {
      "concept_id": "slug:p:term",
      "term": {"primary": "Термин", "aliases": ["term", "синоним"]},
      "definition": "Определение концепта"
    }
  ]
}
```

**LearningChunkGraph_raw.json:**
```json
{
  "nodes": [
    {
      "id": "slug:c:token_start",
      "type": "Chunk|Concept|Assessment",
      "text": "Текст узла",
      "local_start": 0,
      "difficulty": 1,
      "definition": "Для узлов типа Concept"
    }
  ],
  "edges": [
    {
      "source": "node_id",
      "target": "node_id", 
      "type": "PREREQUISITE|MENTIONS|...",
      "weight": 0.8
    }
  ]
}
```

**Bad Response Format ({slice_id}_bad.json):**
```json
{
  "slice_id": "slice_001",
  "timestamp": "2024-01-15T10:30:00Z",
  "original_response": "невалидный ответ LLM",
  "validation_error": "описание ошибки",
  "repair_response": "ответ после repair (если был)"
}
```

## Test Coverage

- **test_slice_processor**: 21 тестов
  - Инициализация и загрузка промптов
  - Форматирование входных данных
  - Обновление концептов и узлов
  - Валидация рёбер
  - Обработка LLM ответов
  - **Автоматическое добавление MENTIONS edges (5 тестов)**
  
- **test_processing_flow**: 8 тестов
  - Загрузка слайсов
  - Применение патчей
  - Сохранение bad responses
  - Успешная обработка и repair
  - Полный прогон pipeline

- **test_itext2kg_error_handling**: 9 тестов
  - Сохранение bad responses
  - Создание временных дампов
  - Repair-reprompt механизм
  - Graceful degradation
  - Обработка прерываний
  - Все виды сбоев

- **test_itext2kg_deduplication**: 11 тестов
  - Дедупликация узлов (Chunk/Assessment)
  - Обработка перекрывающихся текстов
  - Инкрементальная валидация
  - **Дедупликация рёбер (4 теста)**
  - Фильтрация дублированных MENTIONS
  - Сценарий с previous_response_id

## Dependencies
- **Standard Library**: json, logging, sys, time, pathlib, datetime, typing, dataclasses, re
- **External**: None (использует utils)
- **Internal**: utils.config, utils.exit_codes, utils.llm_client, utils.validation (включая validate_graph_invariants_intermediate), utils.console_encoding

## Performance Notes
- Последовательная обработка для сохранения контекста
- TPM контроль через llm_client с safety margin
- Детальное логирование в JSON Lines формате
- Прогресс выводится в терминал в реальном времени
- Checkpoint логирование каждые 10 слайсов
- Минимальная задержка при repair благодаря сохранению previous_response_id
- Промежуточная валидация добавляет минимальный overhead (< 50ms на слайс)
- Проверка дубликатов рёбер добавляет минимальный overhead (< 10ms на патч)
- Предварительная очистка HTML атрибутов добавляет минимальный overhead (< 1ms на ответ)

## Usage Examples
```bash
# Подготовка слайсов
python -m src.slicer

# Запуск извлечения
python -m src.itext2kg

# Проверка результатов
ls data/out/
# ConceptDictionary.json
# LearningChunkGraph_raw.json

# Просмотр логов при ошибках
cat logs/itext2kg_*.log | grep ERROR

# Анализ bad responses
ls logs/*_bad.json

# Восстановление из временных дампов
ls logs/*_temp_*.json
# ConceptDictionary_temp_interrupted_20240115_103045.json
# LearningChunkGraph_temp_interrupted_20240115_103045.json
# processing_stats_interrupted_20240115_103045.json
# ConceptDictionary_temp_validation_error_slice_042_20240115_103045.json
```