# cli_refiner.md

## Status: READY

CLI-утилита для добавления дальних связей в граф знаний. Ищет пропущенные связи между узлами, которые не встречались в одном контексте при первичной обработке. Использует семантическое сходство через embeddings для поиска кандидатов и LLM для анализа типов связей.

## CLI Interface

### Usage
```bash
python -m src.refiner
```

### Input Files
- **Primary**: `/data/out/LearningChunkGraph_dedup.json` - граф после дедупликации
- **Prompt**: `/prompts/refiner_relation.md` - промпт для LLM с подстановкой весов

### Output Files
- **Result**: `/data/out/LearningChunkGraph.json` - финальный граф с дальними связями
- **Logs**: `/logs/refiner_YYYY-MM-DD_HH-MM-SS.log` - JSON Lines логи
- **Errors**: `/logs/{node_id}_bad.json` - проблемные ответы LLM (при ошибках)

### Exit Codes
- 0 (SUCCESS) - успешное выполнение
- 1 (CONFIG_ERROR) - ошибки конфигурации
- 2 (INPUT_ERROR) - отсутствует входной файл
- 3 (RUNTIME_ERROR) - критические ошибки выполнения
- 4 (API_LIMIT_ERROR) - превышены лимиты API
- 5 (IO_ERROR) - ошибки записи файлов

## Core Algorithm

### Pipeline Overview
1. **Проверка конфигурации**: если `run = false` - просто копирование файла (без инициализации JSON логирования, используется простой вывод)
2. **Загрузка графа**: валидация по схеме и извлечение целевых узлов
3. **Генерация кандидатов**:
   - Получение embeddings для Chunk/Assessment узлов
   - Построение FAISS индекса для быстрого поиска
   - Поиск top-K похожих пар с учетом порога сходства
4. **LLM анализ связей**:
   - Последовательная обработка с сохранением контекста
   - Формирование запросов с текстами и существующими рёбрами
   - Валидация ответов с repair-retry при ошибках
5. **Обновление графа**: добавление/обновление рёбер с маркировкой
6. **Финальная валидация**: проверка инвариантов и сохранение

### Embeddings Processing
- **API**: OpenAI Embeddings API (/v1/embeddings)
- **Model**: настраивается через `embedding_model` (обычно text-embedding-3-small)
- **TPM Limit**: контролируется через `embedding_tpm_limit` (350,000)
- **Batching**: до 2048 текстов за запрос, до 8192 токенов на текст
- **Token counting**: через cl100k_base для embeddings API
- **Normalization**: векторы автоматически нормализованы (L2 norm = 1)

### FAISS Index Configuration
- **Index type**: HNSW (Hierarchical Navigable Small World)
- **Parameters**:
  - M: настраивается через `faiss_M` (связность графа)
  - efConstruction: настраивается через `faiss_efC` (точность построения)
  - Metric: INNER_PRODUCT или L2 через `faiss_metric`
- **Ordering**: детерминированный порядок через сортировку по local_start

### LLM Analysis
- **Sequential processing**: с сохранением previous_response_id между узлами
- **Context preservation**: LLM видит историю анализа предыдущих узлов
- **Auto-detection**: reasoning модели определяются по префиксу "o*"
- **Retry logic**: один repair-retry при битом JSON
- **Rate limiting**: контроль TPM через response headers
- **Prompt template**: загружается из файла с подстановкой весов

### Graph Update Logic
- **New edges**: добавляются с `conditions: "added_by=refiner_v1"`
- **Duplicate edges**: обновляется вес на максимальный из старого и нового
- **Type replacement**: только если новый вес ≥ старого, с `conditions: "fixed_by=refiner_v1"`
- **Direction conflicts**: A→B и B→A могут сосуществовать
- **Self-loop cleanup**: удаление PREREQUISITE self-loops после обновления

## Terminal Output

Утилита использует структурированный вывод прогресса с унифицированным форматом:
```
[HH:MM:SS] TAG      | Данные
```

### Формат вывода прогресса

**START - начало обработки:**
```
[10:30:00] START    | 89 nodes | model=o4-mini-2025-04-16 | tpm=100k
```

**NODE - обработка узла:**
```
[10:30:05] NODE     | ✅ 001/089 | pairs=15 | tokens=1240 | 320ms | edges_added=3
[10:30:12] NODE     | ✅ 002/089 | pairs=8 | tokens=890 | 285ms | edges_added=1
```

**ERROR - ошибки обработки:**
```
[10:30:45] ERROR    | ⚠️ RateLimitError | will retry...
[10:31:02] ERROR    | ⚠️ JSONDecodeError: Expecting value: line 1 column 1...
```

**END - завершение работы:**
```
[10:45:30] END      | Done | nodes=89 | edges_added=47 | time=8m 12s
```

### Простой вывод при run=false
Когда `run = false` в конфиге, используется простой print без логирования:
```
Refiner is disabled (run=false), copying file without changes
Copied data/out/LearningChunkGraph_dedup.json to data/out/LearningChunkGraph.json
```

### JSON Lines логирование
В файлах логов используется структурированный формат для детального анализа:
```json
{"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "event": "node_processed", "node_id": "n1", "pairs_count": 15, "tokens_used": 1240, "duration_ms": 320, "edges_added": 3}
{"timestamp": "2024-01-15T10:30:01Z", "level": "DEBUG", "event": "edge_added", "source": "n1", "target": "n2", "type": "PREREQUISITE", "weight": 0.8, "conditions": "added_by=refiner_v1"}
```

## Public Functions

### validate_refiner_config(config: Dict) -> None
Валидация параметров конфигурации.
- **Checks**: обязательные параметры, диапазоны значений, согласованность весов
- **Raises**: ValueError с описанием проблемы

### load_and_validate_graph(graph_path: Path) -> Dict
Загрузка и валидация графа из файла.
- **Input**: путь к JSON файлу графа
- **Returns**: валидный граф
- **Validates**: JSON schema соответствие

### extract_target_nodes(graph: Dict) -> List[Dict]
Извлечение узлов для анализа.
- **Filters**: только типы Chunk и Assessment
- **Returns**: список узлов с непустыми текстами

### get_node_embeddings(nodes: List[Dict], config: Dict, logger) -> Dict[str, np.ndarray]
Получение embeddings для узлов через OpenAI API.
- **Batching**: автоматическая группировка запросов
- **Filtering**: пропуск узлов с пустыми текстами
- **Returns**: словарь {node_id: embedding_vector}

### build_similarity_index(embeddings_dict, nodes, config, logger) -> Tuple[faiss.Index, List[str]]
Построение FAISS индекса для поиска.
- **Ordering**: сортировка по local_start для детерминированности
- **Configuration**: использует faiss_M, faiss_metric из конфига
- **Returns**: (index, ordered_node_ids)

### generate_candidate_pairs(nodes, embeddings_dict, index, node_ids, edges_index, config, logger) -> List[Dict]
Генерация пар кандидатов для анализа.
- **Filtering**: similarity ≥ sim_threshold, source.local_start < target.local_start
- **Limit**: max_pairs_per_node ближайших соседей
- **Skips**: пары с существующими рёбрами

### analyze_candidate_pairs(candidate_pairs, graph, config, logger) -> List[Dict]
Анализ пар через LLM для определения типов связей.
- **Sequential**: обработка с previous_response_id
- **Retry**: repair-reprompt при ошибках JSON
- **Validation**: проверка типов рёбер и весов
- **Returns**: список новых рёбер

### update_graph_with_new_edges(graph, new_edges, logger) -> Dict
Обновление графа новыми рёбрами.
- **Logic**: добавление/обновление/замена согласно правилам из ТЗ
- **Cleanup**: удаление PREREQUISITE self-loops
- **Statistics**: подсчёт изменений для логирования

## Configuration

### Required Parameters
```toml
[refiner]
run = true                           # false = просто копирование файла
embedding_model = "text-embedding-3-small"
embedding_api_key = "sk-..."         # можно оставить пустым (берётся из api_key)
embedding_tpm_limit = 350000         # лимит для embeddings API
sim_threshold = 0.80                 # порог косинусного сходства [0,1]
max_pairs_per_node = 20              # макс. кандидатов на узел
model = "gpt-4o"                     # модель для анализа связей
api_key = "sk-..."                   # ключ для LLM API
tpm_limit = 100000                   # лимит токенов/мин для LLM
tpm_safety_margin = 0.15             # запас для TPM контроля
max_completion = 4096                # макс. токенов ответа
temperature = 0.6                    # для обычных моделей
reasoning_effort = "medium"          # для reasoning моделей (o*)
reasoning_summary = "auto"           # для reasoning моделей (o*)
timeout = 45                         # таймаут запроса (сек)
max_retries = 3                      # количество retry попыток
weight_low = 0.3                     # вес для слабых связей
weight_mid = 0.6                     # вес для средних связей
weight_high = 0.9                    # вес для сильных связей
faiss_M = 32                         # параметр HNSW индекса
faiss_efC = 200                      # точность построения индекса
faiss_metric = "INNER_PRODUCT"       # метрика сходства (или L2)
log_level = "info"                   # уровень логирования (info/debug)
```

### Validation Rules
- api_key не пустой
- sim_threshold ∈ [0,1]
- max_pairs_per_node > 0
- max_completion ≤ 4096
- 0 ≤ weight_low < weight_mid < weight_high ≤ 1
- faiss_metric ∈ ["INNER_PRODUCT", "L2"]

## Error Handling

### API Errors
- **RateLimitError**: exponential backoff с retry
- **Embeddings limit**: обрезка текстов > 8192 токенов с логированием
- **Network errors**: retry с тем же контекстом

### LLM Response Errors
- **Invalid JSON**: один repair-retry с промптом об ошибке
- **Invalid edges**: фильтрация с логированием
- **Failed nodes**: сохранение в `/logs/{node_id}_bad.json`

### Critical Errors
- **Missing config params**: EXIT_CONFIG_ERROR
- **Input file not found**: EXIT_INPUT_ERROR
- **All retries exhausted**: EXIT_API_LIMIT_ERROR
- **File write errors**: EXIT_IO_ERROR

## Performance Notes

### Embeddings Optimization
- Batch processing до 2048 текстов для минимизации API вызовов
- Кэширование в памяти на время работы утилиты
- Параллельная обработка не используется (sequential for context)

### FAISS Performance
- HNSW индекс обеспечивает O(log N) поиск
- Память: ~4KB на вектор (1536 dims × 4 bytes + overhead)
- Построение индекса: O(N log N) для N узлов

### LLM Optimization
- Минимизация промптов через компактный формат
- Сохранение контекста уменьшает повторы
- Early termination при отсутствии кандидатов

## Test Coverage

- **test_refiner_config**: 8 тестов
  - Валидация параметров конфигурации
  - Проверка весов и диапазонов
  - Обработка run=false

- **test_embeddings_processing**: 10 тестов  
  - Получение embeddings
  - Построение FAISS индекса
  - Поиск кандидатов

- **test_llm_analysis**: 12 тестов
  - Формирование запросов
  - Валидация ответов
  - Repair механизм
  - Обработка битых JSON

- **test_graph_update**: 15 тестов
  - Добавление новых рёбер
  - Обновление весов
  - Замена типов
  - Очистка self-loops

## Dependencies

- **Standard Library**: json, logging, shutil, sys, pathlib, typing
- **External**: numpy, faiss-cpu  
- **Internal**: utils.config, utils.validation, utils.exit_codes, utils.llm_embeddings, utils.llm_client

## Usage Examples

### Basic Run
```bash
# Обычный запуск с конфигом по умолчанию
python -m src.refiner
```

### Skip Processing
```bash
# В config.toml установить run = false
# Просто скопирует dedup → final без обработки
python -m src.refiner
```

### Debug Mode
```bash
# В config.toml установить log_level = "debug"
# Детальные логи с промптами и ответами
python -m src.refiner
```

### Проверка результатов
```python
import json

# Сравнение графов до и после
with open('data/out/LearningChunkGraph_dedup.json') as f:
    before = json.load(f)
    
with open('data/out/LearningChunkGraph.json') as f:
    after = json.load(f)

# Статистика
edges_before = len(before['edges'])
edges_after = len(after['edges'])
edges_added = edges_after - edges_before

print(f"Рёбер было: {edges_before}")
print(f"Рёбер стало: {edges_after}")
print(f"Добавлено: {edges_added}")

# Проверка новых рёбер
new_edges = [e for e in after['edges'] if e.get('conditions') == 'added_by=refiner_v1']
print(f"Новых рёбер с маркировкой: {len(new_edges)}")
```