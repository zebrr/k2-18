# cli_dedup.md

## Status: READY

Утилита для удаления дубликатов узлов из графа знаний. Использует векторные эмбеддинги и FAISS для поиска похожих узлов типа Chunk и Assessment.

## CLI Interface

**Запуск:**
```bash
python -m src.dedup
```

**Входные каталоги:**
- `/data/out/LearningChunkGraph_raw.json` - граф после itext2kg

**Выходные каталоги:**
- `/data/out/LearningChunkGraph_dedup.json` - граф без дубликатов
- `/logs/dedup_map.csv` - маппинг дубликатов

## Core Algorithm

1. **Загрузка и валидация** - проверка входного графа по схеме
2. **Фильтрация узлов** - отбор Chunk/Assessment с непустым текстом
3. **Получение эмбеддингов** - через OpenAI API (text-embedding-3-small)
4. **Построение FAISS индекса** - HNSW для быстрого поиска
5. **Поиск дубликатов** - по cosine similarity и length ratio
6. **Кластеризация** - Union-Find для транзитивных дубликатов
7. **Перезапись графа** - удаление дубликатов и пустых узлов

## Terminal Output

Утилита использует стандартное логирование с форматом:
```
[HH:MM:SS] LEVEL    | Сообщение
```

Примеры вывода:
```
[10:30:00] INFO     | Загрузка графа знаний...
[10:30:01] INFO     | Отфильтровано 156 узлов из 189 для дедупликации
[10:30:02] INFO     | Получение эмбеддингов для 156 узлов...
[10:30:15] INFO     | Построение FAISS индекса...
[10:30:15] INFO     | Поиск дубликатов...
[10:30:16] INFO     | Найдено 23 потенциальных дубликатов
[10:30:16] INFO     | Кластеризация дубликатов...
[10:30:16] INFO     | Сформировано 8 кластеров, 15 узлов помечены как дубликаты
[10:30:16] INFO     | Перезапись графа...
[10:30:16] INFO     | Удалено 15 узлов-дубликатов, 3 пустых узлов
[10:30:16] INFO     | Обновлено 42 рёбер, финальное количество: 287
[10:30:17] INFO     | Сохранение результатов...
[10:30:17] INFO     | Сохранён маппинг дубликатов в logs/dedup_map.csv
[10:30:17] INFO     | Дедупликация завершена за 17.24 секунд
[10:30:17] INFO     | Узлов было: 189, стало: 171
[10:30:17] INFO     | Рёбер было: 312, стало: 287
```

При ошибках:
```
[10:30:00] ERROR    | Входной файл не найден: data\out\LearningChunkGraph_raw.json
[10:30:00] WARNING  | Недостаточно узлов для дедупликации, копируем граф без изменений
[10:30:00] ERROR    | Превышен лимит API: Rate limit exceeded
```

## Public Functions

### filter_nodes_for_dedup(nodes: List[Dict]) -> List[Dict]
Фильтрация узлов для дедупликации.
- **Input**: nodes - список всех узлов графа
- **Returns**: отфильтрованные узлы (Chunk/Assessment с текстом)

### build_faiss_index(embeddings: np.ndarray, config: Dict) -> faiss.IndexHNSWFlat
Создание FAISS индекса.
- **Input**: embeddings - матрица эмбеддингов, config - параметры FAISS
- **Returns**: построенный индекс

### find_duplicates(nodes, embeddings, index, config) -> List[Tuple[str, str, float]]
Поиск кандидатов-дубликатов.
- **Input**: nodes - узлы, embeddings - векторы, index - FAISS, config - параметры
- **Returns**: список (master_id, duplicate_id, similarity)

### cluster_duplicates(duplicates) -> Dict[str, str]
Кластеризация через Union-Find.
- **Input**: duplicates - пары дубликатов
- **Returns**: словарь {duplicate_id: master_id}

### rewrite_graph(graph, dedup_map) -> Dict
Перезапись графа с удалением дубликатов и пустых узлов.
- **Input**: graph - исходный граф, dedup_map - маппинг дубликатов
- **Returns**: новый граф без дубликатов

## Output Format

### LearningChunkGraph_dedup.json
```json
{
  "nodes": [
    {
      "id": "string",
      "type": "Chunk|Concept|Assessment",
      "text": "string",
      "local_start": 0,
      // другие поля согласно схеме
    }
  ],
  "edges": [
    {
      "source": "string",
      "target": "string", 
      "type": "PREREQUISITE|ELABORATES|...",
      "weight": 0.5
    }
  ]
}
```

### dedup_map.csv
```csv
duplicate_id,master_id,similarity
chunk_123,chunk_042,0.9823
chunk_789,chunk_042,0.9756
```

## Configuration

Секция `[dedup]` в config.toml:

- **embedding_model**: "text-embedding-3-small" - модель для эмбеддингов
- **embedding_api_key**: "sk-..." - API ключ (опционально, берется из api_key)
- **embedding_tpm_limit**: 350000 - лимит токенов в минуту
- **sim_threshold**: 0.97 - порог косинусной близости
- **len_ratio_min**: 0.8 - минимальное отношение длин текстов
- **faiss_M**: 32 - параметр HNSW (связность графа)
- **faiss_efC**: 200 - параметр HNSW (качество построения)
- **k_neighbors**: 5 - количество ближайших соседей

## Error Handling & Exit Codes

- **0 (SUCCESS)**: Успешное выполнение
- **1 (CONFIG_ERROR)**: Ошибки конфигурации (неверные параметры)
- **2 (INPUT_ERROR)**: Отсутствует входной файл или не проходит валидацию
- **3 (RUNTIME_ERROR)**: Ошибки выполнения (FAISS, обработка данных)
- **4 (API_LIMIT_ERROR)**: Превышены лимиты OpenAI API
- **5 (IO_ERROR)**: Ошибки записи файлов

## Boundary Cases

**Недостаточно узлов для дедупликации (<2):**
- Граф копируется без изменений
- Создается пустой dedup_map.csv
- Код выхода: 0 (SUCCESS)

**Пустые узлы (Chunk/Assessment):**
- Удаляются из финального графа
- Рёбра на них также удаляются

**Слишком длинные тексты (>8192 токенов):**
- Обрезаются до 8000 токенов с предупреждением
- Обработка продолжается

**Нет дубликатов найдено:**
- Граф копируется с удалением пустых узлов
- Создается пустой dedup_map.csv

**Master selection при равных local_start:**
- Выбирается узел с лексикографически меньшим ID

## Test Coverage

- **test_dedup**: 24 теста
  - test_union_find_basic
  - test_filter_nodes_for_dedup
  - test_find_duplicates_with_mock
  - test_cluster_duplicates
  - test_rewrite_graph
  - test_main_success
  - test_edge_cases

- **test_dedup_integration**: 5 тестов
  - test_full_dedup_process
  - test_no_duplicates_case
  - test_transitive_duplicates
  - test_edge_cases_with_real_api
  - test_performance_large_graph

## Dependencies

- **Standard Library**: sys, json, logging, time, pathlib, csv
- **External**: numpy, faiss-cpu
- **Internal**: utils.config, utils.validation, utils.llm_embeddings, utils.exit_codes

## Performance Notes

- **Эмбеддинги**: Батчевая обработка до 2048 текстов за запрос
- **FAISS индекс**: HNSW обеспечивает O(log N) поиск соседей
- **Память**: ~2GB для 10K узлов (эмбеддинги + индекс)
- **Скорость**: ~1000 узлов/минуту (включая API запросы)
- **TPM контроль**: Автоматический через EmbeddingsClient

## Usage Examples

### Запуск дедупликации
```bash
# Убедитесь, что есть входной файл
dir data\out\LearningChunkGraph_raw.json

# Запустите дедупликацию
python -m src.dedup

# Проверьте результаты
dir data\out\LearningChunkGraph_dedup.json
type logs\dedup_map.csv
```

### Проверка результатов
```python
import json

# Загрузка графов
with open('data/out/LearningChunkGraph_raw.json') as f:
    raw_graph = json.load(f)
    
with open('data/out/LearningChunkGraph_dedup.json') as f:
    dedup_graph = json.load(f)

# Статистика
print(f"Узлов было: {len(raw_graph['nodes'])}")
print(f"Узлов стало: {len(dedup_graph['nodes'])}")
print(f"Удалено: {len(raw_graph['nodes']) - len(dedup_graph['nodes'])}")
```