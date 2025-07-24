# util_config.md

## Status: READY

Модуль для загрузки и валидации конфигурации iText2KG из TOML файла. Поддерживает Python 3.11+ (tomllib) и более ранние версии (tomli).

## Public API

### load_config(config_path: Union[str, Path] = None) -> Dict[str, Any]
Загружает и валидирует конфигурацию из TOML файла.
- **Input**: config_path (optional) - путь к файлу конфигурации, по умолчанию src/config.toml
- **Returns**: Dict[str, Any] - словарь с проверенной конфигурацией
- **Raises**: 
  - ConfigValidationError - при ошибках валидации
  - FileNotFoundError - если файл конфигурации не найден

### ConfigValidationError(Exception)
Исключение для ошибок валидации конфигурации.

## Configuration Sections

### [slicer]
- max_tokens (int, >0) - размер окна в токенах
- overlap (int, ≥0) - перекрытие между слайсами
- soft_boundary (bool) - использовать мягкие границы
- soft_boundary_max_shift (int, ≥0) - максимальное смещение для мягких границ
- tokenizer (str, ="o200k_base") - токенизатор
- allowed_extensions (list, не пустой) - допустимые расширения файлов

### [itext2kg]
- model (str) - модель LLM
- tpm_limit (int, >0) - лимит токенов в минуту
- max_completion (int, 1-100000) - максимум генерируемых токенов
- log_level (str, debug/info/warning/error) - уровень логирования
- api_key (str, не пустой) - ключ OpenAI API
- timeout (int, >0) - таймаут запроса в секундах
- max_retries (int, ≥0) - количество повторов
- poll_interval (int, >0) - интервал проверки статуса в секундах (для асинхронного режима)

### [dedup]
- embedding_model (str) - модель для эмбеддингов
- sim_threshold (float, 0.0-1.0) - порог схожести
- len_ratio_min (float, 0.0-1.0) - минимальное соотношение длин
- faiss_M (int, >0) - параметр HNSW графа
- faiss_efC (int, >0) - параметр HNSW конструкции
- faiss_metric (str, INNER_PRODUCT/L2) - метрика FAISS
- k_neighbors (int, >0) - количество соседей

### [refiner]
- run (bool) - запускать ли refiner
- embedding_model (str) - модель для эмбеддингов
- sim_threshold (float, 0.0-1.0) - порог схожести
- max_pairs_per_node (int, >0) - максимум пар на узел
- model (str) - модель LLM
- api_key (str, не пустой) - ключ OpenAI API
- tpm_limit (int, >0) - лимит токенов в минуту
- max_completion (int, 1-100000) - максимум генерируемых токенов
- timeout (int, >0) - таймаут запроса
- max_retries (int, ≥0) - количество повторов
- poll_interval (int, >0) - интервал проверки статуса в секундах (для асинхронного режима)
- weight_low (float, 0.0-1.0) - низкий вес связи
- weight_mid (float, 0.0-1.0) - средний вес связи
- weight_high (float, 0.0-1.0) - высокий вес связи

## Validation Rules

- Все секции обязательны: [slicer], [itext2kg], [dedup], [refiner]
- При overlap > 0: soft_boundary_max_shift ≤ overlap * 0.8
- Веса должны удовлетворять: weight_low < weight_mid < weight_high
- API ключи не могут быть пустыми (включая пробелы)
- Строгая проверка типов для всех полей

## Test Coverage

- **test_config_loading**: 3 теста
  - test_load_valid_config
  - test_missing_config_file
  - test_invalid_toml_syntax

- **test_slicer_validation**: 3 теста
  - test_missing_slicer_section
  - test_invalid_max_tokens
  - test_overlap_soft_boundary_validation

- **test_itext2kg_validation**: 2 теста
  - test_invalid_log_level
  - test_empty_api_key

- **test_refiner_validation**: 2 теста
  - test_invalid_weight_order
  - test_weight_out_of_range

- **test_type_validation**: 1 тест
  - test_wrong_type_validation

## Dependencies
- **Standard Library**: sys, pathlib, typing
- **External**: tomllib (Python 3.11+), tomli (Python <3.11)
- **Internal**: None

## Usage Examples
```python
from src.utils.config import load_config, ConfigValidationError

# Загрузка с дефолтным путем
try:
    config = load_config()
    slicer_config = config["slicer"]
    max_tokens = slicer_config["max_tokens"]
except ConfigValidationError as e:
    print(f"Configuration error: {e}")

# Загрузка с кастомным путем
config = load_config("custom_config.toml")
```
