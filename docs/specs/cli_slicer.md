# cli_slicer.md

## Status: READY

CLI-утилита для разделения образовательных текстов на слайсы. Читает файлы из /data/raw/, применяет препроцессинг и нарезает их на фрагменты фиксированного размера с учетом семантических границ.

## CLI Interface

### Usage
```bash
python -m src.slicer
```

### Input Directory
- **Source**: `/data/raw/` - исходные файлы корпуса
- **Supported formats**: .json, .txt, .md, .html (настраивается в конфиге)
- **Processing order**: лексикографический (для детерминированности)
- **Case handling**: поиск файлов учитывает регистр расширений (ищет .txt и .TXT)

### Output Directory  
- **Target**: `/data/staging/` - слайсы в формате *.slice.json
- **Naming**: slice_001.slice.json, slice_002.slice.json, etc.

## Core Algorithm

### File Processing Pipeline
1. **File Discovery**: поиск файлов по allowed_extensions в лексикографическом порядке (с учетом регистра)
2. **Preprocessing**: Unicode нормализация NFC + удаление <script>/<style> тегов через BeautifulSoup
3. **Validation**: проверка на пустоту после препроцессинга
4. **Slicing**: нарезка скользящим окном с учетом soft boundaries
5. **Output**: сохранение в JSON формате с валидацией

### Sliding Window Algorithm
- **Window size**: max_tokens (настраивается)
- **Overlap**: configurable overlap между слайсами 
- **Soft boundaries**: поиск семантических границ в пределах soft_boundary_max_shift токенов
- **Token conversion**: soft_boundary_max_shift конвертируется из символов в токены (≈1 токен = 4 символа)
- **Boundary detection**: использует find_safe_token_boundary для поиска оптимальной границы
- **Token calculation**: точный подсчет через tiktoken o200k_base

### Boundary Rules
- При overlap = 0: slice_token_start(next) = slice_token_end(current)
- При overlap > 0: slice_token_start(next) = slice_token_end(current) - overlap
- Защита от бесконечного цикла: если новый start ≤ старого, то start = старый + 1
- Граничные случаи: последний фрагмент при overlap обновляет предыдущий слайс
- File boundaries: жесткие границы, overlap никогда не захватывает следующий файл

## Terminal Output

### Console Encoding
На Windows автоматически настраивается UTF-8 кодировка консоли через `setup_console_encoding()`.

### Log Format
Утилита использует стандартное логирование с форматом:
```
[HH:MM:SS] LEVEL    | Сообщение
```

### Standard Output Examples
```
[10:30:00] INFO     | Запуск slicer.py
[10:30:00] INFO     | Найдено 3 файлов для обработки
[10:30:00] INFO     | Обработка файла: chapter1.md
[10:30:01] INFO     | Soft boundary найдена: сдвиг +15 токенов
[10:30:01] INFO     | Сохранен слайс: data\staging\slice_001.slice.json
[10:30:01] INFO     | Файл chapter1.md: создано 4 слайсов
[10:30:02] INFO     | Обработка завершена: 8 слайсов сохранено в data\staging
[10:30:02] INFO     | Завершение работы с кодом: SUCCESS (0)
```

### Debug Output Examples
В режиме DEBUG показывается тип найденной границы:
```
[10:30:01] INFO     | Soft boundary найдена: сдвиг -23 токенов
[10:30:01] DEBUG    | Тип границы: двойной перенос строки
[10:30:01] INFO     | Soft boundary найдена: сдвиг +8 токенов
[10:30:01] DEBUG    | Тип границы: конец предложения
[10:30:01] INFO     | Soft boundary найдена: сдвиг +12 токенов
[10:30:01] DEBUG    | Тип границы: HTML заголовок
```

### Warning and Error Examples
```
[10:30:00] WARNING  | Не найдено файлов для обработки в data\raw
[10:30:00] WARNING  | Поддерживаемые расширения: ['json', 'txt', 'md', 'html']
[10:30:00] WARNING  | Unsupported file skipped: image.png
[10:30:00] ERROR    | Ошибка конфигурации: slicer.max_tokens должен быть положительным целым числом
[10:30:00] ERROR    | Ошибка входных данных в файле empty.md: Empty file detected: empty.md. Please remove empty files from /data/raw/
[10:30:00] ERROR    | Ошибка ввода/вывода при обработке file.txt: Не удалось сохранить слайс slice_001
```

## Public Functions

### create_slug(filename: str) -> str
Создает slug из имени файла.
- **Rules**: удаление расширения, транслитерация кириллицы (unidecode), lowercase, пробелы → "_"
- **Examples**: "Алгоритмы.txt" → "algoritmy", "My Course 1.md" → "my_course_1"

### preprocess_text(text: str) -> str
Применяет препроцессинг к тексту.
- **Steps**: Unicode нормализация NFC, удаление <script>/<style> через BeautifulSoup
- **Preservation**: остальное содержимое остается без изменений
- **Raises**: ValueError если входной параметр не строка

### validate_config_parameters(config: Dict[str, Any]) -> None
Валидация параметров конфигурации slicer.
- **Checks**: обязательные параметры, типы, диапазоны, специальные правила overlap
- **Constraint**: при overlap > 0, soft_boundary_max_shift ≤ overlap * 0.8
- **Raises**: ValueError с детальным описанием ошибки

### slice_text_with_window(text: str, max_tokens: int, overlap: int, soft_boundary: bool, soft_boundary_max_shift: int) -> List[Tuple[str, int, int]]
Основной алгоритм нарезки текста на слайсы.
- **Returns**: список кортежей (slice_text, slice_token_start, slice_token_end)
- **Features**: soft boundary detection через find_safe_token_boundary, overlap handling, граничные случаи
- **Token conversion**: soft_boundary_max_shift делится на 4 для конвертации в токены

### load_and_validate_file(file_path: Path, allowed_extensions: List[str]) -> str
Загрузка файла с автоопределением кодировки.
- **Encodings**: utf-8 → cp1251 → latin1 (fallback chain)
- **Validation**: проверка расширения, проверка на пустоту после препроцессинга
- **Raises**: InputError для пустых файлов или неподдерживаемых расширений

### process_file(file_path: Path, config: Dict[str, Any], global_slice_counter: int) -> Tuple[List[Dict[str, Any]], int]
Обрабатывает один файл и возвращает список слайсов.
- **Input**: путь к файлу, конфигурация, глобальный счетчик слайсов
- **Returns**: кортеж (список слайсов, обновленный счетчик)
- **Workflow**: загрузка → создание slug → нарезка → формирование slice объектов
- **Raises**: InputError, RuntimeError, IOError

### save_slice(slice_data: Dict[str, Any], output_dir: Path) -> None
Сохраняет слайс в JSON файл.
- **Filename**: {slice_id}.slice.json
- **Encoding**: UTF-8 с ensure_ascii=False
- **Raises**: IOError при ошибках записи

### setup_logging(log_level: str = "info") -> None
Настройка логирования для slicer.
- **Levels**: debug, info, warning, error
- **Format**: [HH:MM:SS] LEVEL | Message
- **Handler**: консольный вывод в stdout

## Output Format

### Slice JSON Structure
```json
{
  "id": "slice_042",
  "order": 42,
  "source_file": "chapter03.md", 
  "slug": "chapter03",
  "text": "processed content...",
  "slice_token_start": 52000,
  "slice_token_end": 92000
}
```

### ID Generation
- **Pattern**: slice_{order:03d} (slice_001, slice_002, ...)
- **Uniqueness**: global counter across all files
- **Deterministic**: повторные запуски дают идентичные ID

## Configuration

### Required Parameters (slicer section)
- **max_tokens** (int, >0) - размер окна в токенах
- **overlap** (int, ≥0) - перекрытие между слайсами  
- **soft_boundary** (bool) - использовать мягкие границы
- **soft_boundary_max_shift** (int, ≥0) - максимальное смещение для поиска границ (в символах)
- **tokenizer** (str, ="o200k_base") - токенизатор
- **allowed_extensions** (list, не пустой) - допустимые расширения файлов
- **log_level** (str) - уровень логирования (debug/info/warning/error)

### Validation Rules
- overlap < max_tokens
- При overlap > 0: soft_boundary_max_shift ≤ overlap * 0.8
- allowed_extensions не пустой список

## Error Handling & Exit Codes

### Custom Exceptions
- **InputError** - специальное исключение для ошибок входных данных (пустые файлы, неподдерживаемые расширения)

### Exit Codes
- **0 (EXIT_SUCCESS)** - успешное выполнение
- **1 (EXIT_CONFIG_ERROR)** - ошибки конфигурации  
- **2 (EXIT_INPUT_ERROR)** - пустые файлы, неподдерживаемые расширения
- **3 (EXIT_RUNTIME_ERROR)** - ошибки обработки
- **5 (EXIT_IO_ERROR)** - ошибки записи файлов, доступа к каталогам

### Error Types
- **InputError** - пустые файлы, неподдерживаемые расширения
- **ValueError** - некорректные параметры конфигурации
- **IOError** - проблемы записи в /data/staging/
- **RuntimeError** - неожиданные ошибки обработки

### Exit Code Logging
Используется функция `log_exit()` для логирования кода завершения в читаемом формате.

## Boundary Cases

### Empty Files
Ошибка EXIT_INPUT_ERROR с сообщением: "Empty file detected: {filename}. Please remove empty files from /data/raw/"

### Unsupported Files  
Предупреждение: "Unsupported file skipped: {filename}"

### Last Fragment Handling
- **overlap = 0**: создается отдельный слайс независимо от размера
- **overlap > 0**: если последний фрагмент < overlap, обновляется предыдущий слайс

### No Files Found
При отсутствии файлов для обработки:
- Выводится предупреждение о поддерживаемых расширениях
- Возвращается EXIT_SUCCESS (не считается ошибкой)

### Infinite Loop Protection
При расчете overlap добавлена защита: если новый start ≤ старого, принудительно увеличивается на 1.

## Test Coverage

- **test_create_slug**: 6 тестов
  - test_cyrillic_transliteration
  - test_english_with_spaces  
  - test_hyphens_preserved
  - test_extension_removal
  - test_complex_filename
  - test_special_characters

- **test_preprocess_text**: множество тестов
  - test_unicode_normalization
  - test_script_tag_removal (через BeautifulSoup)
  - test_style_tag_removal (через BeautifulSoup)
  - test_plain_text_unchanged
  - test_invalid_input_type

- **test_validate_config_parameters**: тесты валидации
  - test_valid_config
  - test_missing_parameters
  - test_invalid_types
  - test_overlap_constraint

- **test_slice_text_with_window**: тесты алгоритма нарезки
  - test_single_slice
  - test_multiple_slices_no_overlap
  - test_overlap_handling
  - test_soft_boundary_detection
  - test_infinite_loop_protection

- **test_process_file**: тесты обработки файлов
  - test_successful_processing
  - test_empty_file_error
  - test_unsupported_extension

- **integration tests**: полный pipeline тесты
- **large file tests**: производительность на больших файлах

## Dependencies
- **Standard Library**: argparse, json, logging, sys, unicodedata, pathlib, typing, re
- **External**: unidecode, beautifulsoup4 (bs4), tiktoken
- **Internal**: utils.config, utils.tokenizer (find_safe_token_boundary), utils.validation, utils.exit_codes, utils.console_encoding

## Performance Notes
- Детерминированная обработка (лексикографический порядок файлов)
- Точный подсчет токенов через tiktoken o200k_base
- Эффективная обработка больших файлов через streaming токенизацию
- Soft boundary поиск с автоматической конвертацией символов в токены
- Поиск файлов с учетом регистра расширений для кроссплатформенности

## Usage Examples
```bash
# Простой запуск (использует config.toml)
python -m src.slicer

# Проверка результатов
dir data\staging\
# slice_001.slice.json
# slice_002.slice.json
# ...

# Структура файлов до:
/data/raw/
  chapter1.md
  lesson2.txt
  exercises.json
  README.TXT    # будет обработан (регистр расширения)

# Структура файлов после:  
/data/staging/
  slice_001.slice.json  # из chapter1.md
  slice_002.slice.json  # из chapter1.md (продолжение)
  slice_003.slice.json  # из lesson2.txt
  slice_004.slice.json  # из exercises.json
  slice_005.slice.json  # из README.TXT
  
# Просмотр слайса
type data\staging\slice_001.slice.json

# Проверка кодировки файлов
python -m src.slicer
# Автоматически обработает разные кодировки (utf-8, cp1251, latin1)

# Debug режим для анализа soft boundaries
# Установите log_level = "debug" в config.toml
python -m src.slicer
# [10:30:01] INFO     | Soft boundary найдена: сдвиг +15 токенов
# [10:30:01] DEBUG    | Тип границы: двойной перенос строки
```