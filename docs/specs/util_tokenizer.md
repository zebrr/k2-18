# util_tokenizer.md

## Status: READY

Модуль для работы с токенизацией текста и поиском семантических границ. Использует кодек o200k_base (совместим со всеми современными моделями OpenAI). Поддерживает иерархический поиск границ и безопасное разрезание на уровне токенов.

## Public API

### count_tokens(text: str) -> int
Подсчитывает количество токенов в тексте используя кодек o200k_base.
- **Input**: text (str) - входной текст для токенизации
- **Returns**: int - количество токенов в тексте  
- **Raises**: ValueError - если text не является строкой

### find_soft_boundary(text: str, target_pos: int, max_shift: int) -> Optional[int]
Ищет ближайшую семантическую границу в тексте для мягкого разрыва используя иерархию приоритетов.
- **Input**: 
  - text (str) - исходный текст
  - target_pos (int) - целевая позиция в символах
  - max_shift (int) - максимальное смещение от target_pos
- **Returns**: Optional[int] - позиция границы в символах или None если не найдена
- **Algorithm**: Использует систему весов для выбора лучшей границы: score = weight × distance

#### Иерархия границ (от высшего приоритета к низшему):
1. **section (вес 1)** - границы разделов:
   - HTML заголовки (`</h1-6>`)
   - Markdown заголовки (`#-####`)
   - Текстовые заголовки (Глава, Параграф, Часть, Chapter, Section, Раздел, Урок, Тема)
2. **paragraph (вес 2)** - границы абзацев:
   - Двойные переносы строк (`\n\n`)
   - Конец блока кода (` ``` `)
   - Конец блока формулы (`$$`)
   - HTML/Markdown ссылки (`</a>`, `](url)`)
3. **sentence (вес 3)** - границы предложений:
   - Конец предложения (`. ! ?` + пробел) с проверкой на сокращения
   - Точка с запятой (`;` + пробел)
4. **phrase (вес 4)** - границы фраз:
   - Запятая (`,` + пробел)
   - Двоеточие (`:` + пробел)  
   - Тире (`—–-` + пробелы)
5. **word (вес 5)** - границы слов (fallback):
   - Любой пробельный символ

### find_safe_token_boundary(text: str, tokens: List[int], encoding, target_token_pos: int, max_shift_tokens: int) -> int
Находит безопасную границу для разреза на уровне токенов.
- **Input**:
  - text (str) - исходный текст
  - tokens (List[int]) - список токенов
  - encoding - объект tiktoken encoding
  - target_token_pos (int) - целевая позиция в токенах
  - max_shift_tokens (int) - максимальное смещение в токенах
- **Returns**: int - безопасная позиция в токенах для разреза
- **Algorithm**: Проверяет каждую позицию на безопасность и выбирает лучшую по качеству

## Internal Methods

### is_safe_cut_position(text: str, tokens: List[int], encoding, pos: int) -> bool
Проверяет безопасность разреза в данной позиции токенов.
- Не режет внутри слова
- Не режет внутри URL
- Не режет внутри markdown ссылки
- Не режет внутри HTML тега
- Не режет внутри формулы
- Не режет внутри блока кода

### evaluate_boundary_quality(text: str, tokens: List[int], encoding, pos: int) -> float
Оценивает качество границы (чем меньше значение, тем лучше).
- Заголовки: score = 1.0
- Двойной перенос: score = 5.0
- Конец предложения: score = 10.0
- Конец абзаца: score = 15.0
- После запятой/точки с запятой: score = 20.0
- Между словами: score = 50.0
- Прочие: score = 100.0

### is_inside_url(text_before: str, text_after: str) -> bool
Проверяет, находится ли позиция внутри URL.

### is_inside_markdown_link(text_before: str, text_after: str) -> bool
Проверяет, находится ли позиция внутри markdown ссылки `[текст](url)`.

### is_inside_html_tag(text_before: str, text_after: str) -> bool
Проверяет, находится ли позиция внутри HTML тега.

### is_inside_formula(text_before: str, text_after: str) -> bool
Проверяет, находится ли позиция внутри математической формулы `$...$` или `$$...$$`.

### is_inside_code_block(text_before: str, text_after: str) -> bool
Проверяет, находится ли позиция внутри блока кода ` ```...``` `.

## Test Coverage

- **test_count_tokens**: 7 тестов
  - test_basic_text
  - test_empty_string
  - test_russian_text
  - test_mixed_languages
  - test_special_characters
  - test_code_content
  - test_invalid_input_type

- **test_find_soft_boundary**: 14 тестов
  - test_invalid_inputs
  - test_no_boundaries_found
  - test_markdown_headers
  - test_html_headers
  - test_text_headers_russian
  - test_text_headers_english
  - test_double_newlines
  - test_sentence_endings
  - test_code_blocks
  - test_formula_blocks
  - test_html_links
  - test_markdown_links
  - test_closest_boundary_selection
  - test_boundary_within_range

- **test_fixtures**: 4 теста
  - test_markdown_fixture
  - test_html_fixture
  - test_mixed_fixture
  - test_token_consistency

## Dependencies
- **Standard Library**: re, logging, typing
- **External**: tiktoken
- **Internal**: None

## Performance Notes
- Использует регулярные выражения для поиска границ - может быть медленным на очень больших текстах
- find_soft_boundary выполняет множественные regex-поиски по иерархии приоритетов
- Алгоритм выбора оптимизирован для образовательного контента
- tiktoken достаточно быстрый для обработки больших текстов

## Usage Examples
```python
from src.utils.tokenizer import count_tokens, find_soft_boundary, find_safe_token_boundary
import tiktoken

# Подсчет токенов
tokens = count_tokens("Hello world!")  # возвращает количество токенов

# Поиск мягкой границы с иерархией приоритетов
boundary = find_soft_boundary(text, target_pos=1000, max_shift=100)
if boundary:
    # Найдена оптимальная граница на позиции boundary
    left_part = text[:boundary]
    right_part = text[boundary:]

# Безопасный разрез на уровне токенов
encoding = tiktoken.get_encoding("o200k_base")
tokens = encoding.encode(text)
safe_pos = find_safe_token_boundary(
    text, tokens, encoding, 
    target_token_pos=40000, 
    max_shift_tokens=500
)
# Разрезаем по безопасной позиции
left_tokens = tokens[:safe_pos]
right_tokens = tokens[safe_pos:]
```