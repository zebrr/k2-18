# util_console_encoding.md

## Status: READY

Утилита для безопасной настройки UTF-8 кодировки в Windows консоли. Решает проблему с отображением кириллицы и специальных символов, не ломая при этом pytest и другие инструменты.

## Public API

### setup_console_encoding() -> None
Настраивает UTF-8 кодировку для Windows консоли безопасным способом.
- **Input**: нет
- **Returns**: None
- **Raises**: никогда не выбрасывает исключения

Функция автоматически определяет:
- Операционную систему (работает только в Windows)
- Запущен ли код под pytest (не трогает потоки под тестами)
- Переопределены ли уже потоки (не делает двойную обертку)
- Является ли вывод терминалом (не трогает при перенаправлении)

## Test Coverage

- **test_console_encoding**: 4 теста
  - test_setup_console_encoding_windows
  - test_setup_console_encoding_non_windows
  - test_setup_console_encoding_under_pytest
  - test_setup_console_encoding_non_tty

## Dependencies
- **Standard Library**: sys, os, io
- **External**: None
- **Internal**: None

## Performance Notes
- Функция выполняется мгновенно (< 1ms)
- Не влияет на производительность вывода
- Безопасна для многократного вызова

## Usage Examples
```python
from src.utils.console_encoding import setup_console_encoding

# В начале CLI-утилиты
setup_console_encoding()

# Теперь можно безопасно выводить кириллицу
print("Привет, мир!")
logging.info("Обработка файла: Алгоритмы.txt")
```

### Замена старого кода
Вместо небезопасного:
```python
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

Используйте:
```python
from src.utils.console_encoding import setup_console_encoding
setup_console_encoding()
```
