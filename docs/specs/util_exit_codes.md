# util_exit_codes.md

## Status: READY

Модуль стандартных кодов завершения для всех CLI-утилит проекта. Обеспечивает консистентную обработку ошибок через единую систему кодов с поддержкой читаемых названий и логирования.

## Public API

### Constants

#### Exit Codes
- **EXIT_SUCCESS** = 0 - Успешное выполнение
- **EXIT_CONFIG_ERROR** = 1 - Ошибки конфигурации (отсутствующий API ключ, битый config.toml)
- **EXIT_INPUT_ERROR** = 2 - Ошибки входных данных (пустые файлы, битые JSON схемы)
- **EXIT_RUNTIME_ERROR** = 3 - Ошибки выполнения (LLM отказы, битые slice после repair)
- **EXIT_API_LIMIT_ERROR** = 4 - TPM лимиты, rate limits (требуют retry/ожидания)
- **EXIT_IO_ERROR** = 5 - Ошибки записи файлов, доступа к каталогам

#### Dictionaries
- **EXIT_CODE_NAMES** - Словарь {код: название} для логирования
- **EXIT_CODE_DESCRIPTIONS** - Словарь {код: описание} для документации

### Functions

#### get_exit_code_name(code: int) -> str
Возвращает читаемое название кода завершения.
- **Input**: code - код завершения
- **Returns**: название кода (например "CONFIG_ERROR") или "UNKNOWN(код)" для неизвестных
- **Usage**: для формирования читаемых логов и сообщений об ошибках

#### get_exit_code_description(code: int) -> str
Возвращает описание кода завершения.
- **Input**: code - код завершения  
- **Returns**: описание кода или "Неизвестный код завершения: {код}"
- **Usage**: для подробных сообщений об ошибках и документации

#### log_exit(logger, code: int, message: str = None) -> None
Логирует код завершения с опциональным сообщением.
- **Input**: 
  - logger - объект логгера (logging.Logger)
  - code - код завершения
  - message - дополнительное сообщение (опционально)
- **Behavior**: 
  - SUCCESS логируется через logger.info()
  - Все остальные коды через logger.error()
  - Включает название и описание кода в лог

## Test Coverage

- **TestExitCodeConstants**: 3 теста
  - test_exit_code_values - проверка значений констант
  - test_exit_code_names_completeness - все коды имеют названия
  - test_exit_code_descriptions_completeness - все коды имеют описания

- **TestGetExitCodeName**: 2 теста
  - test_valid_codes - получение названий для всех кодов
  - test_unknown_code - обработка неизвестных кодов

- **TestGetExitCodeDescription**: 2 теста
  - test_valid_codes - получение описаний для всех кодов
  - test_unknown_code - обработка неизвестных кодов

- **TestLogExit**: 5 тестов
  - test_log_success_without_message
  - test_log_success_with_message
  - test_log_error_without_message
  - test_log_error_with_message
  - test_log_all_error_types

## Dependencies

- **Standard Library**: logging
- **External**: None
- **Internal**: None

## Usage Examples

### Базовое использование в CLI утилитах
```python
from utils.exit_codes import EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR
import sys

# Успешное завершение
if all_processed:
    sys.exit(EXIT_SUCCESS)

# Ошибка конфигурации
if not api_key:
    print("Ошибка: API ключ не найден в конфигурации", file=sys.stderr)
    sys.exit(EXIT_CONFIG_ERROR)

# Ошибка входных данных
if not input_files:
    print("Ошибка: Нет файлов для обработки", file=sys.stderr)
    sys.exit(EXIT_INPUT_ERROR)
```

### Использование с логированием
```python
from utils.exit_codes import EXIT_SUCCESS, EXIT_RUNTIME_ERROR, log_exit
import logging

logger = logging.getLogger(__name__)

try:
    # Обработка данных
    process_data()
    log_exit(logger, EXIT_SUCCESS, f"Обработано {count} файлов")
    sys.exit(EXIT_SUCCESS)
except Exception as e:
    log_exit(logger, EXIT_RUNTIME_ERROR, f"Критическая ошибка: {e}")
    sys.exit(EXIT_RUNTIME_ERROR)
```

### Обработка в главной функции
```python
from utils.exit_codes import (
    EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR, EXIT_API_LIMIT_ERROR, EXIT_IO_ERROR,
    get_exit_code_name
)

def main():
    try:
        # Основная логика
        return EXIT_SUCCESS
    except ConfigError:
        return EXIT_CONFIG_ERROR
    except InputError:
        return EXIT_INPUT_ERROR
    except RateLimitError:
        return EXIT_API_LIMIT_ERROR
    except IOError:
        return EXIT_IO_ERROR
    except Exception:
        return EXIT_RUNTIME_ERROR

if __name__ == "__main__":
    exit_code = main()
    if exit_code != EXIT_SUCCESS:
        print(f"Завершено с ошибкой: {get_exit_code_name(exit_code)}")
    sys.exit(exit_code)
```

## Error Code Guidelines

**Коды 1-2**: Проблемы настройки/входных данных, исправляются пользователем
- CONFIG_ERROR: проверить config.toml, API ключи, параметры
- INPUT_ERROR: проверить входные файлы, форматы, схемы

**Код 3**: Runtime ошибки, требуют анализа логов
- RUNTIME_ERROR: неожиданные исключения, битые данные после попыток восстановления

**Код 4**: Временные ограничения API, можно повторить позже
- API_LIMIT_ERROR: подождать и запустить заново

**Код 5**: Проблемы файловой системы
- IO_ERROR: проверить права доступа, наличие места на диске