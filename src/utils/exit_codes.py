#!/usr/bin/env python3
"""
Стандартные коды завершения для всех утилит проекта iText2KG.

Система кодов завершения обеспечивает консистентную обработку ошибок:
- Коды 1-2: проблемы настройки/входных данных, исправляются пользователем
- Код 3: runtime ошибки, требуют анализа логов
- Код 4: временные ограничения API, можно повторить позже
- Код 5: проблемы файловой системы, проверить права доступа
"""

# Коды завершения согласно ТЗ
EXIT_SUCCESS = 0           # Успешное выполнение
EXIT_CONFIG_ERROR = 1      # Ошибки конфигурации (отсутствующий API ключ, битый config.toml)
EXIT_INPUT_ERROR = 2       # Ошибки входных данных (пустые файлы, битые JSON схемы)
EXIT_RUNTIME_ERROR = 3     # Ошибки выполнения (LLM отказы, битые slice после repair)
EXIT_API_LIMIT_ERROR = 4   # TPM лимиты, rate limits (требуют retry/ожидания)
EXIT_IO_ERROR = 5          # Ошибки записи файлов, доступа к каталогам

# Словарь для читаемых названий (для логирования)
EXIT_CODE_NAMES = {
    EXIT_SUCCESS: "SUCCESS",
    EXIT_CONFIG_ERROR: "CONFIG_ERROR",
    EXIT_INPUT_ERROR: "INPUT_ERROR",
    EXIT_RUNTIME_ERROR: "RUNTIME_ERROR",
    EXIT_API_LIMIT_ERROR: "API_LIMIT_ERROR",
    EXIT_IO_ERROR: "IO_ERROR"
}

# Словарь с описаниями для документации/логов
EXIT_CODE_DESCRIPTIONS = {
    EXIT_SUCCESS: "Успешное выполнение",
    EXIT_CONFIG_ERROR: "Ошибки конфигурации",
    EXIT_INPUT_ERROR: "Ошибки входных данных",
    EXIT_RUNTIME_ERROR: "Ошибки выполнения",
    EXIT_API_LIMIT_ERROR: "Лимиты API",
    EXIT_IO_ERROR: "Ошибки файловой системы"
}


def get_exit_code_name(code: int) -> str:
    """
    Возвращает читаемое название кода завершения.
    
    Args:
        code: Код завершения
        
    Returns:
        Название кода или 'UNKNOWN' для неизвестных кодов
    """
    return EXIT_CODE_NAMES.get(code, f"UNKNOWN({code})")


def get_exit_code_description(code: int) -> str:
    """
    Возвращает описание кода завершения.
    
    Args:
        code: Код завершения
        
    Returns:
        Описание кода или 'Неизвестный код' для неизвестных кодов
    """
    return EXIT_CODE_DESCRIPTIONS.get(code, f"Неизвестный код завершения: {code}")


def log_exit(logger, code: int, message: str = None) -> None:
    """
    Логирует код завершения с опциональным сообщением.
    
    Args:
        logger: Объект логгера
        code: Код завершения
        message: Дополнительное сообщение (опционально)
    """
    code_name = get_exit_code_name(code)
    code_desc = get_exit_code_description(code)
    
    if code == EXIT_SUCCESS:
        if message:
            logger.info(f"Завершение: {code_name} - {message}")
        else:
            logger.info(f"Завершение: {code_name}")
    else:
        if message:
            logger.error(f"Завершение с ошибкой: {code_name} ({code_desc}) - {message}")
        else:
            logger.error(f"Завершение с ошибкой: {code_name} ({code_desc})")
