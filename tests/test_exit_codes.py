"""
Unit тесты для модуля exit_codes.py

Тестирует:
- Константы кодов ошибок
- Функции получения названий и описаний
- Функцию логирования
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import Mock

# Добавляем src в path для импорта
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.exit_codes import (
    EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR, EXIT_API_LIMIT_ERROR, EXIT_IO_ERROR,
    EXIT_CODE_NAMES, EXIT_CODE_DESCRIPTIONS,
    get_exit_code_name, get_exit_code_description, log_exit
)


class TestExitCodeConstants:
    """Тесты для констант кодов ошибок."""
    
    def test_exit_code_values(self):
        """Проверка значений констант согласно ТЗ."""
        assert EXIT_SUCCESS == 0
        assert EXIT_CONFIG_ERROR == 1
        assert EXIT_INPUT_ERROR == 2
        assert EXIT_RUNTIME_ERROR == 3
        assert EXIT_API_LIMIT_ERROR == 4
        assert EXIT_IO_ERROR == 5
    
    def test_exit_code_names_completeness(self):
        """Проверка, что все коды имеют названия."""
        expected_codes = [
            EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR,
            EXIT_RUNTIME_ERROR, EXIT_API_LIMIT_ERROR, EXIT_IO_ERROR
        ]
        
        for code in expected_codes:
            assert code in EXIT_CODE_NAMES
            assert isinstance(EXIT_CODE_NAMES[code], str)
            assert len(EXIT_CODE_NAMES[code]) > 0
    
    def test_exit_code_descriptions_completeness(self):
        """Проверка, что все коды имеют описания."""
        expected_codes = [
            EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR,
            EXIT_RUNTIME_ERROR, EXIT_API_LIMIT_ERROR, EXIT_IO_ERROR
        ]
        
        for code in expected_codes:
            assert code in EXIT_CODE_DESCRIPTIONS
            assert isinstance(EXIT_CODE_DESCRIPTIONS[code], str)
            assert len(EXIT_CODE_DESCRIPTIONS[code]) > 0


class TestGetExitCodeName:
    """Тесты для функции get_exit_code_name."""
    
    def test_valid_codes(self):
        """Проверка получения названий для валидных кодов."""
        assert get_exit_code_name(EXIT_SUCCESS) == "SUCCESS"
        assert get_exit_code_name(EXIT_CONFIG_ERROR) == "CONFIG_ERROR"
        assert get_exit_code_name(EXIT_INPUT_ERROR) == "INPUT_ERROR"
        assert get_exit_code_name(EXIT_RUNTIME_ERROR) == "RUNTIME_ERROR"
        assert get_exit_code_name(EXIT_API_LIMIT_ERROR) == "API_LIMIT_ERROR"
        assert get_exit_code_name(EXIT_IO_ERROR) == "IO_ERROR"
    
    def test_unknown_code(self):
        """Проверка обработки неизвестного кода."""
        assert get_exit_code_name(999) == "UNKNOWN(999)"
        assert get_exit_code_name(-1) == "UNKNOWN(-1)"


class TestGetExitCodeDescription:
    """Тесты для функции get_exit_code_description."""
    
    def test_valid_codes(self):
        """Проверка получения описаний для валидных кодов."""
        assert "Успешное выполнение" in get_exit_code_description(EXIT_SUCCESS)
        assert "конфигурации" in get_exit_code_description(EXIT_CONFIG_ERROR)
        assert "входных данных" in get_exit_code_description(EXIT_INPUT_ERROR)
        assert "выполнения" in get_exit_code_description(EXIT_RUNTIME_ERROR)
        assert "API" in get_exit_code_description(EXIT_API_LIMIT_ERROR)
        assert "файловой системы" in get_exit_code_description(EXIT_IO_ERROR)
    
    def test_unknown_code(self):
        """Проверка обработки неизвестного кода."""
        desc = get_exit_code_description(999)
        assert "Неизвестный код завершения: 999" in desc


class TestLogExit:
    """Тесты для функции log_exit."""
    
    def test_log_success_without_message(self):
        """Проверка логирования успешного завершения без сообщения."""
        mock_logger = Mock()
        log_exit(mock_logger, EXIT_SUCCESS)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "SUCCESS" in call_args
        assert "Завершение:" in call_args
    
    def test_log_success_with_message(self):
        """Проверка логирования успешного завершения с сообщением."""
        mock_logger = Mock()
        log_exit(mock_logger, EXIT_SUCCESS, "Обработано 10 файлов")
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "SUCCESS" in call_args
        assert "Обработано 10 файлов" in call_args
    
    def test_log_error_without_message(self):
        """Проверка логирования ошибки без сообщения."""
        mock_logger = Mock()
        log_exit(mock_logger, EXIT_CONFIG_ERROR)
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "CONFIG_ERROR" in call_args
        assert "конфигурации" in call_args
    
    def test_log_error_with_message(self):
        """Проверка логирования ошибки с сообщением."""
        mock_logger = Mock()
        log_exit(mock_logger, EXIT_INPUT_ERROR, "Файл пустой")
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "INPUT_ERROR" in call_args
        assert "Файл пустой" in call_args
        assert "входных данных" in call_args
    
    def test_log_all_error_types(self):
        """Проверка логирования всех типов ошибок."""
        error_codes = [
            EXIT_CONFIG_ERROR,
            EXIT_INPUT_ERROR,
            EXIT_RUNTIME_ERROR,
            EXIT_API_LIMIT_ERROR,
            EXIT_IO_ERROR
        ]
        
        for code in error_codes:
            mock_logger = Mock()
            log_exit(mock_logger, code)
            mock_logger.error.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
