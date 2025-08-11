"""
Глобальная конфигурация для pytest.
Автоматически загружает переменные окружения из .env файла.
"""

from pathlib import Path
import pytest
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла при запуске тестов
load_dotenv()


@pytest.fixture
def temp_project_dir(tmp_path):
    """
    Создает временную структуру проекта для тестов.
    Возвращает кортеж (temp_path, raw_dir, staging_dir, out_dir, logs_dir).
    """
    # Создаем структуру директорий как в реальном проекте
    raw_dir = tmp_path / "data" / "raw"
    staging_dir = tmp_path / "data" / "staging"
    out_dir = tmp_path / "data" / "out"
    logs_dir = tmp_path / "logs"
    
    raw_dir.mkdir(parents=True)
    staging_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    logs_dir.mkdir()
    
    return tmp_path, raw_dir, staging_dir, out_dir, logs_dir
