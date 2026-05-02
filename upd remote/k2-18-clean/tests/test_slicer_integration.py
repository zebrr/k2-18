"""
Интеграционные тесты для slicer.py

Проверяет полный workflow:
- Загрузка файлов из data/raw
- Препроцессинг и нарезка
- Сохранение в data/staging
- Валидация выходных данных
"""

import json
import shutil
# Добавляем src в path для импорта
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import slicer
from utils.tokenizer import count_tokens


class TestSlicerIntegration:
    """Интеграционные тесты для slicer.py."""

    @pytest.fixture
    def temp_project_dir(self, monkeypatch):
        """Создает временную структуру проекта для тестов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Создаем структуру директорий
            raw_dir = temp_path / "data" / "raw"
            staging_dir = temp_path / "data" / "staging"
            raw_dir.mkdir(parents=True)
            staging_dir.mkdir(parents=True)

            # Копируем тестовые файлы
            fixtures_dir = Path(__file__).parent / "fixtures"
            for fixture_file in fixtures_dir.glob("*"):
                if fixture_file.is_file():
                    shutil.copy2(fixture_file, raw_dir)

            # Меняем рабочую директорию для slicer.py
            monkeypatch.chdir(temp_path)

            yield temp_path

    def test_full_workflow_no_overlap(self, temp_project_dir):
        """Тест полного workflow без перекрытий."""
        # Конфигурация для тестов
        test_config = {
            "slicer": {
                "max_tokens": 50,  # Маленький размер для гарантии множественных слайсов
                "soft_boundary": True,
                "soft_boundary_max_shift": 20,
                "allowed_extensions": ["json", "txt", "md", "html"],
            }
        }

        # Мокаем конфигурацию
        original_load_config = slicer.load_config
        slicer.load_config = lambda: test_config

        try:
            # Запускаем main функцию slicer'а с пустыми аргументами
            exit_code = slicer.main([])

            # Проверяем успешное завершение
            assert exit_code == 0

            # Проверяем что создались slice файлы
            staging_dir = temp_project_dir / "data" / "staging"
            slice_files = list(staging_dir.glob("*.slice.json"))

            assert len(slice_files) > 0, "Не создались slice файлы"

            # Проверяем формат каждого slice файла
            for slice_file in slice_files:
                with open(slice_file, "r", encoding="utf-8") as f:
                    slice_data = json.load(f)

                # Проверяем обязательные поля
                required_fields = [
                    "id",
                    "order",
                    "source_file",
                    "slug",
                    "text",
                    "slice_token_start",
                    "slice_token_end",
                ]
                for field in required_fields:
                    assert (
                        field in slice_data
                    ), f"Отсутствует поле {field} в {slice_file.name}"

                # Проверяем типы данных
                assert isinstance(slice_data["id"], str)
                assert isinstance(slice_data["order"], int)
                assert isinstance(slice_data["source_file"], str)
                assert isinstance(slice_data["slug"], str)
                assert isinstance(slice_data["text"], str)
                assert isinstance(slice_data["slice_token_start"], int)
                assert isinstance(slice_data["slice_token_end"], int)

                # Проверяем логику токенов
                assert slice_data["slice_token_start"] >= 0
                assert slice_data["slice_token_end"] > slice_data["slice_token_start"]

                # Проверяем что текст не пустой
                assert slice_data["text"].strip(), f"Пустой текст в {slice_file.name}"

                # Проверяем slug формат
                slug = slice_data["slug"]
                assert slug.islower(), f"Slug должен быть в нижнем регистре: {slug}"
                assert " " not in slug, f"Slug не должен содержать пробелы: {slug}"

        finally:
            # Восстанавливаем оригинальную функцию
            slicer.load_config = original_load_config

    def test_deterministic_slice_ids(self, temp_project_dir):
        """Тест детерминированности ID слайсов при повторных запусках."""
        test_config = {
            "slicer": {
                "max_tokens": 30,
                "soft_boundary": False,  # Отключаем для стабильности
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["txt", "md"],
            }
        }

        original_load_config = slicer.load_config
        slicer.load_config = lambda: test_config

        try:
            # Первый запуск
            exit_code1 = slicer.main([])
            assert exit_code1 == 0

            staging_dir = temp_project_dir / "data" / "staging"
            first_run_files = {
                f.name: f.read_text(encoding="utf-8")
                for f in staging_dir.glob("*.slice.json")
            }

            # Очищаем staging
            for f in staging_dir.glob("*.slice.json"):
                f.unlink()

            # Второй запуск
            exit_code2 = slicer.main([])
            assert exit_code2 == 0

            second_run_files = {
                f.name: f.read_text(encoding="utf-8")
                for f in staging_dir.glob("*.slice.json")
            }

            # Сравниваем результаты
            assert (
                first_run_files.keys() == second_run_files.keys()
            ), "Разные наборы файлов"

            for filename in first_run_files:
                first_data = json.loads(first_run_files[filename])
                second_data = json.loads(second_run_files[filename])

                # ID должны быть идентичными
                assert first_data["id"] == second_data["id"], f"Разные ID в {filename}"
                assert (
                    first_data["order"] == second_data["order"]
                ), f"Разный order в {filename}"

        finally:
            slicer.load_config = original_load_config

    def test_slug_generation(self, temp_project_dir):
        """Тест генерации slug для разных типов файлов."""
        # Создаем дополнительные тестовые файлы с разными именами
        raw_dir = temp_project_dir / "data" / "raw"

        test_files = {
            "Алгоритмы и Структуры.txt": "Тестовое содержимое с кириллицей.",
            "My Course Chapter 1.md": "Test content in English.",
            "python-basics.html": "<p>HTML content with hyphens.</p>",
        }

        for filename, content in test_files.items():
            (raw_dir / filename).write_text(content, encoding="utf-8")

        test_config = {
            "slicer": {
                "max_tokens": 100,
                "soft_boundary": False,
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["txt", "md", "html"],
            }
        }

        original_load_config = slicer.load_config
        slicer.load_config = lambda: test_config

        try:
            exit_code = slicer.main([])
            assert exit_code == 0

            staging_dir = temp_project_dir / "data" / "staging"
            slice_files = list(staging_dir.glob("*.slice.json"))

            # Собираем все созданные slug'и
            slugs_by_source = {}
            for slice_file in slice_files:
                with open(slice_file, "r", encoding="utf-8") as f:
                    slice_data = json.load(f)

                source_file = slice_data["source_file"]
                slug = slice_data["slug"]

                if source_file not in slugs_by_source:
                    slugs_by_source[source_file] = slug
                else:
                    # Все слайсы одного файла должны иметь одинаковый slug
                    assert slugs_by_source[source_file] == slug

            # Проверяем ожидаемые slug'и
            expected_slugs = {
                "Алгоритмы и Структуры.txt": "algoritmy_i_struktury",
                "My Course Chapter 1.md": "my_course_chapter_1",
                "python-basics.html": "python-basics",
            }

            for source_file, expected_slug in expected_slugs.items():
                assert source_file in slugs_by_source, f"Не найден файл {source_file}"
                actual_slug = slugs_by_source[source_file]
                assert (
                    actual_slug == expected_slug
                ), f"Неверный slug для {source_file}: {actual_slug} != {expected_slug}"

        finally:
            slicer.load_config = original_load_config

    def test_empty_raw_directory(self, temp_project_dir):
        """Тест обработки пустой директории raw."""
        # Очищаем raw директорию
        raw_dir = temp_project_dir / "data" / "raw"
        for f in raw_dir.iterdir():
            if f.is_file():
                f.unlink()

        test_config = {
            "slicer": {
                "max_tokens": 100,
                "soft_boundary": False,
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["txt"],
            }
        }

        original_load_config = slicer.load_config
        slicer.load_config = lambda: test_config

        try:
            exit_code = slicer.main([])

            # Должно завершиться успешно, но без создания файлов
            assert exit_code == 0

            staging_dir = temp_project_dir / "data" / "staging"
            slice_files = list(staging_dir.glob("*.slice.json"))
            assert (
                len(slice_files) == 0
            ), "Не должно быть slice файлов для пустой директории"

        finally:
            slicer.load_config = original_load_config

    def test_unsupported_file_types(self, temp_project_dir):
        """Тест обработки неподдерживаемых типов файлов."""
        raw_dir = temp_project_dir / "data" / "raw"

        # Создаем файлы неподдерживаемых типов
        (raw_dir / "document.pdf").write_text("PDF content", encoding="utf-8")
        (raw_dir / "image.jpg").write_bytes(b"fake jpg content")
        (raw_dir / "valid.txt").write_text("Valid text content", encoding="utf-8")

        test_config = {
            "slicer": {
                "max_tokens": 100,
                "soft_boundary": False,
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["txt"],  # Только txt поддерживается
            }
        }

        original_load_config = slicer.load_config
        slicer.load_config = lambda: test_config

        try:
            exit_code = slicer.main([])
            assert exit_code == 0

            staging_dir = temp_project_dir / "data" / "staging"
            slice_files = list(staging_dir.glob("*.slice.json"))

            # Должен быть создан только один slice для valid.txt
            assert len(slice_files) > 0, "Должен быть создан slice для valid.txt"

            # Проверяем что обработан только поддерживаемый файл
            processed_sources = set()
            for slice_file in slice_files:
                with open(slice_file, "r", encoding="utf-8") as f:
                    slice_data = json.load(f)
                processed_sources.add(slice_data["source_file"])

            assert "valid.txt" in processed_sources
            assert "document.pdf" not in processed_sources
            assert "image.jpg" not in processed_sources

        finally:
            slicer.load_config = original_load_config

    def test_token_count_consistency(self, temp_project_dir):
        """Тест консистентности подсчета токенов."""
        # Создаем файл с известным содержимым
        raw_dir = temp_project_dir / "data" / "raw"
        test_content = "Это тестовый файл для проверки консистентности подсчета токенов при разбиении на слайсы."
        (raw_dir / "test_tokens.txt").write_text(test_content, encoding="utf-8")

        test_config = {
            "slicer": {
                "max_tokens": 10,  # Маленький размер для множественных слайсов
                "soft_boundary": False,
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["txt"],
            }
        }

        original_load_config = slicer.load_config
        slicer.load_config = lambda: test_config

        try:
            exit_code = slicer.main([])
            assert exit_code == 0

            # Подсчитываем токены в исходном файле
            original_token_count = count_tokens(test_content)

            # Собираем все токены из слайсов
            staging_dir = temp_project_dir / "data" / "staging"
            slice_files = list(staging_dir.glob("*.slice.json"))

            total_tokens_from_slices = 0
            for slice_file in slice_files:
                with open(slice_file, "r", encoding="utf-8") as f:
                    slice_data = json.load(f)

                if slice_data["source_file"] == "test_tokens.txt":
                    token_count = (
                        slice_data["slice_token_end"] - slice_data["slice_token_start"]
                    )
                    total_tokens_from_slices += token_count

            # Проверяем что общее количество токенов совпадает
            assert (
                total_tokens_from_slices == original_token_count
            ), f"Несоответствие токенов: {total_tokens_from_slices} != {original_token_count}"

        finally:
            slicer.load_config = original_load_config
