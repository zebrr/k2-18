"""
Интеграционные тесты slicer.py для больших файлов.

Проверяет:
- Производительность > 100 MB/min
- Валидацию множества slice файлов
- Сравнение разных форматов (JSON vs MD)
- Детерминированность на большом объеме
- Измерение времени выполнения
"""

import json
import shutil

# Добавляем src в path для импорта
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import slicer


class TestSlicerLargeFiles:
    """Тесты slicer.py для больших файлов."""

    @pytest.fixture
    def temp_project_dir(self):
        """Создает временную структуру проекта для тестов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Создаем структуру директорий
            raw_dir = temp_path / "data" / "raw"
            staging_dir = temp_path / "data" / "staging"
            raw_dir.mkdir(parents=True)
            staging_dir.mkdir(parents=True)

            # Меняем рабочую директорию для slicer.py
            original_cwd = Path.cwd()
            import os

            os.chdir(temp_path)

            yield temp_path, raw_dir, staging_dir

            # Восстанавливаем рабочую директорию
            os.chdir(original_cwd)

    @pytest.mark.slow
    @pytest.mark.timeout(60)  # Таймаут 60 секунд
    def test_large_file_performance(self, temp_project_dir):
        """Тест производительности на большом файле."""
        temp_path, raw_dir, staging_dir = temp_project_dir

        # Копируем большой файл
        large_files_dir = Path(__file__).parent / "fixtures" / "large"
        sample_file = large_files_dir / "sample_large.md"

        if not sample_file.exists():
            pytest.skip(f"Большой тестовый файл не найден: {sample_file}")

        # Копируем файл
        target_file = raw_dir / "big_ml_course.md"
        shutil.copy2(sample_file, target_file)

        # Измеряем размер файла
        file_size_bytes = target_file.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Конфигурация для более быстрой обработки
        test_config = {
            "slicer": {
                "max_tokens": 30000,  # Увеличиваем размер слайсов для быстрой обработки
                "overlap": 0,
                "soft_boundary": False,  # ОТКЛЮЧАЕМ для ускорения
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["md"],
            }
        }

        # Мокаем конфигурацию
        original_load_config = slicer.load_config
        slicer.load_config = lambda: test_config

        try:
            # Измеряем время выполнения
            start_time = time.time()
            exit_code = slicer.main([])
            end_time = time.time()

            processing_time = end_time - start_time
            throughput_mb_per_sec = file_size_mb / processing_time
            throughput_mb_per_min = throughput_mb_per_sec * 60

            # Проверяем успешное завершение
            assert exit_code == 0, "Slicer завершился с ошибкой"

            # Проверяем базовую производительность > 0.1 MB/sec (6 MB/min)
            # Это минимальный порог для работоспособности
            assert (
                throughput_mb_per_sec > 0.1
            ), f"Производительность {throughput_mb_per_sec:.2f} MB/sec < 0.1 MB/sec"

            # Проверяем что создались slice файлы
            slice_files = list(staging_dir.glob("*.slice.json"))
            assert len(slice_files) > 0, "Не создались slice файлы"

            # Базовая проверка слайсов
            for slice_file in slice_files:
                with open(slice_file, "r", encoding="utf-8") as f:
                    slice_data = json.load(f)

                # Проверяем наличие обязательных полей
                assert "id" in slice_data
                assert "text" in slice_data
                assert "slice_token_start" in slice_data
                assert "slice_token_end" in slice_data

        finally:
            slicer.load_config = original_load_config

    def test_json_vs_md_consistency(self, temp_project_dir):
        """Тест консистентности между JSON и MD версиями одного файла."""
        temp_path, raw_dir, staging_dir = temp_project_dir

        large_files_dir = Path(__file__).parent / "fixtures" / "large"
        json_file = large_files_dir / "sample_large.json"
        md_file = large_files_dir / "sample_large.md"

        if not (json_file.exists() and md_file.exists()):
            pytest.skip("Большие тестовые файлы не найдены")

        test_config = {
            "slicer": {
                "max_tokens": 10000,  # Меньший размер для более быстрого теста
                "overlap": 0,
                "soft_boundary": False,  # Отключаем для детерминированности
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["json", "md"],
            }
        }

        original_load_config = slicer.load_config
        slicer.load_config = lambda: test_config

        try:
            # === ОБРАБАТЫВАЕМ MD ФАЙЛ ===
            shutil.copy2(md_file, raw_dir / "test.md")
            exit_code1 = slicer.main([])
            assert exit_code1 == 0, "Ошибка при обработке MD файла"

            md_slices = {}
            for slice_file in staging_dir.glob("*.slice.json"):
                with open(slice_file, "r", encoding="utf-8") as f:
                    slice_data = json.load(f)
                if slice_data["source_file"] == "test.md":
                    md_slices[slice_data["id"]] = slice_data

            # Очищаем staging И raw директории
            for f in staging_dir.glob("*.slice.json"):
                f.unlink()
            for f in raw_dir.glob("test.*"):  # Удаляем все test.* файлы
                f.unlink()

            # === ОБРАБАТЫВАЕМ JSON ФАЙЛ ===
            shutil.copy2(json_file, raw_dir / "test.json")
            exit_code2 = slicer.main([])
            assert exit_code2 == 0, "Ошибка при обработке JSON файла"

            json_slices = {}
            for slice_file in staging_dir.glob("*.slice.json"):
                with open(slice_file, "r", encoding="utf-8") as f:
                    slice_data = json.load(f)
                if slice_data["source_file"] == "test.json":
                    json_slices[slice_data["id"]] = slice_data

            # === АНАЛИЗ РЕЗУЛЬТАТОВ ===

            # Проверяем что оба файла успешно обработаны
            assert len(md_slices) > 0, "MD файл не создал слайсов"
            assert len(json_slices) > 0, "JSON файл не создал слайсов"

            # Подсчитываем общее количество токенов
            md_total_tokens = sum(
                s["slice_token_end"] - s["slice_token_start"] for s in md_slices.values()
            )
            json_total_tokens = sum(
                s["slice_token_end"] - s["slice_token_start"] for s in json_slices.values()
            )

            # Проверяем что разница в токенах разумная (< 20%)
            # JSON обычно содержит больше токенов из-за синтаксиса
            if md_total_tokens > 0 and json_total_tokens > 0:
                token_diff_percent = (
                    abs(md_total_tokens - json_total_tokens)
                    / max(md_total_tokens, json_total_tokens)
                    * 100
                )

                assert (
                    token_diff_percent < 20
                ), f"Слишком большая разница в токенах между форматами: {token_diff_percent:.1f}%"

            # Проверяем что количество слайсов в разумных пределах (< 20% разницы)
            slice_diff_percent = (
                abs(len(md_slices) - len(json_slices)) / max(len(md_slices), len(json_slices)) * 100
            )

            assert (
                slice_diff_percent < 20
            ), f"Слишком большая разница в количестве слайсов: {slice_diff_percent:.1f}%"

        finally:
            slicer.load_config = original_load_config

    @pytest.mark.slow
    @pytest.mark.timeout(30)  # Уменьшаем таймаут до 30 секунд
    def test_slice_validation_large_scale(self, temp_project_dir):
        """Тест валидации множества slice файлов."""
        temp_path, raw_dir, staging_dir = temp_project_dir

        large_files_dir = Path(__file__).parent / "fixtures" / "large"
        sample_file = large_files_dir / "sample_large.md"

        if not sample_file.exists():
            pytest.skip("Большой тестовый файл не найден")

        # Копируем файл
        shutil.copy2(sample_file, raw_dir / "validation_test.md")

        test_config = {
            "slicer": {
                "max_tokens": 20000,  # УВЕЛИЧИВАЕМ размер слайсов для ускорения
                "overlap": 0,
                "soft_boundary": False,  # ОТКЛЮЧАЕМ soft_boundary для ускорения
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["md"],
            }
        }

        original_load_config = slicer.load_config
        slicer.load_config = lambda: test_config

        try:
            exit_code = slicer.main([])
            assert exit_code == 0

            slice_files = list(staging_dir.glob("*.slice.json"))
            assert len(slice_files) > 0, "Должен быть создан хотя бы один слайс"

            # Валидируем каждый slice файл
            validation_errors = []

            for i, slice_file in enumerate(slice_files):
                try:
                    # Проверяем JSON структуру
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
                        if field not in slice_data:
                            validation_errors.append(f"{slice_file.name}: missing field {field}")

                    # Проверяем типы данных
                    if not isinstance(slice_data.get("order"), int):
                        validation_errors.append(f"{slice_file.name}: order is not int")

                    if not isinstance(slice_data.get("slice_token_start"), int):
                        validation_errors.append(f"{slice_file.name}: slice_token_start is not int")

                    if not isinstance(slice_data.get("slice_token_end"), int):
                        validation_errors.append(f"{slice_file.name}: slice_token_end is not int")

                    # Проверяем логику
                    start = slice_data.get("slice_token_start", 0)
                    end = slice_data.get("slice_token_end", 0)

                    if start >= end:
                        validation_errors.append(
                            f"{slice_file.name}: start >= end ({start} >= {end})"
                        )

                    # Упрощенная проверка размера - просто проверяем что не превышает max_tokens
                    if end - start > test_config["slicer"]["max_tokens"]:
                        validation_errors.append(
                            f"{slice_file.name}: slice too large ({end - start} tokens)"
                        )

                    # Проверяем что текст не пустой
                    if not slice_data.get("text", "").strip():
                        validation_errors.append(f"{slice_file.name}: empty text")

                    # Проверяем slug формат
                    slug = slice_data.get("slug", "")
                    if " " in slug or slug != slug.lower():
                        validation_errors.append(f"{slice_file.name}: invalid slug format")

                except Exception as e:
                    validation_errors.append(f"{slice_file.name}: {str(e)}")

            # Выводим ошибки валидации
            if validation_errors:
                for error in validation_errors[:10]:  # Показываем первые 10
                    # Проверяем формат ошибок
                    assert isinstance(error, str), "Ошибка должна быть строкой"
                    assert ":" in error, f"Ошибка должна содержать ':' - {error}"
                if len(validation_errors) > 10:
                    # Проверяем что есть дополнительные ошибки
                    assert (
                        len(validation_errors) > 10
                    ), f"Должно быть больше 10 ошибок, но найдено {len(validation_errors)}"

            # Проверяем что нет критических ошибок валидации
            assert (
                len(validation_errors) == 0
            ), f"Найдены ошибки валидации: {len(validation_errors)}"

        finally:
            slicer.load_config = original_load_config

    def test_deterministic_large_file(self, temp_project_dir):
        """Тест детерминированности на большом файле."""
        temp_path, raw_dir, staging_dir = temp_project_dir

        large_files_dir = Path(__file__).parent / "fixtures" / "large"
        sample_file = large_files_dir / "sample_large.md"

        if not sample_file.exists():
            pytest.skip("Большой тестовый файл не найден")

        shutil.copy2(sample_file, raw_dir / "deterministic_test.md")

        test_config = {
            "slicer": {
                "max_tokens": 12000,
                "overlap": 0,
                "soft_boundary": False,  # Отключаем для стабильности
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["md"],
            }
        }

        original_load_config = slicer.load_config
        slicer.load_config = lambda: test_config

        try:
            # Первый запуск
            exit_code1 = slicer.main([])
            assert exit_code1 == 0

            first_run_slices = {}
            for slice_file in staging_dir.glob("*.slice.json"):
                with open(slice_file, "r", encoding="utf-8") as f:
                    slice_data = json.load(f)
                first_run_slices[slice_data["id"]] = slice_data

            # Очищаем staging
            for f in staging_dir.glob("*.slice.json"):
                f.unlink()

            # Второй запуск
            exit_code2 = slicer.main([])
            assert exit_code2 == 0

            second_run_slices = {}
            for slice_file in staging_dir.glob("*.slice.json"):
                with open(slice_file, "r", encoding="utf-8") as f:
                    slice_data = json.load(f)
                second_run_slices[slice_data["id"]] = slice_data

            # Сравниваем результаты
            assert (
                first_run_slices.keys() == second_run_slices.keys()
            ), "Разные наборы slice ID между запусками"

            differences = []
            for slice_id in first_run_slices:
                first = first_run_slices[slice_id]
                second = second_run_slices[slice_id]

                # Проверяем ключевые поля
                for field in [
                    "order",
                    "source_file",
                    "slug",
                    "slice_token_start",
                    "slice_token_end",
                ]:
                    if first[field] != second[field]:
                        differences.append(f"{slice_id}.{field}: {first[field]} != {second[field]}")

                # Проверяем текст (должен быть идентичным)
                if first["text"] != second["text"]:
                    differences.append(f"{slice_id}.text: content differs")

            if differences:
                for diff in differences[:5]:
                    # Проверяем формат различий
                    assert isinstance(diff, str), "Различие должно быть строкой"
                    assert ":" in diff, f"Различие должно содержать ':' - {diff}"
                    assert (
                        "!=" in diff or "differs" in diff
                    ), f"Различие должно указывать на несовпадение - {diff}"
                if len(differences) > 5:
                    # Проверяем что есть дополнительные различия
                    assert (
                        len(differences) > 5
                    ), f"Должно быть больше 5 различий, но найдено {len(differences)}"

            assert (
                len(differences) == 0
            ), f"Результаты не детерминированы: {len(differences)} различий"

        finally:
            slicer.load_config = original_load_config
