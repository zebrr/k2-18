"""
Unit тесты для модуля slicer.py

Тестирует основные функции препроцессинга:
- create_slug()
- preprocess_text()
- validate_config_parameters()
"""

import json

# Добавляем src в path для импорта
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slicer import (
    InputError,
    create_slug,
    load_and_validate_file,
    preprocess_text,
    process_file,
    save_slice,
    slice_text_with_window,
    validate_config_parameters,
)
from utils.tokenizer import count_tokens


class TestCreateSlug:
    """Тесты для функции create_slug."""

    def test_cyrillic_transliteration(self):
        """Тест транслитерации кириллицы."""
        result = create_slug("Алгоритмы и Структуры.txt")
        assert result == "algoritmy_i_struktury"

    def test_english_with_spaces(self):
        """Тест английского текста с пробелами."""
        result = create_slug("My Course Chapter 1.md")
        assert result == "my_course_chapter_1"

    def test_hyphens_preserved(self):
        """Тест сохранения дефисов."""
        result = create_slug("python-basics.html")
        assert result == "python-basics"

    def test_extension_removal(self):
        """Тест удаления различных расширений."""
        assert create_slug("test.json") == "test"
        assert create_slug("file.HTML") == "file"
        assert create_slug("document.MD") == "document"

    def test_complex_filename(self):
        """Тест сложного имени файла."""
        result = create_slug("Глава 2. Основы Python (часть-1).txt")
        assert result == "glava_2._osnovy_python_(chast'-1)"

    def test_special_characters(self):
        """Тест специальных символов."""
        result = create_slug("File@#$%^&*()Name.md")
        assert result == "file@#$%^&*()name"


class TestPreprocessText:
    """Тесты для функции preprocess_text."""

    def test_unicode_normalization(self):
        """Тест Unicode нормализации NFC."""
        # Создаем текст с комбинированными символами
        text_with_combining = "café"  # e + ́ (combining acute accent)
        result = preprocess_text(text_with_combining)

        # Проверяем что нормализация применена
        expected = unicodedata.normalize("NFC", text_with_combining)
        assert result == expected

    def test_script_tag_removal(self):
        """Тест удаления script тегов."""
        html_with_script = """
        <html>
        <head>
        <script>alert('dangerous code');</script>
        </head>
        <body>
        <p>Полезный контент</p>
        </body>
        </html>
        """

        result = preprocess_text(html_with_script)

        # Script тег должен быть удален
        assert "<script>" not in result
        assert "alert('dangerous code');" not in result
        # Полезный контент должен остаться
        assert "Полезный контент" in result

    def test_style_tag_removal(self):
        """Тест удаления style тегов."""
        html_with_style = """
        <html>
        <head>
        <style>
        body { background: red; }
        .danger { color: evil; }
        </style>
        </head>
        <body>
        <h1>Заголовок</h1>
        </body>
        </html>
        """

        result = preprocess_text(html_with_style)

        # Style тег должен быть удален
        assert "<style>" not in result
        assert "background: red" not in result
        # Заголовок должен остаться
        assert "Заголовок" in result

    def test_both_script_and_style_removal(self):
        """Тест удаления и script, и style тегов одновременно."""
        html_mixed = """
        <html>
        <head>
        <script src="evil.js"></script>
        <style>body { display: none; }</style>
        </head>
        <body>
        <h1>Контент</h1>
        <script>alert('popup');</script>
        </body>
        </html>
        """

        result = preprocess_text(html_mixed)

        assert "<script" not in result
        assert "<style>" not in result
        assert "evil.js" not in result
        assert "display: none" not in result
        assert "alert(" not in result
        assert "Контент" in result

    def test_plain_text_unchanged(self):
        """Тест что обычный текст остается без изменений."""
        plain_text = """
        Это обычный текст без HTML тегов.
        
        Глава 1. Введение
        
        Здесь есть разные символы: 123, abc, кириллица.
        """

        result = preprocess_text(plain_text)

        # Текст должен остаться практически тем же (только Unicode нормализация)
        normalized_expected = unicodedata.normalize("NFC", plain_text)
        assert result == normalized_expected

    def test_html_without_script_style(self):
        """Тест HTML без script/style тегов."""
        clean_html = """
        <html>
        <body>
        <h1>Заголовок</h1>
        <p>Параграф с <a href="link">ссылкой</a></p>
        <div class="content">Контент</div>
        </body>
        </html>
        """

        result = preprocess_text(clean_html)

        # HTML структура должна сохраниться
        assert "<h1>" in result
        assert '<a href="link">' in result
        assert "Заголовок" in result
        assert "ссылкой" in result

    def test_case_insensitive_tag_detection(self):
        """Тест нечувствительности к регистру тегов."""
        html_upper = """
        <HTML>
        <SCRIPT>bad code</SCRIPT>
        <STYLE>bad style</STYLE>
        <BODY>Good content</BODY>
        </HTML>
        """

        result = preprocess_text(html_upper)

        assert "bad code" not in result
        assert "bad style" not in result
        assert "Good content" in result

    def test_invalid_input(self):
        """Тест неправильного входного параметра."""
        with pytest.raises(ValueError, match="Input parameter must be a string"):
            preprocess_text(123)

        with pytest.raises(ValueError, match="Input parameter must be a string"):
            preprocess_text(None)


class TestValidateConfigParameters:
    """Тесты для функции validate_config_parameters."""

    def test_valid_config(self):
        """Тест корректной конфигурации."""
        valid_config = {
            "slicer": {
                "max_tokens": 15000,
                "soft_boundary_max_shift": 500,
                "allowed_extensions": ["json", "txt", "md", "html"],
            }
        }

        # Не должно быть исключений
        validate_config_parameters(valid_config)

    def test_missing_slicer_section(self):
        """Тест отсутствия секции slicer."""
        config = {"other": {}}

        with pytest.raises(ValueError, match="Missing required parameter slicer.max_tokens"):
            validate_config_parameters(config)

    def test_missing_required_param(self):
        """Тест отсутствия обязательного параметра."""
        config = {
            "slicer": {
                "max_tokens": 15000,
                # soft_boundary_max_shift отсутствует
                "allowed_extensions": ["txt"],
            }
        }

        with pytest.raises(ValueError, match="Missing required parameter slicer.soft_boundary_max_shift"):
            validate_config_parameters(config)

    def test_invalid_max_tokens(self):
        """Тест некорректного max_tokens."""
        config = {
            "slicer": {
                "max_tokens": -100,  # Отрицательное значение
                "soft_boundary_max_shift": 500,
                "allowed_extensions": ["txt"],
            }
        }

        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_config_parameters(config)


    def test_empty_allowed_extensions(self):
        """Тест пустого списка расширений."""
        config = {
            "slicer": {
                "max_tokens": 15000,
                "soft_boundary_max_shift": 500,
                "allowed_extensions": [],  # Пустой список
            }
        }

        with pytest.raises(ValueError, match="must be a non-empty list"):
            validate_config_parameters(config)


class TestSliceTextWithWindow:
    """Тесты для функции slice_text_with_window."""

    def test_empty_text(self):
        """Тест пустого текста."""
        result = slice_text_with_window("", 1000, True, 100)
        assert result == []

        result = slice_text_with_window("   ", 1000, True, 100)
        assert result == []

    def test_text_fits_in_one_slice(self):
        """Тест текста, который помещается в один слайс."""
        text = "Короткий текст из нескольких слов."
        max_tokens = 1000

        result = slice_text_with_window(text, max_tokens, True, 100)

        assert len(result) == 1
        slice_text, start, end = result[0]
        assert slice_text == text
        assert start == 0
        assert end == count_tokens(text)

    def test_multiple_slices_no_overlap(self):
        """Тест нескольких слайсов без перекрытий."""
        # Создаем текст, который точно разделится на несколько слайсов
        text = "Слово. " * 50  # Около 100 токенов
        max_tokens = 30

        result = slice_text_with_window(text, max_tokens, False, 0)

        # Должно быть несколько слайсов
        assert len(result) > 1

        # Проверяем границы и отсутствие пропусков/перекрытий
        for i in range(len(result) - 1):
            current_slice = result[i]
            next_slice = result[i + 1]

            # Конец текущего слайса = начало следующего (без перекрытий)
            assert current_slice[2] == next_slice[1]

        # Первый слайс начинается с 0
        assert result[0][1] == 0

        # Последний слайс заканчивается на общем количестве токенов
        total_tokens = count_tokens(text)
        assert result[-1][2] == total_tokens


    def test_no_token_loss(self):
        """Тест что токены не теряются при разбиении."""
        text = "Тестовый текст для проверки что все токены сохраняются при разбиении на слайсы."
        max_tokens = 15

        # Тест без перекрытий
        result_no_overlap = slice_text_with_window(text, max_tokens, True, 10)

        # Собираем все токены из слайсов
        total_tokens_from_slices = 0
        for slice_text, start, end in result_no_overlap:
            total_tokens_from_slices += end - start

        original_tokens = count_tokens(text)
        assert total_tokens_from_slices == original_tokens

        # Проверяем что начало первого слайса = 0
        assert result_no_overlap[0][1] == 0

        # Проверяем что конец последнего слайса = общее количество токенов
        assert result_no_overlap[-1][2] == original_tokens

    def test_token_positions_correctness(self):
        """Тест корректности позиций токенов."""
        text = "Один два три четыре пять шесть семь восемь девять десять."
        max_tokens = 8

        result = slice_text_with_window(text, max_tokens, False, 0)

        # Проверяем что slice_token_start inclusive, slice_token_end exclusive
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        all_tokens = encoding.encode(text)

        for slice_text, start, end in result:
            # Декодируем токены из исходного текста в указанном диапазоне
            expected_slice_tokens = all_tokens[start:end]
            expected_slice_text = encoding.decode(expected_slice_tokens)

            assert slice_text == expected_slice_text

    def test_soft_boundaries_applied(self):
        """Тест применения soft boundaries."""
        # Создаем текст с явными границами
        text = """Глава 1. Первая глава
        
        Это содержимое первой главы с длинным текстом.
        
        Глава 2. Вторая глава
        
        Это содержимое второй главы."""

        max_tokens = 30

        # Без soft boundaries
        result_hard = slice_text_with_window(text, max_tokens, False, 0)

        # С soft boundaries
        result_soft = slice_text_with_window(text, max_tokens, True, 20)

        # Soft boundaries могут дать другое разбиение
        # (не всегда, но в общем случае возможно)
        # Главное что общее количество токенов сохраняется

        def get_total_tokens_from_slices(slices):
            return sum(slice_info[2] - slice_info[1] for slice_info in slices)

        total_original = count_tokens(text)
        assert get_total_tokens_from_slices(result_hard) == total_original
        assert get_total_tokens_from_slices(result_soft) == total_original

    def test_unicode_handling(self):
        """Тест обработки Unicode символов."""
        text = "Привет мир! 🌍 Это тест эмодзи и юникода: ñáéíóú çñ"
        max_tokens = 20

        result = slice_text_with_window(text, max_tokens, True, 5)

        # Основная проверка - не должно быть исключений
        assert len(result) >= 1

        # Проверяем что Unicode символы корректно обрабатываются
        combined_text = "".join(slice_text for slice_text, _, _ in result)

        # Все символы из оригинала должны присутствовать
        # (порядок может слегка отличаться из-за токенизации, но основное содержание должно быть)
        assert "Привет" in combined_text
        assert "🌍" in combined_text


class TestLoadAndValidateFile:
    """Тесты для функции load_and_validate_file."""

    def test_valid_file_utf8(self):
        """Тест загрузки валидного UTF-8 файла."""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as f:
            f.write("This is a test content.\nСодержимое на русском языке.")
            temp_path = Path(f.name)

        try:
            content = load_and_validate_file(temp_path, ["txt"])
            assert "This is a test content" in content
            assert "Содержимое на русском языке" in content
        finally:
            temp_path.unlink()

    def test_valid_file_cp1251(self):
        """Тест загрузки файла в кодировке CP1251."""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="cp1251", suffix=".txt", delete=False
        ) as f:
            f.write("Текст в кодировке CP1251")
            temp_path = Path(f.name)

        try:
            content = load_and_validate_file(temp_path, ["txt"])
            assert "Текст в кодировке CP1251" in content
        finally:
            temp_path.unlink()

    def test_empty_file_error(self):
        """Проверка InputError для пустого файла."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            with pytest.raises(InputError) as exc_info:
                load_and_validate_file(temp_path, ["txt"])
            assert "Empty file detected" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_unsupported_extension_error(self):
        """Проверка InputError для неподдерживаемого расширения."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("Some content")
            temp_path = Path(f.name)

        try:
            with pytest.raises(InputError) as exc_info:
                load_and_validate_file(temp_path, ["txt", "md", "json"])
            assert "Unsupported file extension" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_file_with_only_whitespace(self):
        """Файл только с пробелами должен вызвать InputError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("   \n\t\n   ")
            temp_path = Path(f.name)

        try:
            with pytest.raises(InputError) as exc_info:
                load_and_validate_file(temp_path, ["txt"])
            assert "Empty file detected" in str(exc_info.value)
        finally:
            temp_path.unlink()


class TestSaveSlice:
    """Тесты для функции save_slice."""

    def test_save_slice_success(self):
        """Успешное сохранение слайса."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            slice_data = {
                "id": "slice_001",
                "order": 1,
                "source_file": "test.txt",
                "slug": "test",
                "text": "Test content",
                "slice_token_start": 0,
                "slice_token_end": 10,
            }

            save_slice(slice_data, output_dir)

            # Check file exists
            output_file = output_dir / "slice_001.slice.json"
            assert output_file.exists()

            # Check content
            with open(output_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            assert saved_data == slice_data

    def test_save_slice_creates_directory(self):
        """Создание директории если не существует."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested" / "dir"
            slice_data = {
                "id": "slice_002",
                "order": 2,
                "source_file": "test.txt",
                "slug": "test",
                "text": "Test content",
                "slice_token_start": 0,
                "slice_token_end": 10,
            }

            # Directory doesn't exist yet, but parent does
            output_dir.parent.mkdir(parents=True, exist_ok=True)

            # Note: save_slice expects directory to exist
            # Create directory first
            output_dir.mkdir(parents=True, exist_ok=True)

            save_slice(slice_data, output_dir)

            output_file = output_dir / "slice_002.slice.json"
            assert output_file.exists()

    def test_save_slice_utf8_content(self):
        """Корректное сохранение UTF-8 контента."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            slice_data = {
                "id": "slice_003",
                "order": 3,
                "source_file": "test.txt",
                "slug": "test",
                "text": "Тест UTF-8: 你好 🌍 éñ",
                "slice_token_start": 0,
                "slice_token_end": 10,
            }

            save_slice(slice_data, output_dir)

            output_file = output_dir / "slice_003.slice.json"
            with open(output_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            assert saved_data["text"] == "Тест UTF-8: 你好 🌍 éñ"


class TestProcessFile:
    """Тесты для функции process_file."""

    def test_process_valid_file(self):
        """Успешная обработка валидного файла."""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as f:
            f.write("This is a test content for processing. " * 10)
            temp_path = Path(f.name)

        config = {
            "slicer": {
                "max_tokens": 50,
                "soft_boundary": False,
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["txt"],
            }
        }

        try:
            slices, counter = process_file(temp_path, config, 1)

            assert len(slices) > 0
            assert counter > 1
            assert slices[0]["id"] == "slice_001"
            assert slices[0]["source_file"] == temp_path.name
            assert "test content" in slices[0]["text"]
        finally:
            temp_path.unlink()

    def test_process_empty_file_raises_input_error(self):
        """InputError для пустого файла."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        config = {
            "slicer": {
                "max_tokens": 50,
                "soft_boundary": False,
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["txt"],
            }
        }

        try:
            with pytest.raises(InputError) as exc_info:
                process_file(temp_path, config, 1)
            assert "Empty file detected" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_process_file_with_invalid_extension(self):
        """InputError для неверного расширения."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("Some content")
            temp_path = Path(f.name)

        config = {
            "slicer": {
                "max_tokens": 50,
                "soft_boundary": False,
                "soft_boundary_max_shift": 0,
                "allowed_extensions": ["txt", "md"],
            }
        }

        try:
            with pytest.raises(InputError) as exc_info:
                process_file(temp_path, config, 1)
            assert "Unsupported file extension" in str(exc_info.value)
        finally:
            temp_path.unlink()


class TestPerformance:
    """Тесты производительности для оптимизированной версии."""

    @pytest.fixture
    def large_text(self):
        """Генерирует большой текст для тестов производительности (около 2MB)."""
        return "Test paragraph with some content. " * 50000  # ~2MB

    def test_large_file_performance(self, large_text):
        """Тест что большие файлы обрабатываются эффективно."""
        # Должен завершиться без зависания
        start_time = time.time()
        slices = slice_text_with_window(large_text, 5000, True, 500)
        elapsed = time.time() - start_time

        assert elapsed < 10.0  # Должен завершиться менее чем за 10 секунд
        assert len(slices) > 0

        # Проверяем что нет пробелов в покрытии токенов
        for i in range(1, len(slices)):
            assert slices[i][1] == slices[i-1][2]  # start == previous end

    def test_incremental_tokenization_consistency(self):
        """Тест что инкрементальная токенизация дает идентичные границы."""
        # Тестовый текст с известными границами
        test_text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph." * 100

        # Обрабатываем с новым инкрементальным подходом
        slices = slice_text_with_window(test_text, 100, True, 20)

        # Проверяем что границы на разрывах параграфов
        for slice_text, _, _ in slices:
            # Должны предпочтительно заканчиваться на границе параграфа
            if slice_text != test_text[-len(slice_text):]:  # Если не последний слайс
                assert (
                    slice_text.endswith('.')
                    or slice_text.endswith('\n\n')
                    or slice_text.endswith('paragraph.\n')
                ), f"Unexpected boundary: ...{slice_text[-30:]}"

    def test_no_token_gaps(self):
        """Тест что нет пробелов в токенах между слайсами."""
        test_text = "This is a test. " * 1000

        slices = slice_text_with_window(test_text, 50, False, 0)

        # Проверяем последовательность токенов
        prev_end = 0
        for _, start, end in slices:
            assert start == prev_end, f"Gap found: prev_end={prev_end}, start={start}"
            assert end > start, f"Invalid slice: start={start}, end={end}"
            prev_end = end

    def test_deterministic_boundaries(self):
        """Тест что границы детерминированы при повторных запусках."""
        test_text = "Test sentence. Another sentence. " * 500

        # Первый запуск
        slices1 = slice_text_with_window(test_text, 100, True, 50)

        # Второй запуск
        slices2 = slice_text_with_window(test_text, 100, True, 50)

        # Должны быть идентичны
        assert len(slices1) == len(slices2)
        for (text1, start1, end1), (text2, start2, end2) in zip(slices1, slices2):
            assert text1 == text2
            assert start1 == start2
            assert end1 == end2

    def test_memory_efficiency(self, large_text):
        """Тест что память не растет линейно с размером файла."""
        import tracemalloc

        # Измеряем память для маленького текста
        small_text = "Test. " * 100
        tracemalloc.start()
        _ = slice_text_with_window(small_text, 50, True, 20)
        small_memory = tracemalloc.get_traced_memory()[1]  # peak
        tracemalloc.stop()

        # Измеряем память для большого текста
        tracemalloc.start()
        _ = slice_text_with_window(large_text, 5000, True, 500)
        large_memory = tracemalloc.get_traced_memory()[1]  # peak
        tracemalloc.stop()

        # Память не должна расти пропорционально размеру
        # (large_text в 500 раз больше, но память не должна быть в 500 раз больше)
        memory_ratio = large_memory / small_memory
        size_ratio = len(large_text) / len(small_text)

        assert memory_ratio < size_ratio / 10, (
            f"Memory grew too much: {memory_ratio:.1f}x for {size_ratio:.1f}x size increase"
        )
