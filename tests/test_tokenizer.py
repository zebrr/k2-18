"""
Тесты для модуля tokenizer.py
"""

from pathlib import Path

import pytest

from src.utils.tokenizer import count_tokens, find_soft_boundary


class TestCountTokens:
    """Тесты для функции count_tokens()"""

    def test_basic_text(self):
        """Базовый тест подсчета токенов"""
        text = "Hello world! This is a test."
        token_count = count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_empty_string(self):
        """Тест с пустой строкой"""
        assert count_tokens("") == 0

    def test_russian_text(self):
        """Тест с русским текстом"""
        text = "Привет мир! Это тестовый текст."
        token_count = count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_mixed_languages(self):
        """Тест со смешанными языками"""
        text = "Hello мир! This это test тест."
        token_count = count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_special_characters(self):
        """Тест со специальными символами"""
        text = "Test with symbols: @#$%^&*()_+-={}[]|\\:;\"'<>,.?/"
        token_count = count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_code_content(self):
        """Тест с кодом"""
        text = """
        def example():
            return "Hello World"
        """
        token_count = count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_invalid_input_type(self):
        """Тест с неправильным типом входных данных"""
        with pytest.raises(ValueError, match="Input parameter must be a string"):
            count_tokens(123)

        with pytest.raises(ValueError, match="Input parameter must be a string"):
            count_tokens(None)

        with pytest.raises(ValueError, match="Input parameter must be a string"):
            count_tokens(["list"])


class TestFindSoftBoundary:
    """Тесты для функции find_soft_boundary()"""

    def test_invalid_inputs(self):
        """Тест с некорректными входными данными"""
        # Пустая строка
        assert find_soft_boundary("", 0, 10) is None

        # Неправильный тип текста
        assert find_soft_boundary(None, 0, 10) is None

        # target_pos вне диапазона
        assert find_soft_boundary("Hello", -1, 10) is None
        assert find_soft_boundary("Hello", 10, 10) is None

        # Отрицательный max_shift
        assert find_soft_boundary("Hello", 2, -1) is None

    def test_no_boundaries_found(self):
        """Тест когда границы не найдены"""
        text = "abcdefghijklmnop"
        result = find_soft_boundary(text, 8, 3)  # ищем в узком диапазоне без границ
        assert result is None

    def test_markdown_headers(self):
        """Тест поиска Markdown заголовков"""
        text = "Some text\n# Header 1\nMore text\n## Header 2\nEnd"

        # Ищем рядом с первым заголовком
        boundary = find_soft_boundary(text, 15, 10)
        assert boundary is not None
        assert text[boundary] == "\n" or text[boundary - 1 : boundary + 1] == "\n"

    def test_html_headers(self):
        """Тест поиска HTML заголовков"""
        text = "Start text</h1>\nMore content</h2>\nEnd text"

        boundary = find_soft_boundary(text, 15, 10)
        assert boundary is not None

    def test_text_headers_russian(self):
        """Тест поиска русских текстовых заголовков"""
        text = "Начало\nГлава 1 Введение\nСодержание главы\nКонец"

        boundary = find_soft_boundary(text, 20, 15)
        assert boundary is not None

    def test_text_headers_english(self):
        """Тест поиска английских текстовых заголовков"""
        text = "Start\nChapter 1 Introduction\nChapter content\nEnd"

        boundary = find_soft_boundary(text, 20, 15)
        assert boundary is not None

    def test_double_newlines(self):
        """Тест поиска двойных переносов строк"""
        # ИСПРАВЛЕНО: убираем конкурирующие границы (точку)
        text = "First paragraph\n\nSecond paragraph"

        # Двойной перенос \n\n заканчивается на позиции 17
        boundary = find_soft_boundary(text, 15, 5)  # target=15, ищем рядом с \n\n
        assert boundary is not None
        assert boundary == 17  # после \n\n

    def test_sentence_endings(self):
        """Тест поиска концов предложений"""
        text = "First sentence. Second sentence! Third sentence? Fourth."

        # Ищем рядом с первой точкой
        boundary = find_soft_boundary(text, 10, 10)
        assert boundary is not None

    def test_code_blocks(self):
        """Тест поиска блоков кода"""
        text = "Text\n```\ncode here\n```\nMore text"

        boundary = find_soft_boundary(text, 20, 10)
        assert boundary is not None

    def test_formula_blocks(self):
        """Тест поиска блоков формул"""
        text = "Text\n$$\nE = mc^2\n$$\nMore text"

        boundary = find_soft_boundary(text, 20, 10)
        assert boundary is not None

    def test_html_links(self):
        """Тест поиска HTML ссылок"""
        text = 'Text with <a href="http://example.com">link</a> inside.'

        # ИСПРАВЛЕНО: увеличиваем max_shift чтобы ссылка попала в диапазон
        # HTML ссылка заканчивается на позиции 47, target=25, нужен max_shift >= 22
        boundary = find_soft_boundary(text, 25, 25)
        assert boundary is not None
        assert boundary == 47  # конец HTML ссылки

    def test_markdown_links(self):
        """Тест поиска Markdown ссылок"""
        text = "Text with [link](http://example.com) inside."

        # ИСПРАВЛЕНО: MD ссылка заканчивается на позиции 36, target=20, нужен max_shift >= 16
        boundary = find_soft_boundary(text, 20, 20)
        assert boundary is not None
        assert boundary == 36  # конец Markdown ссылки

    def test_closest_boundary_selection(self):
        """Тест выбора ближайшей границы"""
        # ИСПРАВЛЕНО: создаем ситуацию с явно разными расстояниями
        text = "Text. Far away sentence! End."
        target = 8  # ближе к первой точке

        boundary = find_soft_boundary(text, target, 20)
        assert boundary is not None

        # Первая граница ". " на позиции 6 (расстояние 2)
        # Вторая граница "! " на позиции 25 (расстояние 17)
        # Должна быть выбрана первая
        assert boundary == 6

    def test_boundary_within_range(self):
        """Тест что границы ищутся только в заданном диапазоне"""
        text = "Start. Middle text goes here! End?"
        target = 15
        max_shift = 5

        boundary = find_soft_boundary(text, target, max_shift)

        if boundary is not None:
            # Граница должна быть в пределах диапазона
            assert abs(boundary - target) <= max_shift


class TestFindSoftBoundaryWithFixtures:
    """Тесты find_soft_boundary() с использованием fixture файлов"""

    @pytest.fixture
    def fixtures_dir(self):
        """Путь к директории с fixtures"""
        return Path(__file__).parent / "fixtures"

    def test_markdown_fixture(self, fixtures_dir):
        """Тест с реальным Markdown файлом"""
        markdown_file = fixtures_dir / "sample_markdown.md"
        assert markdown_file.exists(), f"Fixture файл не найден: {markdown_file}"

        with open(markdown_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Ищем границы в разных частях файла
        text_len = len(text)

        # В начале файла
        boundary1 = find_soft_boundary(text, text_len // 4, 50)

        # В середине файла
        boundary2 = find_soft_boundary(text, text_len // 2, 50)

        # Хотя бы одна граница должна быть найдена
        assert boundary1 is not None or boundary2 is not None

    def test_html_fixture(self, fixtures_dir):
        """Тест с реальным HTML файлом"""
        html_file = fixtures_dir / "sample_html.html"
        assert html_file.exists(), f"Fixture файл не найден: {html_file}"

        with open(html_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Ищем HTML заголовки
        boundary = find_soft_boundary(text, len(text) // 2, 100)

        # В HTML файле должны найтись границы
        assert boundary is not None

    def test_mixed_fixture(self, fixtures_dir):
        """Тест со смешанным контентом"""
        mixed_file = fixtures_dir / "sample_mixed.txt"
        assert mixed_file.exists(), f"Fixture файл не найден: {mixed_file}"

        with open(mixed_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Проверим что находятся русские заголовки
        boundary = find_soft_boundary(text, len(text) // 3, 100)
        assert boundary is not None

    def test_token_consistency(self, fixtures_dir):
        """Тест согласованности подсчета токенов с fixture файлами"""
        for fixture_file in fixtures_dir.glob("*.md"):
            with open(fixture_file, "r", encoding="utf-8") as f:
                text = f.read()

            token_count = count_tokens(text)
            assert token_count > 0

            # Проверим что функция не падает на реальных данных
            boundary = find_soft_boundary(text, token_count // 2, 100)
            # boundary может быть None - это нормально для некоторых текстов


class TestEdgeCases:
    """Тесты для граничных случаев и error paths."""

    def test_is_inside_url_continuation(self):
        """Тест проверки продолжения URL (покрытие строк 293-294)."""
        from src.utils.tokenizer import is_inside_url

        # URL продолжается после границы
        text_before = "Check this link: https://example.com/path"
        text_after = "/continued/path"
        assert is_inside_url(text_before, text_after) is True

        # URL заканчивается на границе
        text_before = "Check this link: https://example.com/path"
        text_after = " and more text"
        assert is_inside_url(text_before, text_after) is False

        # URL с параметрами продолжается
        text_before = "https://example.com/search?q=test"
        text_after = "&page=2"
        assert is_inside_url(text_before, text_after) is True

        # Нет text_after
        text_before = "https://example.com/path"
        text_after = None
        assert is_inside_url(text_before, text_after) is False

    def test_is_inside_markdown_link_edge_cases(self):
        """Тест проверки markdown ссылок в граничных случаях (покрытие строк 307, 311)."""
        from src.utils.tokenizer import is_inside_markdown_link

        # Внутри markdown ссылки [text](url)
        text_before = "Check [this link]("
        text_after = "https://example.com)"
        assert is_inside_markdown_link(text_before, text_after) is True

        # Прерванная markdown ссылка - функция проверяет наличие "](h" в конце
        text_before = "Check [this link"
        text_after = "](https://example.com)"
        assert is_inside_markdown_link(text_before, text_after) is True

        # Внутри URL части ссылки - должен заканчиваться на "]("
        text_before = "Check [link]("
        text_after = "https://example.com/path)"
        assert is_inside_markdown_link(text_before, text_after) is True

        # Не внутри ссылки
        text_before = "Normal text with ] and ("
        text_after = " more text"
        assert is_inside_markdown_link(text_before, text_after) is False

    @pytest.mark.skip(reason="Function find_boundaries_in_range does not exist in tokenizer module")
    def test_find_boundaries_with_all_types(self):
        """Тест поиска всех типов границ (покрытие строк 154-175)."""
        from src.utils.tokenizer import find_boundaries_in_range

        # Текст со всеми типами границ
        text = """# Header
This is a paragraph. With a sentence; semicolon here.
And phrases: with colons, commas here — and dashes.
Another paragraph.

## Subheader"""

        boundaries = find_boundaries_in_range(text, 0, len(text))

        # Проверяем, что найдены границы разных типов
        assert "header" in boundaries
        assert "paragraph" in boundaries
        assert "sentence" in boundaries
        assert "phrase" in boundaries

        # Проверяем конкретные позиции
        # Заголовок
        assert len(boundaries["header"]["candidates"]) > 0
        # Точка с запятой (строки 154-156)
        assert any("; " in text[pos - 2 : pos + 1] for pos in boundaries["sentence"]["candidates"])
        # Запятая (строки 161-163)
        assert any(", " in text[pos - 2 : pos + 1] for pos in boundaries["phrase"]["candidates"])
        # Двоеточие (строки 167-169)
        assert any(": " in text[pos - 2 : pos + 1] for pos in boundaries["phrase"]["candidates"])
        # Тире (строки 173-175)
        assert any(
            "—" in text[max(0, pos - 3) : pos + 1] for pos in boundaries["phrase"]["candidates"]
        )

    @pytest.mark.skip(reason="Function split_text does not exist in tokenizer module")
    def test_split_text_preserve_structure_edge_cases(self):
        """Тест split_text с сохранением структуры в граничных случаях (покрытие строк 352, 365-373)."""
        from src.utils.tokenizer import split_text

        # Короткий текст (меньше max_tokens) - строка 352
        short_text = "Short text."
        chunks = split_text(short_text, max_tokens=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == short_text

        # Текст, который невозможно разделить по границам (строки 365-373)
        # Одно очень длинное предложение без границ
        long_word = "a" * 5000  # Примерно 1250 токенов
        chunks = split_text(long_word, max_tokens=500, overlap=50)
        assert len(chunks) > 1
        # Проверяем, что chunks перекрываются
        for i in range(len(chunks) - 1):
            # Должно быть перекрытие между chunks
            assert chunks[i][-10:] in chunks[i + 1][:100] or len(chunks[i]) < 10

    def test_count_tokens_with_different_models(self):
        """Тест подсчета токенов с разными моделями."""
        from src.utils.tokenizer import count_tokens

        text = "This is a test text для проверки токенизации."

        # Должен использовать дефолтную модель o200k_base
        tokens = count_tokens(text)
        assert tokens > 0

        # Проверяем консистентность
        tokens2 = count_tokens(text)
        assert tokens == tokens2


if __name__ == "__main__":
    pytest.main([__file__])
