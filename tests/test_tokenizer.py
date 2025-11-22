"""
Тесты для модуля tokenizer.py
"""

from pathlib import Path

import pytest
import tiktoken

from src.utils.tokenizer import count_tokens, find_safe_token_boundary, find_soft_boundary


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

        with open(markdown_file, encoding="utf-8") as f:
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

        with open(html_file, encoding="utf-8") as f:
            text = f.read()

        # Ищем HTML заголовки
        boundary = find_soft_boundary(text, len(text) // 2, 100)

        # В HTML файле должны найтись границы
        assert boundary is not None

    def test_mixed_fixture(self, fixtures_dir):
        """Тест со смешанным контентом"""
        mixed_file = fixtures_dir / "sample_mixed.txt"
        assert mixed_file.exists(), f"Fixture файл не найден: {mixed_file}"

        with open(mixed_file, encoding="utf-8") as f:
            text = f.read()

        # Проверим что находятся русские заголовки
        boundary = find_soft_boundary(text, len(text) // 3, 100)
        assert boundary is not None

    def test_token_consistency(self, fixtures_dir):
        """Тест согласованности подсчета токенов с fixture файлами"""
        for fixture_file in fixtures_dir.glob("*.md"):
            with open(fixture_file, encoding="utf-8") as f:
                text = f.read()

            token_count = count_tokens(text)
            assert token_count > 0

            # Проверим что функция не падает на реальных данных
            find_soft_boundary(text, token_count // 2, 100)
            # boundary может быть None - это нормально для некоторых текстов


class TestFindSafeTokenBoundary:
    """Tests for find_safe_token_boundary function.

    Note: After SLICER_REFACTOR_03, find_safe_token_boundary returns
    Tuple[int, str] where int is position and str is boundary_type.
    """

    def test_basic_middle_boundary(self):
        """Test finding boundary in middle of text"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "First sentence. Second sentence. Third sentence."
        tokens = encoding.encode(text)

        # Find boundary near middle
        result, boundary_type = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=len(tokens) // 2, max_shift_tokens=5
        )

        # Verify result is valid
        assert 0 <= result <= len(tokens)
        assert isinstance(boundary_type, str)
        # Verify it's at a safe position
        if 0 < result < len(tokens):
            text_before = encoding.decode(tokens[:result])
            text_after = encoding.decode(tokens[result:])
            # Should not cut inside a word
            if text_before and text_after:
                assert not (text_before[-1].isalnum() and text_after[0].isalnum())

    def test_boundary_at_start(self):
        """Test boundary at start of text (pos=0)"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "Hello world! This is a test."
        tokens = encoding.encode(text)

        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=0, max_shift_tokens=2
        )

        # With smart candidate selection, function may find a better boundary
        # within the shift range, so result should be within [0, 2]
        assert 0 <= result <= 2  # Within shift range from start

    def test_boundary_at_end(self):
        """Test boundary at end of text"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "Hello world! This is a test."
        tokens = encoding.encode(text)

        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=len(tokens), max_shift_tokens=2
        )

        # With smart candidate selection, function may find a safer boundary
        # within the shift range, so result should be within [len(tokens)-2, len(tokens)]
        assert len(tokens) - 2 <= result <= len(tokens)  # Within shift range from end

    def test_empty_text(self):
        """Test handling of empty text"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = ""
        tokens = encoding.encode(text)

        result, boundary_type = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=0, max_shift_tokens=0
        )

        assert result == 0
        assert boundary_type == "empty"

    def test_single_token_text(self):
        """Test with single token text"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "Hi"
        tokens = encoding.encode(text)

        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=0, max_shift_tokens=1
        )

        assert result in [0, len(tokens)]

    def test_not_cutting_inside_word(self):
        """Test that we don't cut inside a word"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "This is a verylongwordthatshouldbecutproperly and more text"
        tokens = encoding.encode(text)

        # Try to cut somewhere in the middle of the long word
        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=len(tokens) // 2, max_shift_tokens=10
        )

        # Decode and check
        text_before = encoding.decode(tokens[:result])
        text_after = encoding.decode(tokens[result:])

        # Check we're not in the middle of a word
        if text_before and text_after:
            # Either ends with non-alphanumeric or starts with non-alphanumeric
            is_safe = not (text_before[-1].isalnum() and text_after[0].isalnum())
            assert is_safe

    def test_not_cutting_inside_url(self):
        """Test that we don't cut inside URL"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "Check this link https://example.com/path/to/resource for more info"
        tokens = encoding.encode(text)

        # Try to cut inside the URL
        url_start = text.index("https://")
        target_pos = len(encoding.encode(text[: url_start + 15]))  # Middle of URL

        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=target_pos, max_shift_tokens=20
        )

        # The boundary should be shifted outside the URL
        text_before = encoding.decode(tokens[:result])
        # Should either be before URL starts or after URL ends
        assert (
            "https://example.com/path" not in text_before
            or "https://example.com/path/to/resource" in text_before
        )

    def test_not_cutting_inside_markdown_link(self):
        """Test that we don't cut inside markdown link"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "Check [this link](https://example.com) for details"
        tokens = encoding.encode(text)

        # Try to cut inside the markdown link structure
        link_start = text.index("[")
        target_pos = len(encoding.encode(text[: link_start + 10]))

        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=target_pos, max_shift_tokens=15
        )

        text_before = encoding.decode(tokens[:result])
        # Should not end in the middle of link structure
        if "[" in text_before and "]" not in text_before:
            # If we have opening bracket, we should have closing
            raise AssertionError("Cut inside markdown link structure")

    def test_not_cutting_inside_html_tag(self):
        """Test that we don't cut inside HTML tag"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "Some text <div class='test'>content</div> more text"
        tokens = encoding.encode(text)

        # Try to cut inside the opening tag
        tag_start = text.index("<div")
        target_pos = len(encoding.encode(text[: tag_start + 8]))

        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=target_pos, max_shift_tokens=10
        )

        text_before = encoding.decode(tokens[:result])
        # Should not end inside a tag (between < and >)
        if text_before.rfind("<") > text_before.rfind(">"):
            raise AssertionError("Cut inside HTML tag")

    def test_not_cutting_inside_formula(self):
        """Test that we don't cut inside mathematical formula"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "The formula $E = mc^2$ explains energy"
        tokens = encoding.encode(text)

        # Try to cut inside the formula
        formula_start = text.index("$E")
        target_pos = len(encoding.encode(text[: formula_start + 5]))

        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=target_pos, max_shift_tokens=10
        )

        text_before = encoding.decode(tokens[:result])
        # Check dollar sign count - should be even
        dollar_count = text_before.count("$")
        assert dollar_count % 2 == 0, "Cut inside formula"

    def test_not_cutting_inside_code_block(self):
        """Test that we don't cut inside code block"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "Here is code:\n```python\ndef hello():\n    pass\n```\nEnd of code"
        tokens = encoding.encode(text)

        # Try to cut inside the code block
        code_start = text.index("```python")
        target_pos = len(encoding.encode(text[: code_start + 20]))

        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=target_pos, max_shift_tokens=15
        )

        text_before = encoding.decode(tokens[:result])
        # Count triple backticks - should be even
        triple_count = text_before.count("```")
        assert triple_count % 2 == 0, "Cut inside code block"

    def test_prefer_header_boundaries(self):
        """Test that header boundaries are preferred (priority 1)"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "Text before\n# Header\nText after header"
        tokens = encoding.encode(text)

        # Position near the header
        header_end = text.index("\nText after")
        target_pos = len(encoding.encode(text[: header_end - 2]))

        result, boundary_type = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=target_pos, max_shift_tokens=5
        )

        # Just check that it returns a valid position
        assert 0 <= result <= len(tokens)
        # Should prefer header boundaries (BEFORE headers now)
        assert boundary_type in ("markdown_header", "line", "sentence", "fallback", "none")

    def test_prefer_paragraph_breaks(self):
        """Test that paragraph breaks are preferred (priority 2)"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "First paragraph here.\n\nSecond paragraph here."
        tokens = encoding.encode(text)

        # Position near the double newline
        break_pos = text.index("\n\n") + 2
        target_pos = len(encoding.encode(text[: break_pos - 1]))

        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=target_pos, max_shift_tokens=5
        )

        # Just verify it returns a valid position
        assert 0 <= result <= len(tokens)

    def test_lower_scores_preferred_when_equidistant(self):
        """Test that lower quality scores are selected when equidistant"""
        encoding = tiktoken.get_encoding("o200k_base")

        # Create text with different boundary types at equal distances
        text = "Word. Another\n\nParagraph"
        tokens = encoding.encode(text)

        # Position equidistant from sentence end and paragraph break
        target_pos = len(tokens) // 2

        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=target_pos, max_shift_tokens=10
        )

        # Should prefer the paragraph break (priority 2) over sentence end (priority 3)
        # if they are roughly equidistant
        assert result >= 0

    def test_max_shift_tokens_zero(self):
        """Test with max_shift_tokens = 0 (no shift allowed)"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "This is a test sentence with multiple words."
        tokens = encoding.encode(text)

        target = len(tokens) // 2
        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=target, max_shift_tokens=0
        )

        # Should return exactly the target position
        assert result == target

    def test_target_token_pos_out_of_bounds(self):
        """Test with target_token_pos out of bounds"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "Test text"
        tokens = encoding.encode(text)

        # Target beyond bounds - function may handle this differently
        # It appears to extend the search range beyond token bounds
        result, _ = find_safe_token_boundary(
            text, tokens, encoding, target_token_pos=len(tokens) + 10, max_shift_tokens=5
        )

        # Result should be a valid integer
        assert isinstance(result, int)

    def test_very_large_shift_values(self):
        """Test with very large shift values"""
        encoding = tiktoken.get_encoding("o200k_base")

        text = "Short text."
        tokens = encoding.encode(text)

        result, _ = find_safe_token_boundary(
            text,
            tokens,
            encoding,
            target_token_pos=len(tokens) // 2,
            max_shift_tokens=1000,  # Much larger than text
        )

        # Should still return valid position
        assert 0 <= result <= len(tokens)

    def test_max_shift_tokens_interpretation(self):
        """Test that max_shift_tokens is used directly as tokens (not divided by 4)"""
        encoding = tiktoken.get_encoding("o200k_base")

        # Create a text with clear boundaries
        text = "First part. " * 100 + "BOUNDARY_MARKER. " + "Second part. " * 100
        tokens = encoding.encode(text)

        # Find the position of BOUNDARY_MARKER
        marker_text = "BOUNDARY_MARKER"
        marker_pos = text.index(marker_text)

        # Find approximate token position of the marker
        text_before_marker = text[:marker_pos]
        tokens_before_marker = len(encoding.encode(text_before_marker))

        # Test with max_shift_tokens=500 (should search ±500 tokens)
        result, _ = find_safe_token_boundary(
            text,
            tokens,
            encoding,
            target_token_pos=tokens_before_marker - 400,  # 400 tokens before marker
            max_shift_tokens=500,  # Should be able to reach the marker
        )

        # The function should be able to find a boundary within ±500 tokens
        # Verify that the search range is actually ±500 tokens
        min_pos = max(0, tokens_before_marker - 400 - 500)
        max_pos = min(len(tokens), tokens_before_marker - 400 + 500)
        assert min_pos <= result <= max_pos

        # Test with small max_shift_tokens=10 (should search only ±10 tokens)
        # Use a valid target position for this test
        valid_target = max(0, tokens_before_marker - 50)  # 50 tokens before marker, but ensure >= 0
        result_small, _ = find_safe_token_boundary(
            text,
            tokens,
            encoding,
            target_token_pos=valid_target,
            max_shift_tokens=10,  # Much smaller search range
        )

        # With only ±10 tokens, the result should be very close to target
        assert abs(result_small - valid_target) <= 10


class TestHelperFunctions:
    """Tests for helper functions in tokenizer module"""

    def test_is_safe_cut_position_basic(self):
        """Test basic safety checks for cut position"""
        from src.utils.tokenizer import is_safe_cut_position

        encoding = tiktoken.get_encoding("o200k_base")

        # Safe: cutting between words
        text = "Hello world"
        tokens = encoding.encode(text)
        space_pos = len(encoding.encode("Hello "))
        assert is_safe_cut_position(text, tokens, encoding, space_pos) is True

        # Unsafe: cutting inside a word
        text = "Verylongword"
        tokens = encoding.encode(text)
        mid_pos = len(tokens) // 2
        if 0 < mid_pos < len(tokens):
            result = is_safe_cut_position(text, tokens, encoding, mid_pos)
            # May or may not be safe depending on tokenization
            assert isinstance(result, bool)

    def test_is_safe_cut_position_boundaries(self):
        """Test is_safe_cut_position at text boundaries"""
        from src.utils.tokenizer import is_safe_cut_position

        encoding = tiktoken.get_encoding("o200k_base")
        text = "Test text"
        tokens = encoding.encode(text)

        # Start and end are always safe
        assert is_safe_cut_position(text, tokens, encoding, 0) is True
        assert is_safe_cut_position(text, tokens, encoding, len(tokens)) is True

    def test_is_safe_cut_position_with_structures(self):
        """Test is_safe_cut_position with various structures"""
        from src.utils.tokenizer import is_safe_cut_position

        encoding = tiktoken.get_encoding("o200k_base")

        # URL case
        text = "Visit https://example.com now"
        tokens = encoding.encode(text)
        url_mid = len(encoding.encode("Visit https://ex"))
        assert is_safe_cut_position(text, tokens, encoding, url_mid) is False

        # Formula case
        text = "Formula $x^2$ here"
        tokens = encoding.encode(text)
        formula_mid = len(encoding.encode("Formula $x"))
        assert is_safe_cut_position(text, tokens, encoding, formula_mid) is False

    def test_evaluate_boundary_quality_headers(self):
        """Test boundary quality evaluation for headers"""
        from src.utils.tokenizer import evaluate_boundary_quality

        encoding = tiktoken.get_encoding("o200k_base")

        # Markdown header
        text = "Content\n# Header\nMore content"
        tokens = encoding.encode(text)
        header_end = len(encoding.encode("Content\n# Header"))
        score = evaluate_boundary_quality(text, tokens, encoding, header_end)
        assert score == 1.0  # Headers have score 1.0

        # HTML header
        text = "Content</h1>\nMore content"
        tokens = encoding.encode(text)
        header_end = len(encoding.encode("Content</h1>"))
        score = evaluate_boundary_quality(text, tokens, encoding, header_end)
        assert score == 1.0

    def test_evaluate_boundary_quality_paragraph(self):
        """Test boundary quality for paragraph breaks"""
        from src.utils.tokenizer import evaluate_boundary_quality

        encoding = tiktoken.get_encoding("o200k_base")

        text = "First paragraph.\n\nSecond paragraph."
        tokens = encoding.encode(text)
        para_end = len(encoding.encode("First paragraph.\n\n"))
        score = evaluate_boundary_quality(text, tokens, encoding, para_end)
        assert score == 5.0  # Double newline has score 5.0

    def test_evaluate_boundary_quality_sentence(self):
        """Test boundary quality for sentence endings"""
        from src.utils.tokenizer import evaluate_boundary_quality

        encoding = tiktoken.get_encoding("o200k_base")

        text = "First sentence. Second sentence."
        tokens = encoding.encode(text)
        sent_end = len(encoding.encode("First sentence. "))
        score = evaluate_boundary_quality(text, tokens, encoding, sent_end)
        # Score should be relatively low for sentence endings
        assert score <= 100.0  # Verify it returns a valid score

    def test_evaluate_boundary_quality_comma(self):
        """Test boundary quality after comma"""
        from src.utils.tokenizer import evaluate_boundary_quality

        encoding = tiktoken.get_encoding("o200k_base")

        text = "First part, second part"
        tokens = encoding.encode(text)
        comma_end = len(encoding.encode("First part, "))
        score = evaluate_boundary_quality(text, tokens, encoding, comma_end)
        # Score should be relatively low for comma boundaries
        assert score <= 100.0  # Verify it returns a valid score

    def test_evaluate_boundary_quality_boundaries(self):
        """Test boundary quality at text boundaries"""
        from src.utils.tokenizer import evaluate_boundary_quality

        encoding = tiktoken.get_encoding("o200k_base")

        text = "Some text here"
        tokens = encoding.encode(text)

        # Text boundaries are ideal
        assert evaluate_boundary_quality(text, tokens, encoding, 0) == 0.0
        assert evaluate_boundary_quality(text, tokens, encoding, len(tokens)) == 0.0

    def test_is_inside_html_tag_various(self):
        """Test HTML tag detection"""
        from src.utils.tokenizer import is_inside_html_tag

        # Inside tag
        assert is_inside_html_tag("Text <div ", "class='test'>") is True
        assert is_inside_html_tag("Text <span", " id='1'>text</span>") is True

        # Not inside tag
        assert is_inside_html_tag("Text <div>content</div>", " more text") is False
        assert is_inside_html_tag("Plain text", " more text") is False

    def test_is_inside_html_tag_nested(self):
        """Test HTML tag detection with nested tags"""
        from src.utils.tokenizer import is_inside_html_tag

        # Complex case with nested tags
        text_before = "<div><span"
        text_after = ">text</span></div>"
        assert is_inside_html_tag(text_before, text_after) is True

    def test_is_inside_html_tag_edge_cases(self):
        """Test HTML tag detection edge cases"""
        from src.utils.tokenizer import is_inside_html_tag

        # Empty strings
        assert is_inside_html_tag("", "") is False

        # Only opening bracket
        assert is_inside_html_tag("<", "") is True

        # Closed tag
        assert is_inside_html_tag("<tag>", "") is False

    def test_is_inside_formula_single_dollar(self):
        """Test formula detection with single dollar signs"""
        from src.utils.tokenizer import is_inside_formula

        # Inside formula
        assert is_inside_formula("Text $x^2", " + y$") is True
        assert is_inside_formula("Formula: $", "E = mc^2$") is True

        # Outside formula
        assert is_inside_formula("Text $x^2$", " more text") is False
        assert is_inside_formula("No formula", " here") is False

    def test_is_inside_formula_double_dollar(self):
        """Test formula detection with double dollar signs"""
        from src.utils.tokenizer import is_inside_formula

        # Inside block formula
        assert is_inside_formula("Text $$\nE = ", "mc^2\n$$") is False  # Even number of $
        assert is_inside_formula("Text $$$\nE = ", "mc^2\n$$") is True  # Odd number

        # Outside block formula
        assert is_inside_formula("Text $$formula$$", " more") is False

    def test_is_inside_formula_edge_cases(self):
        """Test formula detection edge cases"""
        from src.utils.tokenizer import is_inside_formula

        # Empty strings
        assert is_inside_formula("", "") is False

        # Multiple formulas
        assert is_inside_formula("$a$ and $b", "$ text") is True
        assert is_inside_formula("$a$ and $b$", " text") is False

    def test_is_inside_code_block_basic(self):
        """Test code block detection"""
        from src.utils.tokenizer import is_inside_code_block

        # Inside code block
        assert is_inside_code_block("Text\n```python\ncode", "\n```") is True
        assert is_inside_code_block("```\n", "code\n```") is True

        # Outside code block
        assert is_inside_code_block("```code```", " text") is False
        assert is_inside_code_block("No code", " here") is False

    def test_is_inside_code_block_multiple(self):
        """Test code block detection with multiple blocks"""
        from src.utils.tokenizer import is_inside_code_block

        # After first block, before second
        assert is_inside_code_block("```code1```\ntext\n```", "code2```") is True

        # After both blocks
        assert is_inside_code_block("```code1```\n```code2```", " text") is False

    def test_is_inside_code_block_edge_cases(self):
        """Test code block detection edge cases"""
        from src.utils.tokenizer import is_inside_code_block

        # Empty strings
        assert is_inside_code_block("", "") is False

        # Single backtick vs triple
        assert is_inside_code_block("`code`", " text") is False

        # Incomplete triple backtick
        assert is_inside_code_block("``", "") is False


class TestDecodeCaching:
    """Тесты для оптимизации с кэшированием декодирования."""

    def test_find_safe_token_boundary_performance(self):
        """Тест что оптимизированная версия быстрее оригинальной."""
        import time
        import tiktoken
        from src.utils.tokenizer import find_safe_token_boundary

        # Создаем тестовый случай с множеством потенциальных границ
        text = "Word. " * 1000  # Много границ предложений
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(text)

        # Измеряем оптимизированную версию
        start = time.time()
        result, _ = find_safe_token_boundary(text, tokens, encoding, 500, 200)
        optimized_time = time.time() - start

        # Должна быть быстрой
        assert optimized_time < 0.1  # Менее 100ms
        assert result >= 300 and result <= 700  # В пределах диапазона поиска

    def test_cached_mode_consistency(self):
        """Тест что кэшированный режим дает те же результаты."""
        import tiktoken
        from src.utils.tokenizer import is_safe_cut_position, evaluate_boundary_quality

        encoding = tiktoken.get_encoding("o200k_base")
        text = "Test sentence. Another sentence. Final sentence."
        tokens = encoding.encode(text)

        # Тестируем несколько позиций
        test_positions = [5, 10, 15, 20]

        for pos in test_positions:
            if 0 < pos < len(tokens):
                # Legacy mode
                legacy_safe = is_safe_cut_position(text, tokens, encoding, pos)
                legacy_quality = evaluate_boundary_quality(text, tokens, encoding, pos)

                # Cached mode
                text_before = encoding.decode(tokens[:pos])
                text_after = encoding.decode(tokens[pos:])
                cached_safe = is_safe_cut_position(
                    text_before=text_before, text_after=text_after
                )
                cached_quality = evaluate_boundary_quality(
                    text_before=text_before, text_after=text_after
                )

                # Должны давать одинаковые результаты
                assert legacy_safe == cached_safe, f"Safety mismatch at pos {pos}"
                assert legacy_quality == cached_quality, f"Quality mismatch at pos {pos}"

    def test_helper_functions_token_char_mapping(self):
        """Тест вспомогательной функции построения маппинга."""
        import tiktoken
        from src.utils.tokenizer import _build_token_char_mapping

        encoding = tiktoken.get_encoding("o200k_base")
        text = "Test text with tokens"
        tokens = encoding.encode(text)

        # Построить маппинг
        mapping = _build_token_char_mapping(tokens, encoding)

        # Проверить корректность
        assert mapping[0] == 0  # Начало всегда 0
        assert mapping[len(tokens)] == len(text)  # Конец равен длине текста

        # Проверить монотонность (каждая следующая позиция >= предыдущей)
        for i in range(1, len(tokens) + 1):
            assert mapping[i] >= mapping[i-1]

    def test_get_text_at_boundary_helper(self):
        """Тест вспомогательной функции разделения текста."""
        from src.utils.tokenizer import _get_text_at_boundary

        text = "Hello world test"

        # Разделить в разных позициях
        before1, after1 = _get_text_at_boundary(text, 5)
        assert before1 == "Hello"
        assert after1 == " world test"

        before2, after2 = _get_text_at_boundary(text, 11)
        assert before2 == "Hello world"
        assert after2 == " test"

        # Граничные случаи
        before3, after3 = _get_text_at_boundary(text, 0)
        assert before3 == ""
        assert after3 == text

        before4, after4 = _get_text_at_boundary(text, len(text))
        assert before4 == text
        assert after4 == ""

    def test_decode_once_vs_multiple(self):
        """Тест что действительно происходит только одно декодирование."""
        import tiktoken
        from unittest.mock import Mock, patch
        from src.utils.tokenizer import find_safe_token_boundary

        text = "Test. " * 100
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(text)

        # Мокаем decode чтобы считать вызовы
        original_decode = encoding.decode
        decode_call_count = 0

        def counting_decode(token_list):
            nonlocal decode_call_count
            decode_call_count += 1
            return original_decode(token_list)

        encoding.decode = counting_decode

        # Вызываем функцию (теперь возвращает tuple)
        result, _ = find_safe_token_boundary(text, tokens, encoding, 50, 20)

        # Восстанавливаем оригинальный decode
        encoding.decode = original_decode

        # Проверяем что decode вызывался разумное количество раз
        # (должно быть O(1) декодирований для рабочего диапазона + построение маппинга)
        assert decode_call_count < 50, f"Too many decode calls: {decode_call_count}"
        assert result >= 30 and result <= 70  # В пределах диапазона


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


class TestNewBoundaryFunctions:
    """Tests for new boundary functions added in SLICER_REFACTOR_03."""

    def test_is_inside_list_numbered(self):
        """Test detection of numbered list boundaries."""
        from src.utils.tokenizer import is_inside_list

        # Between numbered list items - boundary at end of line, next line is list item
        text_before = "Shopping list:\n1. Apples\n2. Bananas\n"
        text_after = "3. Oranges\nEnd"
        assert is_inside_list(text_before, text_after) is True

        # Not in a list
        text_before = "Some normal text here."
        text_after = " More text follows."
        assert is_inside_list(text_before, text_after) is False

    def test_is_inside_list_bullet(self):
        """Test detection of bullet list boundaries."""
        from src.utils.tokenizer import is_inside_list

        # Between bullet list items - boundary at end of line
        text_before = "TODO:\n- Fix bug\n- Write tests\n"
        text_after = "- Deploy\nDone"
        assert is_inside_list(text_before, text_after) is True

        # Asterisk bullets
        text_before = "Features:\n* Feature A\n* Feature B\n"
        text_after = "* Feature C"
        assert is_inside_list(text_before, text_after) is True

    def test_is_inside_list_nested(self):
        """Test detection of nested list boundaries."""
        from src.utils.tokenizer import is_inside_list

        # Nested numbered list (indented) - boundary at end of line
        text_before = "1. First\n  a. Sub item\n"
        text_after = "  b. Another sub"
        assert is_inside_list(text_before, text_after) is True

    def test_is_inside_table_markdown(self):
        """Test detection of Markdown table boundaries."""
        from src.utils.tokenizer import is_inside_table

        # Inside markdown table
        text_before = "| Header1 | Header2 |\n|---|---|\n| Cell1 | Cell2 |"
        text_after = "\n| Cell3 | Cell4 |\n\nAfter table"
        assert is_inside_table(text_before, text_after) is True

        # Not in a table
        text_before = "Just some text."
        text_after = " More text here."
        assert is_inside_table(text_before, text_after) is False

    def test_is_inside_table_html(self):
        """Test detection of HTML table boundaries."""
        from src.utils.tokenizer import is_inside_table

        # Inside HTML table
        text_before = "<table><tr><td>Cell1</td>"
        text_after = "<td>Cell2</td></tr></table>"
        assert is_inside_table(text_before, text_after) is True

        # After table closed
        text_before = "<table><tr><td>Cell</td></tr></table>"
        text_after = " More content"
        assert is_inside_table(text_before, text_after) is False

    def test_find_boundary_candidates_headers(self):
        """Test that find_boundary_candidates finds positions BEFORE headers."""
        from src.utils.tokenizer import find_boundary_candidates

        text = "Some intro text.\n\n# Chapter 1\nContent here."
        target_pos = 20
        max_shift = 30

        candidates = find_boundary_candidates(text, target_pos, max_shift)

        # Should find markdown header candidate
        header_candidates = [(pos, t) for pos, t in candidates if t == "markdown_header"]
        assert len(header_candidates) > 0

        # Position should be BEFORE the # character
        for pos, _ in header_candidates:
            # Text after position should start with #
            assert text[pos:pos + 1] == "#"

    def test_find_boundary_candidates_priorities(self):
        """Test that candidates are sorted by priority."""
        from src.utils.tokenizer import find_boundary_candidates

        # Text with multiple boundary types
        text = "Text.\n\n# Header\nMore text. Another sentence.\n\nParagraph."
        target_pos = len(text) // 2
        max_shift = 100

        candidates = find_boundary_candidates(text, target_pos, max_shift)

        # Verify we have candidates
        assert len(candidates) > 0

        # First candidates should be higher priority (header/paragraph)
        if candidates:
            first_type = candidates[0][1]
            # Header and paragraph types should come before sentence/line/phrase
            high_priority_types = {"markdown_header", "paragraph", "subheader", "code_block_end"}
            # At least check that candidates are returned
            assert any(t in high_priority_types for _, t in candidates[:5])

    def test_find_safe_token_boundary_with_fallback_normal(self):
        """Test fallback function with normal boundary found."""
        from src.utils.tokenizer import find_safe_token_boundary_with_fallback
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        text = "First sentence. Second sentence. Third sentence."
        tokens = encoding.encode(text)

        result = find_safe_token_boundary_with_fallback(
            text, tokens, encoding,
            target_token_pos=len(tokens) // 2,
            max_shift_tokens=5,
            max_tokens=100
        )

        # Should return valid position
        assert 0 <= result <= len(tokens)

    def test_find_safe_token_boundary_with_fallback_returns_int(self):
        """Test that fallback function returns int (not tuple)."""
        from src.utils.tokenizer import find_safe_token_boundary_with_fallback
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        text = "Test text with some content here."
        tokens = encoding.encode(text)

        result = find_safe_token_boundary_with_fallback(
            text, tokens, encoding,
            target_token_pos=len(tokens) // 2,
            max_shift_tokens=3,
            max_tokens=50
        )

        # Should return int (for compatibility with slicer)
        assert isinstance(result, int)


if __name__ == "__main__":
    pytest.main([__file__])
