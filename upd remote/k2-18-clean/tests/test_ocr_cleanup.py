from src.ocr_cleanup import clean_text, profile_text


def test_profile_counts_core_ocr_artifacts():
    text = "сло\u00ad\nво\n52\n\n\fновая страница\n"

    profile = profile_text(text, max_examples=1)

    assert profile["issues"]["soft_hyphen"]["count"] == 1
    assert profile["issues"]["form_feed"]["count"] == 1
    assert profile["issues"]["page_number_before_break"]["count"] == 1
    assert profile["issues"]["soft_hyphen_linebreak"]["count"] == 1


def test_clean_text_removes_page_number_and_soft_hyphen_break():
    text = "1749\n\n\fПРЕДИСЛОВИЕ\nсло\u00ad\nво\n52\n\n\fследующая\n"

    cleaned, rules = clean_text(text)
    replacements = {rule.rule: rule.replacements for rule in rules}

    assert "1749" in cleaned
    assert "52" not in cleaned
    assert "слово" in cleaned
    assert "\u00ad" not in cleaned
    assert "\f" not in cleaned
    assert replacements["remove_1_3_digit_page_number_before_form_feed"] == 1


def test_clean_text_join_lines_is_opt_in():
    text = "Первая строка\nвторой кусок.\n\nНовый абзац."

    cleaned_without_join, _ = clean_text(text)
    cleaned_with_join, _ = clean_text(text, join_lines=True)

    assert "Первая строка\nвторой кусок." in cleaned_without_join
    assert "Первая строка второй кусок." in cleaned_with_join


def test_join_lines_preserves_short_structural_headings():
    text = "ПРЕДИСЛОВИЕ\nПервая строка\nвторой кусок.\n\nI\nНовый абзац\nпродолжение."

    cleaned, _ = clean_text(text, join_lines=True)

    assert "ПРЕДИСЛОВИЕ\nПервая строка второй кусок." in cleaned
    assert "\nI\nНовый абзац продолжение." in cleaned
