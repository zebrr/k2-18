"""
Модуль для работы с токенизацией текста и поиском семантических границ.

Используется кодек o200k_base, который поддерживается всеми современными моделями OpenAI
"""

import tiktoken
import re
import logging
from typing import Optional, List


def count_tokens(text: str) -> int:
    """
    Подсчитывает количество токенов в тексте используя кодек o200k_base.
    
    Args:
        text: Входной текст для токенизации
        
    Returns:
        Количество токенов в тексте
        
    Raises:
        ValueError: Если text не является строкой
    """
    if not isinstance(text, str):
        raise ValueError("Входной параметр должен быть строкой")
    
    # Инициализация токенизатора o200k_base
    encoding = tiktoken.get_encoding("o200k_base")
    
    # Подсчет токенов
    tokens = encoding.encode(text)
    return len(tokens)


def find_soft_boundary(text: str, target_pos: int, max_shift: int) -> Optional[int]:
    """
    Ищет ближайшую семантическую границу в тексте для мягкого разрыва.
    Использует иерархию приоритетов для образовательного контента.
    
    Приоритеты границ (от высшего к низшему):
    1. Границы разделов (заголовки)
    2. Границы абзацев (двойные переносы, блоки кода)
    3. Границы предложений (точки, восклицательные и вопросительные знаки)
    4. Границы фраз (запятые, двоеточия, точки с запятой)
    5. Границы слов (пробелы) - fallback
    
    Args:
        text: Исходный текст
        target_pos: Целевая позиция в символах
        max_shift: Максимальное смещение от target_pos
        
    Returns:
        Позиция границы в символах или None если граница не найдена
    """
    if not isinstance(text, str) or len(text) == 0:
        return None
        
    if target_pos < 0 or target_pos > len(text):
        return None
        
    if max_shift < 0:
        return None
    
    # Определяем диапазон поиска
    start_pos = max(0, target_pos - max_shift)
    end_pos = min(len(text), target_pos + max_shift)
    
    # Иерархия границ с весами (чем меньше вес, тем лучше граница)
    boundary_types = {
        'section': {'weight': 1, 'candidates': []},
        'paragraph': {'weight': 2, 'candidates': []},
        'sentence': {'weight': 3, 'candidates': []},
        'phrase': {'weight': 4, 'candidates': []},
        'word': {'weight': 5, 'candidates': []}
    }
    
    # 1. Границы разделов (приоритет 1)
    # HTML заголовки
    for match in re.finditer(r'</h[1-6]>\s*(?=\n|$)', text, re.IGNORECASE):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['section']['candidates'].append(pos)
    
    # Markdown заголовки
    for match in re.finditer(r'(?:^|\n)(#{1,6})\s+.*?(?=\n|$)', text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['section']['candidates'].append(pos)
    
    # Текстовые заголовки
    section_pattern = r'(?:^|\n)(?:Глава|Параграф|Часть|Chapter|Section|Раздел|Урок|Тема)\s+.*?(?=\n|$)'
    for match in re.finditer(section_pattern, text, re.IGNORECASE | re.MULTILINE):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['section']['candidates'].append(pos)
    
    # 2. Границы абзацев (приоритет 2)
    # Двойной перенос строки
    for match in re.finditer(r'\n\n+', text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['paragraph']['candidates'].append(pos)
    
    # Конец блока кода
    for match in re.finditer(r'(?:^|\n)```\s*(?=\n|$)', text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['paragraph']['candidates'].append(pos)
    
    # Конец блока формулы
    for match in re.finditer(r'(?:^|\n)\$\$\s*(?=\n|$)', text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['paragraph']['candidates'].append(pos)
    
    # HTML/Markdown ссылки
    for match in re.finditer(r'</a>|\]\([^)]+\)', text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['paragraph']['candidates'].append(pos)
    
    # 3. Границы предложений (приоритет 3)
    # Конец предложения
    for match in re.finditer(r'[.!?]\s+', text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            # Проверяем, что это не сокращение
            before_pos = match.start()
            if before_pos > 0:
                # Простая эвристика для сокращений
                word_before = text[max(0, before_pos-10):before_pos].strip()
                if not word_before.endswith(('Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'St', 'vs', 'etc', 'т.д', 'т.п', 'и.д', 'и.п')):
                    boundary_types['sentence']['candidates'].append(pos)
    
    # Точка с запятой
    for match in re.finditer(r';\s+', text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['sentence']['candidates'].append(pos)
    
    # 4. Границы фраз (приоритет 4)
    # Запятая
    for match in re.finditer(r',\s+', text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['phrase']['candidates'].append(pos)
    
    # Двоеточие
    for match in re.finditer(r':\s+', text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['phrase']['candidates'].append(pos)
    
    # Тире
    for match in re.finditer(r'\s+[—–-]\s+', text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['phrase']['candidates'].append(pos)
    
    # 5. Границы слов (приоритет 5 - fallback)
    for match in re.finditer(r'\s+', text):
        pos = match.end()
        if start_pos <= pos <= end_pos:
            boundary_types['word']['candidates'].append(pos)
    
    # Выбираем лучшую границу с учетом приоритетов
    best_boundary = None
    best_score = float('inf')
    
    for boundary_type, data in boundary_types.items():
        weight = data['weight']
        candidates = data['candidates']
        
        # Убираем дубликаты
        candidates = list(set(candidates))
        
        for pos in candidates:
            # Считаем score = weight * distance
            # Чем меньше score, тем лучше
            distance = abs(pos - target_pos)
            score = weight * distance
            
            # Небольшой бонус для границ после target_pos (предпочитаем не обрезать)
            if pos > target_pos:
                score *= 0.9
            
            if score < best_score:
                best_score = score
                best_boundary = pos
    
    return best_boundary

def find_safe_token_boundary(text: str, tokens: List[int], encoding, 
                            target_token_pos: int, max_shift_tokens: int) -> int:
    """
    Находит безопасную границу для разреза на уровне токенов.
    
    Args:
        text: Исходный текст
        tokens: Список токенов
        encoding: Объект tiktoken encoding
        target_token_pos: Целевая позиция в токенах
        max_shift_tokens: Максимальное смещение в токенах
        
    Returns:
        Безопасная позиция в токенах для разреза
    """
    # Определяем диапазон поиска
    start_pos = max(0, target_token_pos - max_shift_tokens)
    end_pos = min(len(tokens), target_token_pos + max_shift_tokens)
    
    best_pos = target_token_pos
    best_score = float('inf')
    
    # Проверяем каждую возможную позицию в диапазоне
    for pos in range(start_pos, end_pos + 1):
        # Проверяем, что не режем внутри структуры
        if is_safe_cut_position(text, tokens, encoding, pos):
            # Оцениваем качество этой позиции
            score = evaluate_boundary_quality(text, tokens, encoding, pos)
            
            # Добавляем штраф за расстояние от target
            distance_penalty = abs(pos - target_token_pos)
            total_score = score + distance_penalty * 0.1
            
            if total_score < best_score:
                best_score = total_score
                best_pos = pos
    
    return best_pos


def is_safe_cut_position(text: str, tokens: List[int], encoding, pos: int) -> bool:
    """
    Проверяет, безопасно ли резать в данной позиции токенов.
    """
    if pos <= 0 or pos >= len(tokens):
        return pos == 0 or pos == len(tokens)
    
    # Декодируем текст до и после позиции
    text_before = encoding.decode(tokens[:pos])
    text_after = encoding.decode(tokens[pos:])
    
    # Проверки на целостность структур
    checks = [
        # Не режем внутри слова
        not (text_before and text_after and 
             text_before[-1].isalnum() and text_after[0].isalnum()),
        
        # Не режем внутри URL
        not is_inside_url(text_before, text_after),
        
        # Не режем внутри markdown ссылки
        not is_inside_markdown_link(text_before, text_after),
        
        # Не режем внутри HTML тега
        not is_inside_html_tag(text_before, text_after),
        
        # Не режем внутри формулы
        not is_inside_formula(text_before, text_after),
        
        # Не режем внутри блока кода
        not is_inside_code_block(text_before, text_after)
    ]
    
    return all(checks)


def is_inside_url(text_before: str, text_after: str) -> bool:
    """Проверяет, находимся ли мы внутри URL."""
    # Ищем начало URL в text_before
    url_pattern = r'https?://[^\s\)>\]]*$'
    if re.search(url_pattern, text_before):
        # Проверяем, продолжается ли URL в text_after
        if text_after and re.match(r'^[^\s\)>\]]+', text_after):
            return True
    return False


def is_inside_markdown_link(text_before: str, text_after: str) -> bool:
    """Проверяет, находимся ли мы внутри markdown ссылки."""
    # Проверяем [текст](url) структуру
    # Считаем незакрытые квадратные и круглые скобки
    open_square = text_before.count('[') - text_before.count(']')
    open_round = text_before.count('(') - text_before.count(')')
    
    # Если есть незакрытая [ и следует ](
    if open_square > 0 and '](h' in text_before[-20:] + text_after[:5]:
        return True
    
    # Если внутри (url) части
    if open_round > 0 and text_before.endswith(']('):
        return True
        
    return False


def is_inside_html_tag(text_before: str, text_after: str) -> bool:
    """Проверяет, находимся ли мы внутри HTML тега."""
    # Проверяем, есть ли незакрытый 
    last_open = text_before.rfind('<')
    last_close = text_before.rfind('>')
    
    # Если последний < идет после последнего >, мы внутри тега
    return last_open > last_close


def is_inside_formula(text_before: str, text_after: str) -> bool:
    """Проверяет, находимся ли мы внутри математической формулы."""
    # Проверяем $...$ и $$...$$
    # Считаем количество $ до позиции
    dollar_count = text_before.count('$')
    
    # Если нечетное количество $, мы внутри формулы
    return dollar_count % 2 == 1


def is_inside_code_block(text_before: str, text_after: str) -> bool:
    """Проверяет, находимся ли мы внутри блока кода."""
    # Считаем тройные кавычки
    triple_quotes = text_before.count('```')
    
    # Если нечетное количество, мы внутри блока кода
    return triple_quotes % 2 == 1


def evaluate_boundary_quality(text: str, tokens: List[int], encoding, pos: int) -> float:
    """
    Оценивает качество границы (чем меньше, тем лучше).
    """
    if pos <= 0 or pos >= len(tokens):
        return 0.0  # Границы текста - идеальные
    
    # Декодируем контекст вокруг границы
    context_before = encoding.decode(tokens[max(0, pos-10):pos])
    context_after = encoding.decode(tokens[pos:min(len(tokens), pos+10)])
    
    score = 100.0  # Базовый score
    
    # Проверяем различные типы границ и назначаем scores
    # (чем меньше score, тем лучше граница)
    
    # Заголовки - лучшие границы
    if re.search(r'</h[1-6]>\s*$', context_before, re.IGNORECASE):
        score = 1.0
    elif re.search(r'\n#{1,6}\s+.*$', context_before):
        score = 1.0
    elif re.search(r'\n(?:Глава|Chapter|Раздел)\s+.*$', context_before, re.IGNORECASE):
        score = 1.0
    
    # Двойной перенос строки - хорошая граница
    elif context_before.endswith('\n\n'):
        score = 5.0
    
    # Конец предложения
    elif re.search(r'[.!?]\s*$', context_before):
        score = 10.0
    
    # Конец абзаца (одинарный перенос)
    elif context_before.endswith('\n'):
        score = 15.0
    
    # После запятой или точки с запятой
    elif re.search(r'[,;]\s*$', context_before):
        score = 20.0
    
    # Между словами (пробел)
    elif context_before.endswith(' '):
        score = 50.0
    
    return score
    