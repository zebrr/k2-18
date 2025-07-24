#!/usr/bin/env python3
"""
CLI-утилита для разделения образовательных текстов на слайсы.

Читает файлы из /data/raw/, применяет препроцессинг и нарезает их на фрагменты 
фиксированного размера с учетом семантических границ.

Использование:
    python slicer.py

Выходные файлы сохраняются в /data/staging/ в формате *.slice.json
"""

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
from src.utils.tokenizer import count_tokens, find_soft_boundary, find_safe_token_boundary

# Добавляем корень проекта в PYTHONPATH для корректных импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

# Внешние зависимости
from unidecode import unidecode
from bs4 import BeautifulSoup

# Импорт утилит проекта
from src.utils.config import load_config
from src.utils.tokenizer import count_tokens, find_soft_boundary
from src.utils.validation import validate_json
from src.utils.exit_codes import (
    EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR, EXIT_API_LIMIT_ERROR, EXIT_IO_ERROR,
    log_exit
)

# Установка UTF-8 кодировки для Windows консоли
from src.utils.console_encoding import setup_console_encoding
setup_console_encoding()

def setup_logging(log_level: str = "info") -> None:
    """
    Настройка логирования для slicer.
    
    Args:
        log_level: Уровень логирования (debug, info, warning, error)
    """
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO, 
        "warning": logging.WARNING,
        "error": logging.ERROR
    }
    
    level = level_map.get(log_level.lower(), logging.INFO)
    
    # Настройка формата логов
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Консольный handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Настройка root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)


def validate_config_parameters(config: Dict[str, Any]) -> None:
    """
    Валидация параметров конфигурации slicer.
    
    Args:
        config: Словарь конфигурации
        
    Raises:
        ValueError: При некорректных параметрах
    """
    slicer_config = config.get('slicer', {})
    
    # Проверка обязательных параметров
    required_params = ['max_tokens', 'overlap', 'soft_boundary_max_shift', 'allowed_extensions']
    for param in required_params:
        if param not in slicer_config:
            raise ValueError(f"Отсутствует обязательный параметр slicer.{param}")
    
    # Проверка типов и диапазонов
    max_tokens = slicer_config['max_tokens']
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"slicer.max_tokens должен быть положительным целым числом, получен: {max_tokens}")
    
    overlap = slicer_config['overlap']
    if not isinstance(overlap, int) or overlap < 0:
        raise ValueError(f"slicer.overlap должен быть неотрицательным целым числом, получен: {overlap}")
    
    if overlap >= max_tokens:
        raise ValueError(f"slicer.overlap ({overlap}) должен быть меньше max_tokens ({max_tokens})")
    
    soft_boundary_max_shift = slicer_config['soft_boundary_max_shift']
    if not isinstance(soft_boundary_max_shift, int) or soft_boundary_max_shift < 0:
        raise ValueError(f"slicer.soft_boundary_max_shift должен быть неотрицательным целым числом")
    
    # Специальная валидация для overlap > 0
    if overlap > 0:
        max_allowed_shift = int(overlap * 0.8)
        if soft_boundary_max_shift > max_allowed_shift:
            raise ValueError(
                f"При overlap > 0, soft_boundary_max_shift ({soft_boundary_max_shift}) "
                f"не должен превышать overlap*0.8 ({max_allowed_shift})"
            )
    
    allowed_extensions = slicer_config['allowed_extensions']
    if not isinstance(allowed_extensions, list) or not allowed_extensions:
        raise ValueError("slicer.allowed_extensions должен быть непустым списком")


def create_slug(filename: str) -> str:
    """
    Создает slug из имени файла согласно ТЗ.
    
    Правила:
    - Удалить расширение
    - Транслитерировать кириллицу в латиницу  
    - Привести к нижнему регистру
    - Заменить пробелы на _
    - Остальные символы без изменений
    
    Args:
        filename: Имя файла с расширением
        
    Returns:
        Обработанный slug
        
    Examples:
        >>> create_slug("Алгоритмы и Структуры.txt")
        'algoritmy_i_struktury'
        >>> create_slug("My Course Chapter 1.md")
        'my_course_chapter_1'
        >>> create_slug("python-basics.html")
        'python-basics'
    """
    # Удаляем расширение
    name_without_ext = Path(filename).stem
    
    # Транслитерируем кириллицу в латиницу
    transliterated = unidecode(name_without_ext)
    
    # Приводим к нижнему регистру
    lowercased = transliterated.lower()
    
    # Заменяем пробелы на подчеркивания
    slug = lowercased.replace(' ', '_')
    
    return slug


def preprocess_text(text: str) -> str:
    """
    Применяет препроцессинг к тексту согласно ТЗ.
    
    Включает:
    - Удаление содержимого <script> и <style> тегов
    - Unicode нормализацию NFC
    - Остальное содержимое остается без изменений
    
    Args:
        text: Исходный текст
        
    Returns:
        Обработанный текст
    """
    if not isinstance(text, str):
        raise ValueError("Входной параметр должен быть строкой")
    
    # Сначала применяем Unicode нормализацию
    normalized_text = unicodedata.normalize("NFC", text)
    
    # Проверяем наличие HTML тегов script или style
    if '<script' in normalized_text.lower() or '<style' in normalized_text.lower():
        # Используем BeautifulSoup для безопасного удаления тегов
        soup = BeautifulSoup(normalized_text, 'html.parser')
        
        # Удаляем все script теги и их содержимое
        for script in soup.find_all('script'):
            script.decompose()
        
        # Удаляем все style теги и их содержимое
        for style in soup.find_all('style'):
            style.decompose()
        
        # Получаем обработанный текст
        processed_text = str(soup)
    else:
        # Если нет script/style тегов, оставляем как есть
        processed_text = normalized_text
    
    return processed_text


class InputError(Exception):
    """Исключение для ошибок входных данных."""
    pass


def load_and_validate_file(file_path: Path, allowed_extensions: List[str]) -> str:
    """
    Загружает и валидирует файл.
    
    Args:
        file_path: Путь к файлу
        allowed_extensions: Список разрешенных расширений
        
    Returns:
        Содержимое файла
        
    Raises:
        InputError: При пустом файле или неподдерживаемом расширении
    """
    # Проверка расширения
    if file_path.suffix.lstrip('.').lower() not in [ext.lower() for ext in allowed_extensions]:
        raise InputError(f"Неподдерживаемое расширение файла: {file_path.suffix}")
    
    try:
        # Загрузка с автоопределением кодировки
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fallback на другие кодировки
        try:
            with open(file_path, 'r', encoding='cp1251') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin1') as f:
                content = f.read()
    
    # Применение препроцессинга
    content = preprocess_text(content)
    
    # Проверка на пустоту после препроцессинга
    if not content.strip():
        raise InputError(f"Empty file detected: {file_path.name}. Please remove empty files from /data/raw/")
    
    return content


def slice_text_with_window(text: str, max_tokens: int, overlap: int, 
                          soft_boundary: bool, soft_boundary_max_shift: int) -> List[Tuple[str, int, int]]:
    """
    Нарезает текст на слайсы с помощью скользящего окна.
    
    Args:
        text: Исходный текст для нарезки
        max_tokens: Максимальный размер слайса в токенах  
        overlap: Количество токенов перекрытия
        soft_boundary: Использовать ли мягкие границы
        soft_boundary_max_shift: Максимальное смещение для мягких границ
        
    Returns:
        Список кортежей (slice_text, slice_token_start, slice_token_end)
    """
    if not text or not text.strip():
        return []
    
    # Подсчитаем общее количество токенов в тексте
    total_tokens = count_tokens(text)
    
    # Если весь текст помещается в один слайс
    if total_tokens <= max_tokens:
        return [(text, 0, total_tokens)]
    
    # Токенизируем весь текст для работы с позициями
    import tiktoken
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(text)
    
    slices = []
    current_token_start = 0
    
    while current_token_start < len(tokens):
        # Определяем конец текущего окна
        window_end = min(current_token_start + max_tokens, len(tokens))
        
        # Если это последний фрагмент, берем до конца
        if window_end == len(tokens):
            slice_tokens = tokens[current_token_start:]
            slice_text = encoding.decode(slice_tokens)
            slice_token_start = current_token_start
            slice_token_end = len(tokens)
            
            slices.append((slice_text, slice_token_start, slice_token_end))
            break
        
        # Ищем soft boundary если включено
        actual_end = window_end
        if soft_boundary and soft_boundary_max_shift > 0:
            # Конвертируем soft_boundary_max_shift из символов в примерное количество токенов
            # Примерное соотношение: 1 токен ≈ 4 символа (для o200k_base)
            max_shift_tokens = max(1, soft_boundary_max_shift // 4)
            
            # Используем новую функцию для поиска безопасной границы
            from src.utils.tokenizer import find_safe_token_boundary
            
            safe_end = find_safe_token_boundary(
                text=text,
                tokens=tokens,
                encoding=encoding,
                target_token_pos=window_end,
                max_shift_tokens=max_shift_tokens
            )
            
            # Логирование для отладки
            if safe_end != window_end:
                shift = safe_end - window_end
                logging.info(f"Soft boundary найдена: сдвиг {shift:+d} токенов")
                
                # Показываем тип границы для отладки
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    context = encoding.decode(tokens[max(0, safe_end-20):safe_end])
                    if context.endswith('\n\n'):
                        logging.debug(f"Тип границы: двойной перенос строки")
                    elif re.search(r'[.!?]\s*$', context):
                        logging.debug(f"Тип границы: конец предложения")
                    elif re.search(r'</h[1-6]>\s*$', context):
                        logging.debug(f"Тип границы: HTML заголовок")
            
            actual_end = safe_end
        
        # Создаем итоговый слайс
        final_slice_tokens = tokens[current_token_start:actual_end]
        slice_text = encoding.decode(final_slice_tokens)
        slice_token_start = current_token_start
        slice_token_end = actual_end
        
        slices.append((slice_text, slice_token_start, slice_token_end))
        
        # Вычисляем начало следующего окна
        if overlap == 0:
            current_token_start = actual_end  # Без перекрытий
        else:
            current_token_start = actual_end - overlap  # С перекрытием
            # Защита от бесконечного цикла
            if current_token_start <= slice_token_start:
                current_token_start = slice_token_start + 1
    
    # Обработка граничных случаев для overlap > 0 согласно ТЗ
    if overlap > 0 and len(slices) >= 2:
        last_slice = slices[-1]
        prev_slice = slices[-2]
        
        # Если последний фрагмент меньше overlap, объединяем с предыдущим
        last_slice_size = last_slice[2] - last_slice[1]  # token_end - token_start
        if last_slice_size < overlap:
            # Обновляем предыдущий слайс
            prev_start = prev_slice[1]
            combined_end = last_slice[2]
            combined_tokens = tokens[prev_start:combined_end]
            combined_text = encoding.decode(combined_tokens)
            
            # Заменяем предыдущий слайс обновленным
            slices[-2] = (combined_text, prev_start, combined_end)
            # Удаляем последний слайс
            slices.pop()
    
    return slices


def save_slice(slice_data: Dict[str, Any], output_dir: Path) -> None:
    """
    Сохраняет слайс в JSON файл.
    
    Args:
        slice_data: Данные слайса
        output_dir: Директория для сохранения
        
    Raises:
        IOError: При ошибках записи файла
    """
    slice_id = slice_data['id']
    output_file = output_dir / f"{slice_id}.slice.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(slice_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Сохранен слайс: {output_file}")
    except Exception as e:
        logging.error(f"Ошибка сохранения слайса {slice_id}: {e}")
        raise IOError(f"Не удалось сохранить слайс {slice_id}: {e}")


def process_file(file_path: Path, config: Dict[str, Any], global_slice_counter: int) -> Tuple[List[Dict[str, Any]], int]:
    """
    Обрабатывает один файл и возвращает список слайсов.
    
    Args:
        file_path: Путь к файлу
        config: Конфигурация slicer
        global_slice_counter: Глобальный счетчик слайсов
        
    Returns:
        Кортеж (список слайсов, обновленный счетчик)
        
    Raises:
        InputError: При ошибках входных данных
        RuntimeError: При ошибках обработки
    """
    slicer_config = config['slicer']
    
    logging.info(f"Обработка файла: {file_path.name}")
    
    try:
        # Загрузка и валидация файла
        content = load_and_validate_file(file_path, slicer_config['allowed_extensions'])
        
        # Создание slug
        slug = create_slug(file_path.name)
        
        # Нарезка на слайсы
        slices_data = slice_text_with_window(
            content,
            slicer_config['max_tokens'],
            slicer_config['overlap'],
            slicer_config['soft_boundary'],
            slicer_config['soft_boundary_max_shift']
        )
        
        # Создание объектов слайсов
        slices = []
        for i, (slice_text, slice_token_start, slice_token_end) in enumerate(slices_data):
            slice_obj = {
                "id": f"slice_{global_slice_counter:03d}",
                "order": global_slice_counter,
                "source_file": file_path.name,
                "slug": slug,
                "text": slice_text,
                "slice_token_start": slice_token_start,
                "slice_token_end": slice_token_end
            }
            slices.append(slice_obj)
            global_slice_counter += 1
        
        logging.info(f"Файл {file_path.name}: создано {len(slices)} слайсов")
        return slices, global_slice_counter
        
    except InputError:
        # Переподнимаем InputError как есть
        raise
    except Exception as e:
        logging.error(f"Ошибка обработки файла {file_path.name}: {e}")
        raise RuntimeError(f"Не удалось обработать файл {file_path.name}: {e}")


def main(argv=None):
    """Основная функция slicer."""
    
    # Настройка CLI
    parser = argparse.ArgumentParser(
        description="Утилита для разделения образовательных текстов на слайсы",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Пока параметров нет - конфиг прошит
    args = parser.parse_args(argv)
    
    try:
        # Загрузка конфигурации
        config = load_config()
        
        # Настройка логирования
        log_level = config.get('slicer', {}).get('log_level', 'info')
        setup_logging(log_level)
        
        logging.info("Запуск slicer.py")
        
        # Валидация параметров конфигурации
        try:
            validate_config_parameters(config)
        except ValueError as e:
            logging.error(f"Ошибка конфигурации: {e}")
            return EXIT_CONFIG_ERROR
        
        # Определение путей
        raw_dir = Path("data/raw")
        staging_dir = Path("data/staging")
        
        # Проверка существования директорий
        if not raw_dir.exists():
            logging.error(f"Директория {raw_dir} не существует")
            return EXIT_INPUT_ERROR
        
        try:
            staging_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Не удалось создать директорию {staging_dir}: {e}")
            return EXIT_IO_ERROR
        
        # Получение списка файлов в лексикографическом порядке
        allowed_extensions = config['slicer']['allowed_extensions']
        input_files = []
        
        for ext in allowed_extensions:
            pattern = f"*.{ext.lower()}"
            input_files.extend(raw_dir.glob(pattern))
            # Также ищем с заглавными буквами
            pattern_upper = f"*.{ext.upper()}"
            input_files.extend(raw_dir.glob(pattern_upper))
        
        # Удаление дубликатов и сортировка
        input_files = sorted(set(input_files))
        
        if not input_files:
            logging.warning(f"Не найдено файлов для обработки в {raw_dir}")
            logging.warning(f"Поддерживаемые расширения: {allowed_extensions}")
            return EXIT_SUCCESS
        
        # Вывод файлов, которые будут пропущены
        all_files = list(raw_dir.iterdir())
        for file_path in all_files:
            if file_path.is_file() and file_path not in input_files:
                logging.warning(f"Unsupported file skipped: {file_path.name}")
        
        logging.info(f"Найдено {len(input_files)} файлов для обработки")
        
        # Обработка файлов
        global_slice_counter = 1
        total_slices = 0
        
        for file_path in input_files:
            try:
                slices, global_slice_counter = process_file(file_path, config, global_slice_counter)
                
                # Сохранение слайсов
                for slice_data in slices:
                    save_slice(slice_data, staging_dir)
                
                total_slices += len(slices)
                
            except InputError as e:
                logging.error(f"Ошибка входных данных в файле {file_path.name}: {e}")
                return EXIT_INPUT_ERROR
            except IOError as e:
                logging.error(f"Ошибка ввода/вывода при обработке {file_path.name}: {e}")
                return EXIT_IO_ERROR
            except RuntimeError as e:
                logging.error(f"Ошибка выполнения при обработке {file_path.name}: {e}")
                return EXIT_RUNTIME_ERROR
            except Exception as e:
                logging.error(f"Неожиданная ошибка при обработке {file_path.name}: {e}")
                return EXIT_RUNTIME_ERROR
        
        logging.info(f"Обработка завершена: {total_slices} слайсов сохранено в {staging_dir}")
        log_exit(logging.getLogger(), EXIT_SUCCESS)
        return EXIT_SUCCESS
        
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
        log_exit(logging.getLogger(), EXIT_RUNTIME_ERROR)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)