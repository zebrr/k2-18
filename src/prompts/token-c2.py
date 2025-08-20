#!/usr/bin/env python3
"""
Утилита для подсчета токенов в файлах.
Использование: 
  python token-c2.py             - подсчет токенов во всех файлах текущей папки
  python token-c2.py <файл>      - детальная информация о конкретном файле
"""

import sys
import os
from pathlib import Path

# Имя файла для исключения из подсчета (сам скрипт)
EXCLUDE_FILENAME = "token-c2.py"

try:
    import tiktoken
except ImportError:
    print("Ошибка: нужно установить tiktoken")
    print("Выполните: pip install tiktoken")
    sys.exit(1)


def count_tokens(text: str) -> int:
    """Подсчет токенов с использованием o200k_base токенизатора."""
    encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))


def format_size(size_bytes: int) -> str:
    """Форматирование размера файла в читаемом виде."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def process_single_file(file_path: Path):
    """Обработка одного файла с детальной информацией."""
    try:
        # Получение размера файла
        file_size = file_path.stat().st_size
        
        # Чтение файла как текст
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Подсчет символов и токенов
        char_count = len(text)
        token_count = count_tokens(text)
        
        # Вывод статистики
        print(f" ")        
        print(f"Файл        : {file_path}")
        print(f"Размер      : {format_size(file_size)}")
        print(f"Символов    : {char_count:,}")
        print(f"Токенов     : {token_count:,}")
        print(f"Соотношение : {char_count / token_count:.2f} символов/токен")
        
    except UnicodeDecodeError:
        print(f"Ошибка: не удалось прочитать файл '{file_path}' как UTF-8 текст")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")
        sys.exit(1)


def process_directory():
    """Обработка всех файлов в текущей директории."""
    current_dir = Path.cwd()
    files_data = []
    total_tokens = 0
    
    # Собираем информацию о файлах
    for item in current_dir.iterdir():
        if item.is_file():
            # Пропускаем файл-исключение
            if item.name == EXCLUDE_FILENAME:
                continue
                
            try:
                with open(item, 'r', encoding='utf-8') as f:
                    text = f.read()
                token_count = count_tokens(text)
                files_data.append((item.name, token_count))
                total_tokens += token_count
            except (UnicodeDecodeError, PermissionError):
                # Пропускаем файлы, которые не можем прочитать
                continue
            except Exception:
                # Пропускаем файлы с другими ошибками
                continue
    
    if not files_data:
        print("Не найдено текстовых файлов в текущей директории")
        return
    
    # Находим максимальную длину имени файла для выравнивания
    max_name_length = max(len(name) for name, _ in files_data)
    
    # Сортируем файлы по имени
    files_data.sort(key=lambda x: x[0])
    
    # Выводим результаты
    print()
    for filename, tokens in files_data:
        # Выравниваем по самому длинному имени файла
        print(f"{filename:<{max_name_length}} -- {tokens:,} токенов")
    
    # Выводим общий итог
    print("-" * (max_name_length + 20))  # разделительная линия
    print(f"{'Всего':<{max_name_length}} -- {total_tokens:,} токенов")
    print()


def main():
    if len(sys.argv) == 1:
        # Режим без аргументов - обработка директории
        process_directory()
    elif len(sys.argv) == 2:
        # Режим с аргументом - обработка одного файла
        file_path = Path(sys.argv[1])
        
        # Проверка существования файла
        if not file_path.exists():
            print(f"Ошибка: файл '{file_path}' не найден")
            sys.exit(1)
        
        if not file_path.is_file():
            print(f"Ошибка: '{file_path}' не является файлом")
            sys.exit(1)
        
        process_single_file(file_path)
    else:
        print("Использование:")
        print("  python token-c2.py             - подсчет токенов во всех файлах текущей папки")
        print("  python token-c2.py <файл>      - детальная информация о конкретном файле")
        sys.exit(1)


if __name__ == "__main__":
    main()