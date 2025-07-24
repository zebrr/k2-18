"""
Утилита для безопасной настройки UTF-8 кодировки в Windows консоли.
"""

import sys
import os
import io


def setup_console_encoding():
    """
    Настраивает UTF-8 кодировку для Windows консоли безопасным способом.
    Не ломает pytest и другие инструменты, которые перехватывают stdout.
    """
    # Проверяем, работаем ли мы в Windows
    if sys.platform != 'win32':
        return
    
    # Проверяем, не запущены ли мы под pytest
    if 'pytest' in sys.modules:
        # Под pytest не трогаем stdout/stderr
        return
    
    # Проверяем, не переопределены ли уже потоки
    if (hasattr(sys.stdout, '_original_stream') or 
        hasattr(sys.stderr, '_original_stream')):
        # Уже настроено, не трогаем
        return
    
    # Проверяем, являются ли потоки TTY (терминалом)
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        # Не терминал (возможно, перенаправление в файл) - не трогаем
        return
    
    try:
        # Пытаемся установить кодировку через переменную окружения
        # Это более безопасный способ для Python 3.7+
        if sys.version_info >= (3, 7):
            # Устанавливаем переменную окружения для будущих процессов
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            # Для текущего процесса используем reconfigure (Python 3.7+)
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
        else:
            # Для старых версий Python используем обертку, но сохраняем оригинал
            # и добавляем флаг, чтобы не переопределять повторно
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            sys.stdout = io.TextIOWrapper(
                original_stdout.buffer, 
                encoding='utf-8',
                line_buffering=original_stdout.line_buffering
            )
            sys.stderr = io.TextIOWrapper(
                original_stderr.buffer,
                encoding='utf-8', 
                line_buffering=original_stderr.line_buffering
            )
            
            # Помечаем, что потоки были изменены
            sys.stdout._original_stream = original_stdout
            sys.stderr._original_stream = original_stderr
            
    except Exception:
        # Если что-то пошло не так, не падаем, а просто работаем как есть
        pass
