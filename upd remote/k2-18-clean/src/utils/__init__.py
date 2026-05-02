"""
Модуль утилит для проекта iText2KG.

Предоставляет вспомогательные функции для работы с:
- Конфигурацией
- Токенизацией
- Валидацией
- LLM клиентом
- Кодами ошибок
- Настройкой кодировки консоли
"""

# Импортируем основные компоненты для удобства использования
from .config import ConfigValidationError, load_config
from .console_encoding import setup_console_encoding
from .exit_codes import (EXIT_API_LIMIT_ERROR, EXIT_CONFIG_ERROR,
                         EXIT_INPUT_ERROR, EXIT_IO_ERROR, EXIT_RUNTIME_ERROR,
                         EXIT_SUCCESS, get_exit_code_description,
                         get_exit_code_name, log_exit)
from .llm_client import OpenAIClient, TPMBucket
from .llm_embeddings import (EmbeddingsClient, cosine_similarity_batch,
                             get_embeddings)
from .tokenizer import count_tokens, find_soft_boundary
from .validation import validate_graph_invariants, validate_json

__all__ = [
    # config
    "load_config",
    "ConfigValidationError",
    # tokenizer
    "count_tokens",
    "find_soft_boundary",
    # validation
    "validate_json",
    "validate_graph_invariants",
    # llm_client
    "OpenAIClient",
    "TPMBucket",
    # llm_embeddings
    "EmbeddingsClient",
    "get_embeddings",
    "cosine_similarity_batch",
    # exit_codes
    "EXIT_SUCCESS",
    "EXIT_CONFIG_ERROR",
    "EXIT_INPUT_ERROR",
    "EXIT_RUNTIME_ERROR",
    "EXIT_API_LIMIT_ERROR",
    "EXIT_IO_ERROR",
    "get_exit_code_name",
    "get_exit_code_description",
    "log_exit",
    # console_encoding
    "setup_console_encoding",
]
