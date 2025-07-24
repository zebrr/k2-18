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
from .config import load_config, ConfigValidationError
from .tokenizer import count_tokens, find_soft_boundary
from .validation import validate_json, validate_graph_invariants
from .llm_client import OpenAIClient, TPMBucket
from .llm_embeddings import EmbeddingsClient, get_embeddings, cosine_similarity_batch
from .exit_codes import (
    EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR, EXIT_API_LIMIT_ERROR, EXIT_IO_ERROR,
    get_exit_code_name, get_exit_code_description, log_exit
)
from .console_encoding import setup_console_encoding

__all__ = [
    # config
    'load_config',
    'ConfigValidationError',
    
    # tokenizer
    'count_tokens',
    'find_soft_boundary',
    
    # validation
    'validate_json',
    'validate_graph_invariants',
    
    # llm_client
    'OpenAIClient',
    'TPMBucket',
    
    # llm_embeddings
    'EmbeddingsClient',
    'get_embeddings',
    'cosine_similarity_batch',
    
    # exit_codes
    'EXIT_SUCCESS',
    'EXIT_CONFIG_ERROR',
    'EXIT_INPUT_ERROR',
    'EXIT_RUNTIME_ERROR',
    'EXIT_API_LIMIT_ERROR',
    'EXIT_IO_ERROR',
    'get_exit_code_name',
    'get_exit_code_description',
    'log_exit',

    # console_encoding
    'setup_console_encoding',
]
