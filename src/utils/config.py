"""
Модуль для загрузки и валидации конфигурации iText2KG из TOML файла.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Union

# Поддержка TOML для разных версий Python
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "tomli library is required for Python < 3.11. "
            "Install it with: pip install tomli>=2.0.0"
        )


class ConfigValidationError(Exception):
    """Исключение для ошибок валидации конфигурации."""
    pass


def load_config(config_path: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Загружает и валидирует конфигурацию из TOML файла.
    
    Args:
        config_path: Путь к файлу конфигурации. 
                    Если None, использует src/config.toml
    
    Returns:
        Словарь с проверенной конфигурацией
        
    Raises:
        ConfigValidationError: При ошибках валидации
        FileNotFoundError: Если файл конфигурации не найден
    """
    if config_path is None:
        # Определяем путь к config.toml относительно этого файла
        current_dir = Path(__file__).parent.parent
        config_path = current_dir / "config.toml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Загружаем TOML файл
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except Exception as e:
        raise ConfigValidationError(f"Failed to parse TOML file: {e}")
    
    # Валидируем конфигурацию
    try:
        _validate_config(config)
    except Exception as e:
        raise ConfigValidationError(f"Configuration validation failed: {e}")
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Валидирует полную структуру конфигурации."""
    required_sections = ["slicer", "itext2kg", "dedup", "refiner"]
    
    for section in required_sections:
        if section not in config:
            raise ConfigValidationError(f"Missing required section: [{section}]")
    
    # Валидируем каждую секцию
    _validate_slicer_section(config["slicer"])
    _validate_itext2kg_section(config["itext2kg"])
    _validate_dedup_section(config["dedup"])
    _validate_refiner_section(config["refiner"])


def _validate_slicer_section(section: Dict[str, Any]) -> None:
    """Валидирует секцию [slicer]."""
    required_fields = {
        "max_tokens": int,
        "overlap": int,
        "soft_boundary": bool,
        "soft_boundary_max_shift": int,
        "tokenizer": str,
        "allowed_extensions": list
    }
    
    _validate_required_fields(section, required_fields, "slicer")
    
    # Проверяем диапазоны значений
    if section["max_tokens"] <= 0:
        raise ConfigValidationError("slicer.max_tokens must be positive")
    
    if section["overlap"] < 0:
        raise ConfigValidationError("slicer.overlap must be non-negative")
    
    if section["soft_boundary_max_shift"] < 0:
        raise ConfigValidationError("slicer.soft_boundary_max_shift must be non-negative")
    
    # Валидация зависимости overlap и soft_boundary_max_shift
    if section["overlap"] > 0:
        max_allowed_shift = int(section["overlap"] * 0.8)
        if section["soft_boundary_max_shift"] > max_allowed_shift:
            raise ConfigValidationError(
                f"slicer.soft_boundary_max_shift ({section['soft_boundary_max_shift']}) "
                f"cannot exceed overlap*0.8 ({max_allowed_shift}) when overlap > 0"
            )
    
    # Проверяем tokenizer
    if section["tokenizer"] != "o200k_base":
        raise ConfigValidationError("slicer.tokenizer must be 'o200k_base'")
    
    # Проверяем allowed_extensions
    if not section["allowed_extensions"]:
        raise ConfigValidationError("slicer.allowed_extensions cannot be empty")


def _validate_itext2kg_section(section: Dict[str, Any]) -> None:
    """Валидирует секцию [itext2kg]."""
    required_fields = {
        "model": str,
        "tpm_limit": int,
        "max_completion": int,
        "log_level": str,
        "api_key": str,
        "timeout": int,
        "max_retries": int
    }
    
    _validate_required_fields(section, required_fields, "itext2kg")
    
    # Проверяем диапазоны
    if section["tpm_limit"] <= 0:
        raise ConfigValidationError("itext2kg.tpm_limit must be positive")
    
    if not (1 <= section["max_completion"] <= 100000):
        raise ConfigValidationError("itext2kg.max_completion must be between 1 and 100000")
    
    if section["log_level"] not in ["debug", "info", "warning", "error"]:
        raise ConfigValidationError("itext2kg.log_level must be one of: debug, info, warning, error")
    
    if not section["api_key"].strip():
        raise ConfigValidationError("itext2kg.api_key cannot be empty")
    
    if section["timeout"] <= 0:
        raise ConfigValidationError("itext2kg.timeout must be positive")
    
    if section["max_retries"] < 0:
        raise ConfigValidationError("itext2kg.max_retries must be non-negative")


def _validate_dedup_section(section: Dict[str, Any]) -> None:
    """Валидирует секцию [dedup]."""
    required_fields = {
        "embedding_model": str,
        "sim_threshold": float,
        "len_ratio_min": float,
        "faiss_M": int,
        "faiss_efC": int,
        "faiss_metric": str,
        "k_neighbors": int
    }
    
    _validate_required_fields(section, required_fields, "dedup")
    
    # Проверяем диапазоны
    if not (0.0 <= section["sim_threshold"] <= 1.0):
        raise ConfigValidationError("dedup.sim_threshold must be between 0.0 and 1.0")
    
    if not (0.0 <= section["len_ratio_min"] <= 1.0):
        raise ConfigValidationError("dedup.len_ratio_min must be between 0.0 and 1.0")
    
    if section["faiss_M"] <= 0:
        raise ConfigValidationError("dedup.faiss_M must be positive")
    
    if section["faiss_efC"] <= 0:
        raise ConfigValidationError("dedup.faiss_efC must be positive")
    
    if section["faiss_metric"] not in ["INNER_PRODUCT", "L2"]:
        raise ConfigValidationError("dedup.faiss_metric must be 'INNER_PRODUCT' or 'L2'")
    
    if section["k_neighbors"] <= 0:
        raise ConfigValidationError("dedup.k_neighbors must be positive")


def _validate_refiner_section(section: Dict[str, Any]) -> None:
    """Валидирует секцию [refiner]."""
    required_fields = {
        "run": bool,
        "embedding_model": str,
        "sim_threshold": float,
        "max_pairs_per_node": int,
        "model": str,
        "api_key": str,
        "tpm_limit": int,
        "max_completion": int,
        "timeout": int,
        "max_retries": int,
        "weight_low": float,
        "weight_mid": float,
        "weight_high": float
    }
    
    _validate_required_fields(section, required_fields, "refiner")
    
    # Проверяем диапазоны
    if not (0.0 <= section["sim_threshold"] <= 1.0):
        raise ConfigValidationError("refiner.sim_threshold must be between 0.0 and 1.0")
    
    if section["max_pairs_per_node"] <= 0:
        raise ConfigValidationError("refiner.max_pairs_per_node must be positive")
    
    if not section["api_key"].strip():
        raise ConfigValidationError("refiner.api_key cannot be empty")
    
    if section["tpm_limit"] <= 0:
        raise ConfigValidationError("refiner.tpm_limit must be positive")
    
    if not (1 <= section["max_completion"] <= 100000):
        raise ConfigValidationError("refiner.max_completion must be between 1 and 100000")
    
    if section["timeout"] <= 0:
        raise ConfigValidationError("refiner.timeout must be positive")
    
    if section["max_retries"] < 0:
        raise ConfigValidationError("refiner.max_retries must be non-negative")
    
    # Проверяем веса
    weights = [section["weight_low"], section["weight_mid"], section["weight_high"]]
    
    for i, weight in enumerate(weights):
        if not (0.0 <= weight <= 1.0):
            weight_names = ["weight_low", "weight_mid", "weight_high"]
            raise ConfigValidationError(f"refiner.{weight_names[i]} must be between 0.0 and 1.0")
    
    if not (section["weight_low"] < section["weight_mid"] < section["weight_high"]):
        raise ConfigValidationError("refiner weights must satisfy: weight_low < weight_mid < weight_high")


def _validate_required_fields(section: Dict[str, Any], required_fields: Dict[str, type], section_name: str) -> None:
    """Проверяет наличие и типы обязательных полей в секции."""
    for field_name, expected_type in required_fields.items():
        if field_name not in section:
            raise ConfigValidationError(f"Missing required field: {section_name}.{field_name}")
        
        actual_value = section[field_name]
        if not isinstance(actual_value, expected_type):
            raise ConfigValidationError(
                f"Field {section_name}.{field_name} must be {expected_type.__name__}, "
                f"got {type(actual_value).__name__}"
            )