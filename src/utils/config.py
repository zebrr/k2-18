"""
Module for loading and validating iText2KG configuration from TOML file.
"""

import os
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# TOML support for different Python versions
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
    """Exception for configuration validation errors."""

    pass


def _inject_env_api_keys(config: Dict[str, Any]) -> None:
    """
    Injects API keys from environment variables.

    Priority:
    1. Environment variable (if set)
    2. Value from config.toml (if not placeholder)
    3. Validation error
    """
    # Main API key
    env_api_key = os.getenv("OPENAI_API_KEY")
    
    # Internal model OAuth token (for internal models)
    env_internal_token = os.getenv("SOY_TOKEN")

    # itext2kg.api_key
    if "itext2kg" in config:
        current_key = config["itext2kg"].get("api_key", "")
        use_internal = config["itext2kg"].get("use_internal_auth", False)
        
        if use_internal and env_internal_token:
            # For internal models, use SOY_TOKEN
            if not current_key or current_key.startswith("sk-..."):
                config["itext2kg"]["api_key"] = env_internal_token
        elif not use_internal and env_api_key:
            # For external models, use OPENAI_API_KEY
            if not current_key or current_key.startswith("sk-..."):
                config["itext2kg"]["api_key"] = env_api_key

    # refiner.api_key
    if "refiner" in config:
        current_key = config["refiner"].get("api_key", "")
        use_internal = config["refiner"].get("use_internal_auth", False)
        
        if use_internal and env_internal_token:
            # For internal models, use SOY_TOKEN
            if not current_key or current_key.startswith("sk-..."):
                config["refiner"]["api_key"] = env_internal_token
        elif not use_internal and env_api_key:
            # For external models, use OPENAI_API_KEY
            if not current_key or current_key.startswith("sk-..."):
                config["refiner"]["api_key"] = env_api_key

    # Embedding API keys (can use separate key)
    env_embedding_key = os.getenv("OPENAI_EMBEDDING_API_KEY", env_api_key)
    env_internal_embedding_key = os.getenv("INTERNAL_EMBEDDING_API_KEY")

    # dedup.embedding_api_key
    if "dedup" in config:
        current_key = config["dedup"].get("embedding_api_key", "")
        use_internal = config["dedup"].get("embedding_use_internal_auth", False)
        if not current_key or current_key.startswith("sk-..."):
            chosen = None
            if use_internal:
                chosen = env_internal_embedding_key
            else:
                chosen = env_embedding_key
            if chosen:
                config["dedup"]["embedding_api_key"] = chosen

    # refiner.embedding_api_key
    if "refiner" in config:
        current_key = config["refiner"].get("embedding_api_key", "")
        use_internal = config["refiner"].get("embedding_use_internal_auth", False)
        if not current_key or current_key.startswith("sk-..."):
            chosen = None
            if use_internal:
                chosen = env_internal_embedding_key
            else:
                chosen = env_embedding_key
            if chosen:
                config["refiner"]["embedding_api_key"] = chosen


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads and validates configuration from TOML file.

    Args:
        config_path: Path to configuration file.
                    If None, uses src/config.toml

    Returns:
        Dictionary with validated configuration

    Raises:
        ConfigValidationError: On validation errors
        FileNotFoundError: If configuration file not found
    """
    if config_path is None:
        # Determine path to config.toml relative to this file
        current_dir = Path(__file__).parent.parent
        config_path = current_dir / "config.toml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load TOML file
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except Exception as e:
        raise ConfigValidationError(f"Failed to parse TOML file: {e}")

    # Check if this is a viz config (has viz-specific sections)
    is_viz_config = "graph2metrics" in config or "visualization" in config

    # Only inject API keys and validate main sections for non-viz configs
    if not is_viz_config:
        # Inject API keys from env variables
        _inject_env_api_keys(config)

        # Validate configuration
        try:
            _validate_config(config)
        except Exception as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")

    # Validate is_reasoning parameter is present (only for main pipeline)
    if "itext2kg" in config:
        if "is_reasoning" not in config["itext2kg"]:
            raise ConfigValidationError(
                "Parameter 'is_reasoning' is required in [itext2kg] section"
            )

    if "refiner" in config:
        if "is_reasoning" not in config["refiner"]:
            raise ConfigValidationError("Parameter 'is_reasoning' is required in [refiner] section")

    # Optional consistency check (warning, not error)
    import logging

    logger = logging.getLogger(__name__)
    for section in ["itext2kg", "refiner"]:
        if section in config:
            is_reasoning = config[section].get("is_reasoning", False)
            has_temperature = config[section].get("temperature") is not None
            has_reasoning_effort = config[section].get("reasoning_effort") is not None

            if is_reasoning and has_temperature:
                logger.warning(
                    f"[{section}] Reasoning model with temperature parameter - "
                    f"might be ignored by API"
                )
            if not is_reasoning and has_reasoning_effort:
                logger.warning(
                    f"[{section}] Non-reasoning model with reasoning_effort - "
                    f"will be ignored by API"
                )

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
        "allowed_extensions": list,
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
        "max_retries": int,
    }

    _validate_required_fields(section, required_fields, "itext2kg")

    # Проверяем диапазоны
    if section["tpm_limit"] <= 0:
        raise ConfigValidationError("itext2kg.tpm_limit must be positive")

    if not (1 <= section["max_completion"] <= 100000):
        raise ConfigValidationError("itext2kg.max_completion must be between 1 and 100000")

    if section["log_level"] not in ["debug", "info", "warning", "error"]:
        raise ConfigValidationError(
            "itext2kg.log_level must be one of: debug, info, warning, error"
        )

    # Updated api_key check - handle both external and internal models
    use_internal = section.get("use_internal_auth", False)
    if not section["api_key"].strip() or section["api_key"].startswith("sk-..."):
        if use_internal:
            # For internal models, check SOY_TOKEN
            if not os.getenv("SOY_TOKEN"):
                raise ConfigValidationError(
                    "itext2kg.api_key not configured for internal model. Either:\n"
                    "1. Set SOY_TOKEN environment variable\n"
                    "2. Provide valid OAuth token in config.toml"
                )
        else:
            # For external models, check OPENAI_API_KEY
            if not os.getenv("OPENAI_API_KEY"):
                raise ConfigValidationError(
                    "itext2kg.api_key not configured. Either:\n"
                    "1. Set OPENAI_API_KEY environment variable\n"
                    "2. Provide valid key in config.toml"
                )

    if section["timeout"] <= 0:
        raise ConfigValidationError("itext2kg.timeout must be positive")

    if section["max_retries"] < 0:
        raise ConfigValidationError("itext2kg.max_retries must be non-negative")

    # Проверяем температуру, если она указана
    if "temperature" in section:
        temp = section["temperature"]
        if not (0 <= temp <= 2):
            raise ConfigValidationError("itext2kg.temperature must be between 0 and 2")

    # Validate optional response_chain_depth
    if "response_chain_depth" in section:
        depth = section["response_chain_depth"]
        if not isinstance(depth, int) or depth < 0:
            raise ConfigValidationError(
                "itext2kg.response_chain_depth must be a non-negative integer"
            )

    # Validate optional truncation
    if "truncation" in section:
        truncation = section["truncation"]
        if truncation not in ["auto", "disabled"]:
            raise ConfigValidationError("itext2kg.truncation must be 'auto' or 'disabled'")

    # Validate internal model parameters
    if use_internal:
        # For internal models, base_url is required
        if "base_url" not in section:
            raise ConfigValidationError("itext2kg.base_url is required when use_internal_auth=true")
        
        # Validate base_url format
        base_url = section["base_url"]
        if not isinstance(base_url, str) or not base_url.startswith("https://"):
            raise ConfigValidationError("itext2kg.base_url must be a valid HTTPS URL")
            
        # model_path is optional but recommended for internal models
        if "model_path" in section:
            model_path = section["model_path"]
            if not isinstance(model_path, str) or not model_path.strip():
                raise ConfigValidationError("itext2kg.model_path must be a non-empty string")


def _validate_dedup_section(section: Dict[str, Any]) -> None:
    """Валидирует секцию [dedup]."""
    required_fields = {
        "embedding_model": str,
        "sim_threshold": float,
        "len_ratio_min": float,
        "faiss_M": int,
        "faiss_efC": int,
        "faiss_metric": str,
        "k_neighbors": int,
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

    # Проверка ключей/токенов для эмбеддингов (внешние/внутренние)
    use_internal_embeddings = section.get("embedding_use_internal_auth", False)
    embedding_key = section.get("embedding_api_key", "")
    if use_internal_embeddings:
        # Для внутренних эмбеддингов: предупреждаем, но не блокируем сборку других утилит (например, slicer)
        if not embedding_key.strip() and not os.getenv("INTERNAL_EMBEDDING_API_KEY"):
            logging.warning(
                "[dedup] Internal embeddings enabled but no token configured. Set dedup.embedding_api_key or INTERNAL_EMBEDDING_API_KEY."
            )
        if "embedding_base_url" not in section or not str(section["embedding_base_url"]).strip():
            logging.warning(
                "[dedup] Internal embeddings enabled but embedding_base_url is missing."
            )
    else:
        # Внешние эмбеддинги OpenAI: предупреждаем, но не валим валидацию
        if not embedding_key or embedding_key.startswith("sk-..."):
            if not (os.getenv("OPENAI_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")):
                logging.warning(
                    "[dedup] External embeddings key not configured. Set dedup.embedding_api_key or OPENAI_EMBEDDING_API_KEY/OPENAI_API_KEY."
                )


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
        # Веса удалены - теперь они прописаны в промптах fw/bw
        # "weight_low": float,
        # "weight_mid": float,
        # "weight_high": float,
    }

    _validate_required_fields(section, required_fields, "refiner")

    # Проверяем диапазоны
    if not (0.0 <= section["sim_threshold"] <= 1.0):
        raise ConfigValidationError("refiner.sim_threshold must be between 0.0 and 1.0")

    if section["max_pairs_per_node"] <= 0:
        raise ConfigValidationError("refiner.max_pairs_per_node must be positive")

    # Updated api_key check - handle both external and internal models
    use_internal = section.get("use_internal_auth", False)
    if not section["api_key"].strip() or section["api_key"].startswith("sk-..."):
        if use_internal:
            # For internal models, check SOY_TOKEN
            if not os.getenv("SOY_TOKEN"):
                raise ConfigValidationError(
                    "refiner.api_key not configured for internal model. Either:\n"
                    "1. Set SOY_TOKEN environment variable\n"
                    "2. Provide valid OAuth token in config.toml"
                )
        else:
            # For external models, check OPENAI_API_KEY
            if not os.getenv("OPENAI_API_KEY"):
                raise ConfigValidationError(
                    "refiner.api_key not configured. Either:\n"
                    "1. Set OPENAI_API_KEY environment variable\n"
                    "2. Provide valid key in config.toml"
                )

    if section["tpm_limit"] <= 0:
        raise ConfigValidationError("refiner.tpm_limit must be positive")

    if not (1 <= section["max_completion"] <= 100000):
        raise ConfigValidationError("refiner.max_completion must be between 1 and 100000")

    if section["timeout"] <= 0:
        raise ConfigValidationError("refiner.timeout must be positive")

    if section["max_retries"] < 0:
        raise ConfigValidationError("refiner.max_retries must be non-negative")

    # Веса больше не проверяем - они теперь в промптах
    # weights = [section["weight_low"], section["weight_mid"], section["weight_high"]]
    # ...проверки весов удалены...

    # Validate optional response_chain_depth
    if "response_chain_depth" in section:
        depth = section["response_chain_depth"]
        if not isinstance(depth, int) or depth < 0:
            raise ConfigValidationError(
                "refiner.response_chain_depth must be a non-negative integer"
            )

    # Validate optional truncation
    if "truncation" in section:
        truncation = section["truncation"]
        if truncation not in ["auto", "disabled"]:
            raise ConfigValidationError("refiner.truncation must be 'auto' or 'disabled'")

    # Validate internal model parameters
    if use_internal:
        # For internal models, base_url is required
        if "base_url" not in section:
            raise ConfigValidationError("refiner.base_url is required when use_internal_auth=true")
        
        # Validate base_url format
        base_url = section["base_url"]
        if not isinstance(base_url, str) or not base_url.startswith("https://"):
            raise ConfigValidationError("refiner.base_url must be a valid HTTPS URL")
            
        # model_path is optional but recommended for internal models
        if "model_path" in section:
            model_path = section["model_path"]
            if not isinstance(model_path, str) or not model_path.strip():
                raise ConfigValidationError("refiner.model_path must be a non-empty string")

    # Embeddings config for refiner: support internal/external switching similar to dedup
    use_internal_embeddings = section.get("embedding_use_internal_auth", False)
    embedding_key = section.get("embedding_api_key", "")
    if use_internal_embeddings:
        if not embedding_key.strip() and not os.getenv("INTERNAL_EMBEDDING_API_KEY"):
            logging.warning(
                "[refiner] Internal embeddings enabled but no token configured. Set refiner.embedding_api_key or INTERNAL_EMBEDDING_API_KEY."
            )
        if "embedding_base_url" not in section or not str(section["embedding_base_url"]).strip():
            logging.warning(
                "[refiner] Internal embeddings enabled but embedding_base_url is missing."
            )
    else:
        if not embedding_key or embedding_key.startswith("sk-..."):
            if not (os.getenv("OPENAI_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")):
                logging.warning(
                    "[refiner] External embeddings key not configured. Set refiner.embedding_api_key or OPENAI_EMBEDDING_API_KEY/OPENAI_API_KEY."
                )


def _validate_required_fields(
    section: Dict[str, Any], required_fields: Dict[str, type], section_name: str
) -> None:
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
