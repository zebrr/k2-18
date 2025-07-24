"""
Тесты для модуля загрузки и валидации конфигурации.
"""

import pytest
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

from src.utils.config import load_config, ConfigValidationError


class TestConfigLoading:
    """Тесты загрузки конфигурации."""
    
    def test_load_valid_config(self):
        """Тест загрузки корректной конфигурации."""
        valid_config = textwrap.dedent("""
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json", "txt", "md", "html"]

        [itext2kg]
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test123"
        timeout = 45
        max_retries = 6

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        run = true
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.80
        max_pairs_per_node = 20
        model = "gpt-4o-mini"
        api_key = "sk-test456"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = 3
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
        """)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(valid_config)
            f.flush()
            
            config = load_config(f.name)
            
            # Проверяем основные секции
            assert "slicer" in config
            assert "itext2kg" in config
            assert "dedup" in config
            assert "refiner" in config
            
            # Проверяем несколько ключевых значений
            assert config["slicer"]["max_tokens"] == 40000
            assert config["itext2kg"]["model"] == "gpt-4o"
            assert config["dedup"]["sim_threshold"] == 0.97
            assert config["refiner"]["weight_low"] == 0.3
        
        Path(f.name).unlink()  # Удаляем временный файл

    def test_missing_config_file(self):
        """Тест ошибки при отсутствующем файле конфигурации."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.toml")

    def test_invalid_toml_syntax(self):
        """Тест ошибки при некорректном синтаксисе TOML."""
        invalid_toml = "[slicer\nmax_tokens = 40000"  # Пропущена закрывающая скобка
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(invalid_toml)
            f.flush()
            
            with pytest.raises(ConfigValidationError, match="Failed to parse TOML file"):
                load_config(f.name)
        
        Path(f.name).unlink()


class TestSlicerValidation:
    """Тесты валидации секции [slicer]."""
    
    def test_missing_slicer_section(self):
        """Тест ошибки при отсутствующей секции [slicer]."""
        config_without_slicer = textwrap.dedent("""
        [itext2kg]
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6
        """)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_without_slicer)
            f.flush()
            
            with pytest.raises(ConfigValidationError, match="Missing required section: \\[slicer\\]"):
                load_config(f.name)
        
        Path(f.name).unlink()

    def test_invalid_max_tokens(self):
        """Тест валидации max_tokens."""
        config_with_invalid_tokens = textwrap.dedent("""
        [slicer]
        max_tokens = -1000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        run = true
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.80
        max_pairs_per_node = 20
        model = "gpt-4o-mini"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = 3
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
        """)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_with_invalid_tokens)
            f.flush()
            
            with pytest.raises(ConfigValidationError, match="slicer.max_tokens must be positive"):
                load_config(f.name)
        
        Path(f.name).unlink()

    def test_overlap_soft_boundary_validation(self):
        """Тест валидации зависимости overlap и soft_boundary_max_shift."""
        config_with_invalid_overlap = textwrap.dedent("""
        [slicer]
        max_tokens = 40000
        overlap = 1000
        soft_boundary = true
        soft_boundary_max_shift = 900
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        run = true
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.80
        max_pairs_per_node = 20
        model = "gpt-4o-mini"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = 3
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
        """)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_with_invalid_overlap)
            f.flush()
            
            with pytest.raises(ConfigValidationError, match="cannot exceed overlap\\*0.8"):
                load_config(f.name)
        
        Path(f.name).unlink()


class TestItext2kgValidation:
    """Тесты валидации секции [itext2kg]."""
    
    def test_invalid_log_level(self):
        """Тест валидации log_level."""
        config_with_invalid_log_level = textwrap.dedent("""
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "invalid_level"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        run = true
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.80
        max_pairs_per_node = 20
        model = "gpt-4o-mini"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = 3
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
        """)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_with_invalid_log_level)
            f.flush()
            
            with pytest.raises(ConfigValidationError, match="log_level must be one of"):
                load_config(f.name)
        
        Path(f.name).unlink()

    def test_empty_api_key(self):
        """Тест валидации пустого API ключа."""
        config_with_empty_api_key = textwrap.dedent("""
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "   "
        timeout = 45
        max_retries = 6

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        run = true
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.80
        max_pairs_per_node = 20
        model = "gpt-4o-mini"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = 3
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
        """)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_with_empty_api_key)
            f.flush()
            
            with pytest.raises(ConfigValidationError, match="api_key cannot be empty"):
                load_config(f.name)
        
        Path(f.name).unlink()


class TestRefinerValidation:
    """Тесты валидации секции [refiner]."""
    
    def test_invalid_weight_order(self):
        """Тест валидации порядка весов."""
        config_with_invalid_weights = textwrap.dedent("""
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        run = true
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.80
        max_pairs_per_node = 20
        model = "gpt-4o-mini"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = 3
        weight_low = 0.9
        weight_mid = 0.6
        weight_high = 0.3
        """)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_with_invalid_weights)
            f.flush()
            
            with pytest.raises(ConfigValidationError, match="weight_low < weight_mid < weight_high"):
                load_config(f.name)
        
        Path(f.name).unlink()

    def test_weight_out_of_range(self):
        """Тест валидации весов вне диапазона [0,1]."""
        config_with_invalid_weight_range = textwrap.dedent("""
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        run = true
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.80
        max_pairs_per_node = 20
        model = "gpt-4o-mini"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = 3
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 1.5
        """)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_with_invalid_weight_range)
            f.flush()
            
            with pytest.raises(ConfigValidationError, match="weight_high must be between 0.0 and 1.0"):
                load_config(f.name)
        
        Path(f.name).unlink()


class TestTypeValidation:
    """Тесты валидации типов данных."""
    
    def test_wrong_type_validation(self):
        """Тест валидации неправильных типов данных."""
        config_with_wrong_types = textwrap.dedent("""
        [slicer]
        max_tokens = "not_a_number"
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        run = true
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.80
        max_pairs_per_node = 20
        model = "gpt-4o-mini"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = 3
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
        """)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_with_wrong_types)
            f.flush()
            
            with pytest.raises(ConfigValidationError, match="must be int, got str"):
                load_config(f.name)
        
        Path(f.name).unlink()