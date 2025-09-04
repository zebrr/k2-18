"""
Тесты для модуля загрузки и валидации конфигурации.
"""

import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from src.utils.config import ConfigValidationError, load_config


class TestConfigLoading:
    """Тесты загрузки конфигурации."""

    def test_load_valid_config(self):
        """Тест загрузки корректной конфигурации."""
        valid_config = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json", "txt", "md", "html"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
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
            # Веса больше не в конфиге - перенесены в промпты
            # assert config["refiner"]["weight_low"] == 0.3

        Path(f.name).unlink()  # Удаляем временный файл

    def test_missing_config_file(self):
        """Тест ошибки при отсутствующем файле конфигурации."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.toml")

    def test_invalid_toml_syntax(self):
        """Тест ошибки при некорректном синтаксисе TOML."""
        invalid_toml = "[slicer\nmax_tokens = 40000"  # Пропущена закрывающая скобка

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(invalid_toml)
            f.flush()

            with pytest.raises(ConfigValidationError, match="Failed to parse TOML file"):
                load_config(f.name)

        Path(f.name).unlink()


class TestSlicerValidation:
    """Тесты валидации секции [slicer]."""

    def test_missing_slicer_section(self):
        """Тест ошибки при отсутствующей секции [slicer]."""
        config_without_slicer = textwrap.dedent(
            """
        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_without_slicer)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="Missing required section: \\[slicer\\]"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_invalid_max_tokens(self):
        """Тест валидации max_tokens."""
        config_with_invalid_tokens = textwrap.dedent(
            """
        [slicer]
        max_tokens = -1000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_tokens)
            f.flush()

            with pytest.raises(ConfigValidationError, match="slicer.max_tokens must be positive"):
                load_config(f.name)

        Path(f.name).unlink()

    def test_overlap_soft_boundary_validation(self):
        """Тест валидации зависимости overlap и soft_boundary_max_shift."""
        config_with_invalid_overlap = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 1000
        soft_boundary = true
        soft_boundary_max_shift = 900
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_overlap)
            f.flush()

            with pytest.raises(ConfigValidationError, match="cannot exceed overlap\\*0.8"):
                load_config(f.name)

        Path(f.name).unlink()

    def test_wrong_tokenizer(self):
        """Test that tokenizer must be 'o200k_base'."""
        config_with_wrong_tokenizer = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "gpt2"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_wrong_tokenizer)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="slicer.tokenizer must be 'o200k_base'"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_empty_allowed_extensions(self):
        """Test that allowed_extensions cannot be empty."""
        config_with_empty_extensions = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = []

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_empty_extensions)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="slicer.allowed_extensions cannot be empty"
            ):
                load_config(f.name)

        Path(f.name).unlink()


class TestItext2kgValidation:
    """Тесты валидации секции [itext2kg]."""

    def test_invalid_log_level(self):
        """Тест валидации log_level."""
        config_with_invalid_log_level = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_log_level)
            f.flush()

            with pytest.raises(ConfigValidationError, match="log_level must be one of"):
                load_config(f.name)

        Path(f.name).unlink()

    def test_empty_api_key(self):
        """Тест валидации пустого API ключа."""
        # Сохраняем текущее значение переменной окружения
        original_api_key = os.environ.get("OPENAI_API_KEY")

        # Очищаем переменную окружения для теста
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            config_with_empty_api_key = textwrap.dedent(
                """
            [slicer]
            max_tokens = 40000
            overlap = 0
            soft_boundary = true
            soft_boundary_max_shift = 500
            tokenizer = "o200k_base"
            allowed_extensions = ["json"]

            [itext2kg]
            is_reasoning = false
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
            is_reasoning = false
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
            """
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                f.write(config_with_empty_api_key)
                f.flush()

                with pytest.raises(ConfigValidationError, match="api_key not configured"):
                    load_config(f.name)
        finally:
            # Восстанавливаем переменную окружения
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key

        Path(f.name).unlink()

    def test_missing_is_reasoning_parameter(self):
        """Test that is_reasoning parameter is required in itext2kg section."""
        config_without_is_reasoning = textwrap.dedent(
            """
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
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_without_is_reasoning)
            f.flush()

            with pytest.raises(
                ConfigValidationError,
                match="Parameter 'is_reasoning' is required in \\[itext2kg\\] section",
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_consistency_warning_reasoning_with_temperature(self, caplog):
        """Test warning when reasoning model has temperature parameter."""
        import logging

        config_with_reasoning_temperature = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = true
        model = "o1-preview"
        temperature = 0.5
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
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_reasoning_temperature)
            f.flush()

            with caplog.at_level(logging.WARNING):
                load_config(f.name)

            # Check that warning was logged
            assert any(
                "Reasoning model with temperature parameter" in record.message
                for record in caplog.records
            )
            assert any("[itext2kg]" in record.message for record in caplog.records)

        Path(f.name).unlink()

    def test_consistency_warning_non_reasoning_with_effort(self, caplog):
        """Test warning when non-reasoning model has reasoning_effort."""
        import logging

        config_with_non_reasoning_effort = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        reasoning_effort = "medium"
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
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_non_reasoning_effort)
            f.flush()

            with caplog.at_level(logging.WARNING):
                load_config(f.name)

            # Check that warning was logged
            assert any(
                "Non-reasoning model with reasoning_effort" in record.message
                for record in caplog.records
            )
            assert any("[itext2kg]" in record.message for record in caplog.records)

        Path(f.name).unlink()

    def test_negative_timeout(self):
        """Test that timeout must be positive."""
        config_with_negative_timeout = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = -5
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
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_negative_timeout)
            f.flush()

            with pytest.raises(ConfigValidationError, match="itext2kg.timeout must be positive"):
                load_config(f.name)

        Path(f.name).unlink()

    def test_negative_max_retries(self):
        """Test that max_retries must be non-negative."""
        config_with_negative_retries = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = -1

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_negative_retries)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="itext2kg.max_retries must be non-negative"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_invalid_response_chain_depth_type(self):
        """Test that response_chain_depth must be an integer."""
        config_with_invalid_depth_type = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6
        response_chain_depth = "not_an_integer"

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_depth_type)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="response_chain_depth must be a non-negative integer"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_negative_response_chain_depth(self):
        """Test that response_chain_depth must be non-negative."""
        config_with_negative_depth = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6
        response_chain_depth = -5

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_negative_depth)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="response_chain_depth must be a non-negative integer"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_invalid_truncation_value(self):
        """Test that truncation must be 'auto' or 'disabled'."""
        config_with_invalid_truncation = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6
        truncation = "invalid_value"

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_truncation)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="truncation must be 'auto' or 'disabled'"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_truncation_wrong_type(self):
        """Test that truncation must be a string."""
        config_with_wrong_type_truncation = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6
        truncation = 123

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_wrong_type_truncation)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="truncation must be 'auto' or 'disabled'"
            ):
                load_config(f.name)

        Path(f.name).unlink()


class TestRefinerValidation:
    """Тесты валидации секции [refiner]."""

    def test_invalid_weight_order(self):
        """Тест валидации порядка весов - ОТКЛЮЧЕН, веса удалены из конфига."""
        pass
        return
        config_with_invalid_weights = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
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
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_weights)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="weight_low < weight_mid < weight_high"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_weight_out_of_range(self):
        """Тест валидации весов вне диапазона [0,1] - ОТКЛЮЧЕН, веса удалены из конфига."""
        pass
        return
        config_with_invalid_weight_range = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
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
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_weight_range)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="weight_high must be between 0.0 and 1.0"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_missing_is_reasoning_parameter(self):
        """Test that is_reasoning parameter is required in refiner section."""
        config_without_is_reasoning = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_without_is_reasoning)
            f.flush()

            with pytest.raises(
                ConfigValidationError,
                match="Parameter 'is_reasoning' is required in \\[refiner\\] section",
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_negative_timeout(self):
        """Test that refiner timeout must be positive."""
        config_with_negative_timeout = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
        run = true
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.80
        max_pairs_per_node = 20
        model = "gpt-4o-mini"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 0
        max_retries = 3
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_negative_timeout)
            f.flush()

            with pytest.raises(ConfigValidationError, match="refiner.timeout must be positive"):
                load_config(f.name)

        Path(f.name).unlink()

    def test_negative_max_retries(self):
        """Test that refiner max_retries must be non-negative."""
        config_with_negative_retries = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
        run = true
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.80
        max_pairs_per_node = 20
        model = "gpt-4o-mini"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = -2
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_negative_retries)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="refiner.max_retries must be non-negative"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_max_pairs_per_node_zero(self):
        """Test that max_pairs_per_node must be positive."""
        config_with_zero_pairs = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
        run = true
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.80
        max_pairs_per_node = 0
        model = "gpt-4o-mini"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = 3
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_zero_pairs)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="refiner.max_pairs_per_node must be positive"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_invalid_response_chain_depth(self):
        """Test that refiner response_chain_depth must be a non-negative integer."""
        config_with_invalid_depth = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
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
        response_chain_depth = -3
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_depth)
            f.flush()

            with pytest.raises(
                ConfigValidationError,
                match="refiner.response_chain_depth must be a non-negative integer",
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_invalid_truncation(self):
        """Test that refiner truncation must be 'auto' or 'disabled'."""
        config_with_invalid_truncation = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
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
        truncation = "always"
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_truncation)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="refiner.truncation must be 'auto' or 'disabled'"
            ):
                load_config(f.name)

        Path(f.name).unlink()


class TestTypeValidation:
    """Тесты валидации типов данных."""

    def test_wrong_type_validation(self):
        """Тест валидации неправильных типов данных."""
        config_with_wrong_types = textwrap.dedent(
            """
        [slicer]
        max_tokens = "not_a_number"
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_wrong_types)
            f.flush()

            with pytest.raises(ConfigValidationError, match="must be int, got str"):
                load_config(f.name)

        Path(f.name).unlink()


class TestEnvironmentVariables:
    """Tests for environment variable injection."""

    def test_api_key_from_environment(self):
        """Test API key injection from OPENAI_API_KEY env var."""
        # Save original env var
        original_api_key = os.environ.get("OPENAI_API_KEY")

        # Set test env var
        os.environ["OPENAI_API_KEY"] = "sk-test-from-env"

        try:
            config_with_placeholder = textwrap.dedent(
                """
            [slicer]
            max_tokens = 40000
            overlap = 0
            soft_boundary = true
            soft_boundary_max_shift = 500
            tokenizer = "o200k_base"
            allowed_extensions = ["json"]

            [itext2kg]
            is_reasoning = false
            model = "gpt-4o"
            tpm_limit = 120000
            max_completion = 4096
            log_level = "info"
            api_key = "sk-..."
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
            is_reasoning = false
            run = true
            embedding_model = "text-embedding-3-small"
            sim_threshold = 0.80
            max_pairs_per_node = 20
            model = "gpt-4o-mini"
            api_key = "sk-..."
            tpm_limit = 60000
            max_completion = 2048
            timeout = 30
            max_retries = 3
            weight_low = 0.3
            weight_mid = 0.6
            weight_high = 0.9
            """
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                f.write(config_with_placeholder)
                f.flush()

                config = load_config(f.name)

                # Check that API keys were injected from env
                assert config["itext2kg"]["api_key"] == "sk-test-from-env"
                assert config["refiner"]["api_key"] == "sk-test-from-env"

            Path(f.name).unlink()

        finally:
            # Restore original env var
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def test_embedding_api_key_from_environment(self):
        """Test embedding API key injection from env vars."""
        # Save original env vars
        original_api_key = os.environ.get("OPENAI_API_KEY")
        original_embedding_key = os.environ.get("OPENAI_EMBEDDING_API_KEY")
        original_internal_embedding_key = os.environ.get("INTERNAL_EMBEDDING_API_KEY")

        # Test with OPENAI_EMBEDDING_API_KEY
        os.environ["OPENAI_EMBEDDING_API_KEY"] = "sk-embedding-key"
        os.environ["OPENAI_API_KEY"] = "sk-general-key"

        try:
            config_with_placeholder = textwrap.dedent(
                """
            [slicer]
            max_tokens = 40000
            overlap = 0
            soft_boundary = true
            soft_boundary_max_shift = 500
            tokenizer = "o200k_base"
            allowed_extensions = ["json"]

            [itext2kg]
            is_reasoning = false
            model = "gpt-4o"
            tpm_limit = 120000
            max_completion = 4096
            log_level = "info"
            api_key = "sk-test"
            timeout = 45
            max_retries = 6

            [dedup]
            embedding_model = "text-embedding-3-small"
            embedding_api_key = "sk-..."
            sim_threshold = 0.97
            len_ratio_min = 0.8
            faiss_M = 32
            faiss_efC = 200
            faiss_metric = "INNER_PRODUCT"
            k_neighbors = 5

            [refiner]
            is_reasoning = false
            run = true
            embedding_model = "text-embedding-3-small"
            embedding_api_key = ""
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
            """
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                f.write(config_with_placeholder)
                f.flush()

                config = load_config(f.name)

                # Check that embedding API keys were injected
                assert config["dedup"]["embedding_api_key"] == "sk-embedding-key"
                assert config["refiner"]["embedding_api_key"] == "sk-embedding-key"

            Path(f.name).unlink()

            # Test fallback to OPENAI_API_KEY when OPENAI_EMBEDDING_API_KEY is not set
            del os.environ["OPENAI_EMBEDDING_API_KEY"]

            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                f.write(config_with_placeholder)
                f.flush()

                config = load_config(f.name)

                # Check fallback to OPENAI_API_KEY
                assert config["dedup"]["embedding_api_key"] == "sk-general-key"
                assert config["refiner"]["embedding_api_key"] == "sk-general-key"

            Path(f.name).unlink()

        finally:
            # Restore original env vars
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

            if original_embedding_key is not None:
                os.environ["OPENAI_EMBEDDING_API_KEY"] = original_embedding_key
            elif "OPENAI_EMBEDDING_API_KEY" in os.environ:
                del os.environ["OPENAI_EMBEDDING_API_KEY"]
            if original_internal_embedding_key is not None:
                os.environ["INTERNAL_EMBEDDING_API_KEY"] = original_internal_embedding_key
            elif "INTERNAL_EMBEDDING_API_KEY" in os.environ:
                del os.environ["INTERNAL_EMBEDDING_API_KEY"]

    def test_internal_embeddings_env_injection(self):
        """When embedding_use_internal_auth=true, INTERNAL_EMBEDDING_API_KEY should be injected."""
        import tempfile
        from pathlib import Path
        import textwrap

        # Save originals
        original_internal_embedding_key = os.environ.get("INTERNAL_EMBEDDING_API_KEY")
        try:
            os.environ["INTERNAL_EMBEDDING_API_KEY"] = "oauth-internal-token"

            config_text = textwrap.dedent(
                """
            [slicer]
            max_tokens = 40000
            overlap = 0
            soft_boundary = true
            soft_boundary_max_shift = 500
            tokenizer = "o200k_base"
            allowed_extensions = ["json"]

            [itext2kg]
            is_reasoning = false
            model = "gpt-4o"
            tpm_limit = 120000
            max_completion = 4096
            log_level = "info"
            api_key = "sk-test"
            timeout = 45
            max_retries = 6

            [dedup]
            embedding_model = "text-embedding-3-small"
            embedding_api_key = "sk-..."
            embedding_use_internal_auth = true
            embedding_base_url = "http://example.local/internal/embed"
            sim_threshold = 0.97
            len_ratio_min = 0.8
            faiss_M = 32
            faiss_efC = 200
            faiss_metric = "INNER_PRODUCT"
            k_neighbors = 5

            [refiner]
            is_reasoning = false
            run = true
            embedding_model = "text-embedding-3-small"
            embedding_api_key = ""
            embedding_use_internal_auth = true
            embedding_base_url = "http://example.local/internal/embed"
            sim_threshold = 0.80
            max_pairs_per_node = 20
            model = "gpt-4o-mini"
            api_key = "sk-test"
            tpm_limit = 60000
            max_completion = 2048
            timeout = 30
            max_retries = 3
            """
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                f.write(config_text)
                f.flush()

                config = load_config(f.name)
                assert config["dedup"]["embedding_api_key"] == "oauth-internal-token"
                assert config["refiner"]["embedding_api_key"] == "oauth-internal-token"

            Path(f.name).unlink()

        finally:
            if original_internal_embedding_key is not None:
                os.environ["INTERNAL_EMBEDDING_API_KEY"] = original_internal_embedding_key
            elif "INTERNAL_EMBEDDING_API_KEY" in os.environ:
                del os.environ["INTERNAL_EMBEDDING_API_KEY"]

    def test_placeholder_detection(self):
        """Test that sk-... placeholders trigger env var lookup."""
        # Save original env var
        original_api_key = os.environ.get("OPENAI_API_KEY")

        # Clear env var to test error
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            config_with_placeholder = textwrap.dedent(
                """
            [slicer]
            max_tokens = 40000
            overlap = 0
            soft_boundary = true
            soft_boundary_max_shift = 500
            tokenizer = "o200k_base"
            allowed_extensions = ["json"]

            [itext2kg]
            is_reasoning = false
            model = "gpt-4o"
            tpm_limit = 120000
            max_completion = 4096
            log_level = "info"
            api_key = "sk-..."
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
            is_reasoning = false
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
            """
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                f.write(config_with_placeholder)
                f.flush()

                # Should raise error because env var is not set and placeholder is used
                with pytest.raises(ConfigValidationError, match="api_key not configured"):
                    load_config(f.name)

            Path(f.name).unlink()

        finally:
            # Restore original env var
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key


class TestParametrizedValidation:
    """Parametrized tests for repetitive validation checks."""

    @pytest.mark.parametrize(
        "section,field,value,error_pattern",
        [
            # itext2kg numeric validations
            ("itext2kg", "timeout", 0, "must be positive"),
            ("itext2kg", "timeout", -1, "must be positive"),
            ("itext2kg", "tpm_limit", 0, "must be positive"),
            ("itext2kg", "tpm_limit", -100, "must be positive"),
            ("itext2kg", "max_completion", 0, "must be between 1 and 100000"),
            ("itext2kg", "max_completion", 100001, "must be between 1 and 100000"),
            ("itext2kg", "max_retries", -1, "must be non-negative"),
            # refiner numeric validations
            ("refiner", "timeout", 0, "must be positive"),
            ("refiner", "timeout", -5, "must be positive"),
            ("refiner", "tpm_limit", 0, "must be positive"),
            ("refiner", "tpm_limit", -1000, "must be positive"),
            ("refiner", "max_completion", 0, "must be between 1 and 100000"),
            ("refiner", "max_completion", 200000, "must be between 1 and 100000"),
            ("refiner", "max_retries", -2, "must be non-negative"),
            ("refiner", "max_pairs_per_node", 0, "must be positive"),
            ("refiner", "max_pairs_per_node", -5, "must be positive"),
            # dedup numeric validations
            ("dedup", "faiss_M", 0, "must be positive"),
            ("dedup", "faiss_M", -1, "must be positive"),
            ("dedup", "faiss_efC", 0, "must be positive"),
            ("dedup", "faiss_efC", -100, "must be positive"),
            ("dedup", "k_neighbors", 0, "must be positive"),
            ("dedup", "k_neighbors", -3, "must be positive"),
            # slicer numeric validations
            ("slicer", "max_tokens", 0, "must be positive"),
            ("slicer", "max_tokens", -1000, "must be positive"),
            ("slicer", "overlap", -10, "must be non-negative"),
            ("slicer", "soft_boundary_max_shift", -5, "must be non-negative"),
        ],
    )
    def test_numeric_field_validation(self, section, field, value, error_pattern):
        """Parametrized test for numeric field validation across sections."""
        config = get_config_with_override(section, field, value)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config)
            f.flush()

            with pytest.raises(ConfigValidationError, match=error_pattern):
                load_config(f.name)

        Path(f.name).unlink()

    @pytest.mark.parametrize(
        "section,field,value,error_pattern",
        [
            # Range validations
            ("dedup", "sim_threshold", -0.1, "must be between 0.0 and 1.0"),
            ("dedup", "sim_threshold", 1.1, "must be between 0.0 and 1.0"),
            ("dedup", "len_ratio_min", -0.5, "must be between 0.0 and 1.0"),
            ("dedup", "len_ratio_min", 2.0, "must be between 0.0 and 1.0"),
            ("refiner", "sim_threshold", -0.2, "must be between 0.0 and 1.0"),
            ("refiner", "sim_threshold", 1.5, "must be between 0.0 and 1.0"),
            # Веса удалены из конфига - теперь в промптах
            # ("refiner", "weight_low", -0.1, "must be between 0.0 and 1.0"),
            # ("refiner", "weight_mid", 1.2, "must be between 0.0 and 1.0"),
            # ("refiner", "weight_high", 2.0, "must be between 0.0 and 1.0"),
        ],
    )
    def test_range_validation(self, section, field, value, error_pattern):
        """Parametrized test for range validation."""
        config = get_config_with_override(section, field, value)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config)
            f.flush()

            with pytest.raises(ConfigValidationError, match=error_pattern):
                load_config(f.name)

        Path(f.name).unlink()

    @pytest.mark.parametrize(
        "section,field,value,error_pattern",
        [
            # String enum validations
            ("itext2kg", "log_level", "invalid", "must be one of"),
            ("itext2kg", "log_level", "trace", "must be one of"),
            ("dedup", "faiss_metric", "COSINE", "must be 'INNER_PRODUCT' or 'L2'"),
            ("dedup", "faiss_metric", "EUCLIDEAN", "must be 'INNER_PRODUCT' or 'L2'"),
            ("slicer", "tokenizer", "gpt2", "must be 'o200k_base'"),
            ("slicer", "tokenizer", "cl100k_base", "must be 'o200k_base'"),
        ],
    )
    def test_string_enum_validation(self, section, field, value, error_pattern):
        """Parametrized test for string enum validation."""
        config = get_config_with_override(section, field, value)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config)
            f.flush()

            with pytest.raises(ConfigValidationError, match=error_pattern):
                load_config(f.name)

        Path(f.name).unlink()


class TestExtremeValues:
    """Tests for extreme values and boundary conditions."""

    def test_very_large_max_tokens(self):
        """Test very large max_tokens value."""
        config = get_config_with_override("slicer", "max_tokens", 999999999)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config)
            f.flush()

            # Should not raise - just checking it handles large values
            result = load_config(f.name)
            assert result["slicer"]["max_tokens"] == 999999999

        Path(f.name).unlink()

    def test_max_completion_at_boundary(self):
        """Test max_completion exactly at boundary values."""
        # Test at lower boundary
        config = get_config_with_override("itext2kg", "max_completion", 1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config)
            f.flush()

            result = load_config(f.name)
            assert result["itext2kg"]["max_completion"] == 1

        Path(f.name).unlink()

        # Test at upper boundary
        config = get_config_with_override("itext2kg", "max_completion", 100000)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config)
            f.flush()

            result = load_config(f.name)
            assert result["itext2kg"]["max_completion"] == 100000

        Path(f.name).unlink()

    def test_float_values_where_int_expected(self):
        """Test float values where integers are expected."""
        config = get_config_with_override("slicer", "max_tokens", 40000.5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config)
            f.flush()

            with pytest.raises(ConfigValidationError, match="must be int"):
                load_config(f.name)

        Path(f.name).unlink()

    def test_weights_at_exact_boundaries(self):
        """Test weight values at exact boundaries."""
        config = (
            get_minimal_valid_config()
            .replace("weight_low = 0.3", "weight_low = 0.0")
            .replace("weight_high = 0.9", "weight_high = 1.0")
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config)
            f.flush()

            result = load_config(f.name)
            # Веса больше не в конфиге - перенесены в промпты
            # assert result["refiner"]["weight_low"] == 0.0
            # assert result["refiner"]["weight_high"] == 1.0

        Path(f.name).unlink()

    def test_zero_response_chain_depth(self):
        """Test response_chain_depth = 0 (independent requests mode)."""
        config = get_minimal_valid_config().replace(
            "max_retries = 3", "max_retries = 3\n        response_chain_depth = 0"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config)
            f.flush()

            result = load_config(f.name)
            assert result["itext2kg"]["response_chain_depth"] == 0

        Path(f.name).unlink()

    def test_very_large_tpm_limit(self):
        """Test very large TPM limit value."""
        config = get_config_with_override("itext2kg", "tpm_limit", 10000000)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config)
            f.flush()

            result = load_config(f.name)
            assert result["itext2kg"]["tpm_limit"] == 10000000

        Path(f.name).unlink()


def get_config_with_override(section: str, field: str, value) -> str:
    """Generate test config with single field override.

    Args:
        section: Configuration section name
        field: Field name to override
        value: New value for the field

    Returns:
        Modified configuration string
    """
    base_config = get_minimal_valid_config()

    # Convert value to proper TOML format
    if isinstance(value, str):
        value_str = f'"{value}"'
    elif isinstance(value, bool):
        value_str = "true" if value else "false"
    elif isinstance(value, list):
        value_str = str(value)
    else:
        value_str = str(value)

    # Find and replace the field in the specified section
    lines = base_config.split("\n")
    in_section = False
    for i, line in enumerate(lines):
        if line.strip() == f"[{section}]":
            in_section = True
        elif line.strip().startswith("[") and in_section:
            # We've moved to another section, field not found
            # Insert the field at the end of the previous section
            lines.insert(i, f"        {field} = {value_str}")
            break
        elif in_section and line.strip().startswith(f"{field} ="):
            # Replace existing field
            indent = len(line) - len(line.lstrip())
            lines[i] = " " * indent + f"{field} = {value_str}"
            break
    else:
        # Field not found and we're at the end of the file
        if in_section:
            # Add to the current section
            lines.append(f"        {field} = {value_str}")

    return "\n".join(lines)


def get_minimal_valid_config():
    """Возвращает минимальную валидную конфигурацию для тестов."""
    return textwrap.dedent(
        """
        [slicer]
        log_level = "info"
        max_tokens = 4000
        overlap = 400
        tokenizer = "o200k_base"
        allowed_extensions = ["md", "txt", "json", "html"]
        soft_boundary = true
        soft_boundary_max_shift = 200

        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = 3
        log_level = "info"

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
    )


class TestDedupValidation:
    """Tests for dedup section validation."""

    def test_embedding_model_required(self):
        """Test that embedding_model is required."""
        config_without_embedding_model = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6

        [dedup]
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_without_embedding_model)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="Missing required field: dedup.embedding_model"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_invalid_similarity_threshold(self):
        """Test sim_threshold must be between 0.0 and 1.0."""
        config_with_invalid_threshold = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        tpm_limit = 120000
        max_completion = 4096
        log_level = "info"
        api_key = "sk-test"
        timeout = 45
        max_retries = 6

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 1.5
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_threshold)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="dedup.sim_threshold must be between 0.0 and 1.0"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_invalid_faiss_parameters(self):
        """Test faiss_M and faiss_efC must be positive."""
        config_with_invalid_faiss = textwrap.dedent(
            """
        [slicer]
        max_tokens = 40000
        overlap = 0
        soft_boundary = true
        soft_boundary_max_shift = 500
        tokenizer = "o200k_base"
        allowed_extensions = ["json"]

        [itext2kg]
        is_reasoning = false
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
        faiss_M = -1
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_faiss)
            f.flush()

            with pytest.raises(ConfigValidationError, match="dedup.faiss_M must be positive"):
                load_config(f.name)

        Path(f.name).unlink()

    def test_slicer_negative_overlap(self):
        """Тест валидации отрицательного overlap (покрытие строки 182)."""
        config_with_negative_overlap = textwrap.dedent(
            """
        [slicer]
        log_level = "info"
        max_tokens = 4000
        overlap = -10
        tokenizer = "o200k_base"
        allowed_extensions = ["md", "txt", "json", "html"]
        soft_boundary = true
        soft_boundary_max_shift = 200

        [itext2kg]
        is_reasoning = false
        model = "gpt-4o"
        api_key = "sk-test"
        tpm_limit = 60000
        max_completion = 2048
        timeout = 30
        max_retries = 3

        [dedup]
        embedding_model = "text-embedding-3-small"
        sim_threshold = 0.97
        len_ratio_min = 0.8
        faiss_M = 32
        faiss_efC = 200
        faiss_metric = "INNER_PRODUCT"
        k_neighbors = 5

        [refiner]
        is_reasoning = false
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
        # Веса удалены - теперь в промптах
        # weight_low = 0.3
        # weight_mid = 0.6
        # weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_negative_overlap)
            f.flush()

            with pytest.raises(ConfigValidationError, match="slicer.overlap must be non-negative"):
                load_config(f.name)

        Path(f.name).unlink()

    def test_slicer_negative_soft_boundary_shift(self):
        """Тест валидации отрицательного soft_boundary_max_shift (покрытие строки 185)."""
        config = get_minimal_valid_config()
        # Заменяем значение soft_boundary_max_shift на отрицательное
        config_with_negative_shift = config.replace(
            "soft_boundary_max_shift = 200", "soft_boundary_max_shift = -100"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_negative_shift)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="slicer.soft_boundary_max_shift must be non-negative"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_itext2kg_invalid_temperature(self):
        """Тест валидации некорректной temperature (покрытие строк 198, 202)."""
        # Температура < 0
        config = get_minimal_valid_config()
        # Добавляем temperature с некорректным значением
        config_with_negative_temp = config.replace(
            "max_retries = 3", "max_retries = 3\n        temperature = -0.5"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_negative_temp)
            f.flush()

            with pytest.raises(ConfigValidationError, match="temperature must be between 0 and 2"):
                load_config(f.name)

        Path(f.name).unlink()

        # Температура > 2
        config = get_minimal_valid_config()
        config_with_high_temp = config.replace(
            "max_retries = 3", "max_retries = 3\n        temperature = 2.5"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_high_temp)
            f.flush()

            with pytest.raises(ConfigValidationError, match="temperature must be between 0 and 2"):
                load_config(f.name)

        Path(f.name).unlink()

    def test_dedup_invalid_similarity_threshold(self):
        """Тест валидации некорректного sim_threshold (покрытие строк 221, 224)."""
        # sim_threshold < 0
        config = get_minimal_valid_config()
        config_with_negative_sim = config.replace("sim_threshold = 0.97", "sim_threshold = -0.1")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_negative_sim)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="dedup.sim_threshold must be between 0.0 and 1.0"
            ):
                load_config(f.name)

        Path(f.name).unlink()

        # sim_threshold > 1
        config = get_minimal_valid_config()
        config_with_high_sim = config.replace("sim_threshold = 0.97", "sim_threshold = 1.5")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_high_sim)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="dedup.sim_threshold must be between 0.0 and 1.0"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_dedup_invalid_len_ratio(self):
        """Тест валидации некорректного len_ratio_min (покрытие строк 241, 244)."""
        # len_ratio_min < 0
        config_with_negative_ratio = get_minimal_valid_config().replace(
            "len_ratio_min = 0.8", "len_ratio_min = -0.1"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_negative_ratio)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="dedup.len_ratio_min must be between 0.0 and 1.0"
            ):
                load_config(f.name)

        Path(f.name).unlink()

        # len_ratio_min > 1
        config_with_high_ratio = get_minimal_valid_config().replace(
            "len_ratio_min = 0.8", "len_ratio_min = 1.1"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_high_ratio)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="dedup.len_ratio_min must be between 0.0 and 1.0"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_refiner_invalid_weight_boundaries(self):
        """Тест валидации некорректных weight границ - ОТКЛЮЧЕН, веса удалены из конфига."""
        pass
        return
        # weight_low отрицательный
        config_invalid_weights = get_minimal_valid_config().replace(
            "weight_low = 0.3", "weight_low = -0.1"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_invalid_weights)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="refiner.weight_low must be between 0.0 and 1.0"
            ):
                load_config(f.name)

        Path(f.name).unlink()

        # weight_mid > 1
        config_invalid_mid = get_minimal_valid_config().replace(
            "weight_mid = 0.6", "weight_mid = 1.5"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_invalid_mid)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="refiner.weight_mid must be between 0.0 and 1.0"
            ):
                load_config(f.name)

        Path(f.name).unlink()

        # weight_high < 0
        config_invalid_high = get_minimal_valid_config().replace(
            "weight_high = 0.9", "weight_high = -0.1"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_invalid_high)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="refiner.weight_high must be between 0.0 and 1.0"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_negative_faiss_efc(self):
        """Test that faiss_efC must be positive."""
        config_with_negative_efc = get_minimal_valid_config().replace(
            "faiss_efC = 200", "faiss_efC = -10"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_negative_efc)
            f.flush()

            with pytest.raises(ConfigValidationError, match="dedup.faiss_efC must be positive"):
                load_config(f.name)

        Path(f.name).unlink()

    def test_zero_k_neighbors(self):
        """Test that k_neighbors must be positive."""
        config_with_zero_k = get_minimal_valid_config().replace(
            "k_neighbors = 5", "k_neighbors = 0"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_zero_k)
            f.flush()

            with pytest.raises(ConfigValidationError, match="dedup.k_neighbors must be positive"):
                load_config(f.name)

        Path(f.name).unlink()

    def test_invalid_faiss_metric(self):
        """Test that faiss_metric must be valid."""
        config_with_invalid_metric = get_minimal_valid_config().replace(
            'faiss_metric = "INNER_PRODUCT"', 'faiss_metric = "COSINE"'
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_metric)
            f.flush()

            with pytest.raises(
                ConfigValidationError, match="dedup.faiss_metric must be 'INNER_PRODUCT' or 'L2'"
            ):
                load_config(f.name)

        Path(f.name).unlink()

    def test_invalid_weight_order(self):
        """Тест валидации порядка weight - ОТКЛЮЧЕН, веса удалены из конфига."""
        pass
        return
        # Make weight_low > weight_mid to violate the order
        config_invalid_order = (
            get_minimal_valid_config()
            .replace("weight_low = 0.3", "weight_low = 0.7")
            .replace("weight_mid = 0.6", "weight_mid = 0.5")
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_invalid_order)
            f.flush()

            with pytest.raises(
                ConfigValidationError,
                match="refiner weights must satisfy: weight_low < weight_mid < weight_high",
            ):
                load_config(f.name)

        Path(f.name).unlink()
