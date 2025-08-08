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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
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
            assert config["refiner"]["weight_low"] == 0.3

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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_overlap)
            f.flush()

            with pytest.raises(ConfigValidationError, match="cannot exceed overlap\\*0.8"):
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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
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


class TestRefinerValidation:
    """Тесты валидации секции [refiner]."""

    def test_invalid_weight_order(self):
        """Тест валидации порядка весов."""
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
        """Тест валидации весов вне диапазона [0,1]."""
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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
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
        weight_low = 0.3
        weight_mid = 0.6
        weight_high = 0.9
        """
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_with_invalid_faiss)
            f.flush()

            with pytest.raises(ConfigValidationError, match="dedup.faiss_M must be positive"):
                load_config(f.name)

        Path(f.name).unlink()
