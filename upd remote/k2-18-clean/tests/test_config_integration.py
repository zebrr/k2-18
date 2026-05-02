"""
Интеграционные тесты для модуля конфигурации с реальным config.toml файлом.
"""

from pathlib import Path

from src.utils.config import load_config


class TestConfigIntegration:
    """Интеграционные тесты с реальным файлом конфигурации."""

    def test_load_real_config(self):
        """Тест загрузки реального config.toml из проекта."""
        # Загружаем реальную конфигурацию (без указания пути)
        config = load_config()

        # Проверяем, что все обязательные секции присутствуют
        required_sections = ["slicer", "itext2kg_concepts", "itext2kg_graph", "dedup", "refiner"]
        for section in required_sections:
            assert section in config, f"Missing section: {section}"

        # Проверяем ключевые параметры slicer
        slicer = config["slicer"]
        assert isinstance(slicer["max_tokens"], int)
        assert slicer["max_tokens"] > 0
        assert isinstance(slicer["soft_boundary"], bool)
        assert slicer["tokenizer"] == "o200k_base"
        assert isinstance(slicer["allowed_extensions"], list)
        assert len(slicer["allowed_extensions"]) > 0

        # Проверяем ключевые параметры itext2kg_concepts
        itext2kg_concepts = config["itext2kg_concepts"]
        assert isinstance(itext2kg_concepts["model"], str)
        assert len(itext2kg_concepts["model"]) > 0
        assert isinstance(itext2kg_concepts["tpm_limit"], int)
        assert itext2kg_concepts["tpm_limit"] > 0
        assert 1 <= itext2kg_concepts["max_completion"] <= 100000
        assert itext2kg_concepts["log_level"] in ["debug", "info", "warning", "error"]
        assert isinstance(itext2kg_concepts["api_key"], str)
        assert len(itext2kg_concepts["api_key"].strip()) > 0

        # Проверяем ключевые параметры itext2kg_graph
        itext2kg_graph = config["itext2kg_graph"]
        assert isinstance(itext2kg_graph["model"], str)
        assert len(itext2kg_graph["model"]) > 0
        assert isinstance(itext2kg_graph["tpm_limit"], int)
        assert itext2kg_graph["tpm_limit"] > 0
        assert 1 <= itext2kg_graph["max_completion"] <= 100000
        assert itext2kg_graph["log_level"] in ["debug", "info", "warning", "error"]
        assert isinstance(itext2kg_graph["api_key"], str)
        assert len(itext2kg_graph["api_key"].strip()) > 0

        # Проверяем ключевые параметры dedup
        dedup = config["dedup"]
        assert isinstance(dedup["embedding_model"], str)
        assert 0.0 <= dedup["sim_threshold"] <= 1.0  # Исправлено название поля
        assert 0.0 <= dedup["len_ratio_min"] <= 1.0
        assert dedup["faiss_M"] > 0
        assert dedup["faiss_efC"] > 0
        assert dedup["k_neighbors"] > 0

        # Проверяем ключевые параметры refiner
        refiner = config["refiner"]
        assert isinstance(refiner["run"], bool)
        assert isinstance(refiner["embedding_model"], str)
        assert 0.0 <= refiner["sim_threshold"] <= 1.0
        assert refiner["max_pairs_per_node"] > 0
        assert isinstance(refiner["model"], str)  # Исправлено название поля
        assert isinstance(refiner["api_key"], str)
        assert len(refiner["api_key"].strip()) > 0
        assert refiner["tpm_limit"] > 0
        assert 1 <= refiner["max_completion"] <= 100000  # Обновлено согласно валидации

        # Веса больше не в конфиге - перенесены в промпты
        # assert refiner["weight_low"] < refiner["weight_mid"] < refiner["weight_high"]
        # assert all(
        #     0.0 <= w <= 1.0
        #     for w in [
        #         refiner["weight_low"],
        #         refiner["weight_mid"],
        #         refiner["weight_high"],
        #     ]
        # )

    def test_config_file_exists(self):
        """Тест проверки существования файла config.toml."""
        # Проверяем, что файл config.toml существует в ожидаемом месте
        config_path = Path(__file__).parent.parent / "src" / "config.toml"
        assert config_path.exists(), f"config.toml not found at {config_path}"
        assert config_path.is_file(), f"{config_path} is not a file"

    def test_config_accessibility_from_utils(self):
        """Тест доступности конфигурации из модулей utils."""
        # Этот тест имитирует как другие модули будут импортировать конфигурацию
        from src.utils.config import load_config

        config = load_config()

        # Проверяем, что конфигурация доступна и содержит нужные данные
        assert config is not None
        assert isinstance(config, dict)
        assert len(config) >= 4  # Минимум 4 секции

        # Проверяем, что можно получить доступ к конкретным параметрам
        # которые будут использоваться в других модулях
        tokenizer_name = config["slicer"]["tokenizer"]
        assert tokenizer_name == "o200k_base"

        model_name = config["itext2kg_concepts"]["model"]
        assert isinstance(model_name, str)
        assert len(model_name) > 0
