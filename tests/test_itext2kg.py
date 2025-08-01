"""Тесты для модуля itext2kg."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.itext2kg import ProcessingStats, SliceData, SliceProcessor
from src.utils.exit_codes import EXIT_INPUT_ERROR, EXIT_SUCCESS
from src.utils.llm_client import ResponseUsage


@pytest.fixture
def mock_config():
    """Тестовая конфигурация."""
    return {
        "itext2kg": {
            "model": "gpt-4o",
            "api_key": "test-key",
            "tpm_limit": 120000,
            "tpm_safety_margin": 0.15,
            "max_completion": 4096,
            "log_level": "info",
            "temperature": 1.0,
            "timeout": 45,
            "max_retries": 3,
        }
    }


@pytest.fixture
def mock_schemas(tmp_path):
    """Создание тестовых схем."""
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()

    # Минимальные валидные схемы
    concept_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"concepts": {"type": "array"}},
    }

    graph_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"nodes": {"type": "array"}, "edges": {"type": "array"}},
    }

    (schemas_dir / "ConceptDictionary.schema.json").write_text(
        json.dumps(concept_schema)
    )
    (schemas_dir / "LearningChunkGraph.schema.json").write_text(
        json.dumps(graph_schema)
    )

    return schemas_dir


@pytest.fixture
def mock_prompt(tmp_path):
    """Создание тестового промпта."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    prompt_content = """Test prompt
    {concept_dictionary_schema}
    {learning_chunk_graph_schema}
    """

    (prompts_dir / "itext2kg_extraction.md").write_text(prompt_content)
    return prompts_dir


@pytest.fixture
def processor(mock_config, mock_schemas, mock_prompt, monkeypatch):
    """Инициализированный процессор для тестов."""
    monkeypatch.setattr("src.itext2kg.SCHEMAS_DIR", mock_schemas)
    monkeypatch.setattr("src.itext2kg.PROMPTS_DIR", mock_prompt)

    with patch("src.itext2kg.OpenAIClient"):
        return SliceProcessor(mock_config)


@pytest.fixture
def sample_slice_file(tmp_path):
    """Создание тестового файла слайса."""
    slice_data = {
        "id": "slice_001",
        "order": 1,
        "source_file": "test.txt",
        "slug": "test",
        "text": "Переменная в Python создается присваиванием: x = 5",
        "slice_token_start": 0,
        "slice_token_end": 100,
    }

    slice_file = tmp_path / "slice_001.slice.json"
    slice_file.write_text(json.dumps(slice_data, ensure_ascii=False), encoding="utf-8")

    return slice_file


@pytest.fixture
def sample_llm_response():
    """Пример корректного ответа LLM."""
    return json.dumps(
        {
            "concepts_added": {
                "concepts": [
                    {
                        "concept_id": "test:p:variable",
                        "term": {
                            "primary": "Переменная",
                            "aliases": ["variable", "var"],
                        },
                        "definition": "Именованная область памяти",
                    }
                ]
            },
            "chunk_graph_patch": {
                "nodes": [
                    {
                        "id": "test:c:0",
                        "type": "Chunk",
                        "text": "Переменная в Python создается присваиванием: x = 5",
                        "local_start": 0,
                        "difficulty": 1,
                    }
                ],
                "edges": [
                    {
                        "source": "test:c:0",
                        "target": "test:p:variable",
                        "type": "PREREQUISITE",  # Изменено с MENTIONS на PREREQUISITE
                        "weight": 0.9,
                    }
                ],
            },
        }
    )


class TestSliceProcessor:
    """Тесты для SliceProcessor."""

    def test_init(self, mock_config, mock_schemas, mock_prompt, monkeypatch):
        """Тест инициализации процессора."""
        # Патчим пути
        monkeypatch.setattr("src.itext2kg.SCHEMAS_DIR", mock_schemas)
        monkeypatch.setattr("src.itext2kg.PROMPTS_DIR", mock_prompt)

        # Патчим OpenAIClient
        with patch("src.itext2kg.OpenAIClient") as mock_client:
            processor = SliceProcessor(mock_config)

            assert processor.config == mock_config["itext2kg"]
            assert processor.concept_dictionary == {"concepts": []}
            assert processor.learning_graph == {"nodes": [], "edges": []}
            assert len(processor.known_node_ids) == 0
            assert isinstance(processor.stats, ProcessingStats)

            # Проверяем, что промпт загружен и схемы подставлены
            assert "Test prompt" in processor.extraction_prompt
            assert "json-schema.org" in processor.extraction_prompt

    def test_load_extraction_prompt_missing_file(self, mock_config, monkeypatch):
        """Тест ошибки при отсутствии файла промпта."""
        monkeypatch.setattr("src.itext2kg.PROMPTS_DIR", Path("/nonexistent"))

        with patch("src.itext2kg.OpenAIClient"):
            with pytest.raises(FileNotFoundError):
                SliceProcessor(mock_config)

    def test_run_no_slices(
        self, mock_config, mock_schemas, mock_prompt, tmp_path, monkeypatch
    ):
        """Тест запуска без слайсов."""
        # Патчим пути
        monkeypatch.setattr("src.itext2kg.SCHEMAS_DIR", mock_schemas)
        monkeypatch.setattr("src.itext2kg.PROMPTS_DIR", mock_prompt)
        monkeypatch.setattr("src.itext2kg.STAGING_DIR", tmp_path / "staging")

        # Создаем пустую директорию staging
        (tmp_path / "staging").mkdir()

        with patch("src.itext2kg.OpenAIClient"):
            processor = SliceProcessor(mock_config)
            result = processor.run()

            assert result == EXIT_INPUT_ERROR

    def test_print_start_status(
        self, mock_config, mock_schemas, mock_prompt, monkeypatch, capsys
    ):
        """Тест вывода начального статуса."""
        monkeypatch.setattr("src.itext2kg.SCHEMAS_DIR", mock_schemas)
        monkeypatch.setattr("src.itext2kg.PROMPTS_DIR", mock_prompt)

        with patch("src.itext2kg.OpenAIClient"):
            processor = SliceProcessor(mock_config)
            processor.stats.total_slices = 42

            processor._print_start_status()

            captured = capsys.readouterr()
            assert "START" in captured.out
            assert "42 slices" in captured.out
            assert "model=gpt-4o" in captured.out
            assert "tpm=120k" in captured.out

    def test_format_slice_input(self, processor):
        """Тест форматирования входных данных для LLM."""
        # Добавляем тестовые концепты
        processor.concept_dictionary["concepts"] = [
            {
                "concept_id": "test:p:variable",
                "term": {"primary": "Переменная", "aliases": ["variable"]},
                "definition": "Именованная область памяти",
            }
        ]

        # Создаем тестовый слайс
        slice_data = SliceData(
            id="slice_001",
            order=1,
            source_file="test.txt",
            slug="test",
            text="Тестовый текст",
            slice_token_start=0,
            slice_token_end=100,
        )

        result = processor._format_slice_input(slice_data)
        parsed = json.loads(result)

        assert "ConceptDictionary" in parsed
        assert "Slice" in parsed
        assert parsed["Slice"]["id"] == "slice_001"
        assert len(parsed["ConceptDictionary"]["concepts"]) == 1

    def test_update_concept_dictionary_new_concept(self, processor):
        """Тест добавления нового концепта."""
        new_concepts = [
            {
                "concept_id": "test:p:function",
                "term": {"primary": "Функция", "aliases": ["function"]},
                "definition": "Блок кода",
            }
        ]

        processor._update_concept_dictionary(new_concepts)

        assert len(processor.concept_dictionary["concepts"]) == 1
        assert (
            processor.concept_dictionary["concepts"][0]["concept_id"]
            == "test:p:function"
        )
        assert "test:p:function" in processor.concept_id_map
        assert processor.stats.total_concepts == 1

    def test_update_concept_dictionary_update_aliases(self, processor):
        """Тест обновления aliases существующего концепта."""
        # Добавляем начальный концепт
        processor.concept_dictionary["concepts"] = [
            {
                "concept_id": "test:p:variable",
                "term": {"primary": "Переменная", "aliases": ["var"]},
                "definition": "Область памяти",
            }
        ]
        processor.concept_id_map["test:p:variable"] = 0

        # Обновляем с новыми aliases
        update_concepts = [
            {
                "concept_id": "test:p:variable",
                "term": {
                    "primary": "Переменная",
                    "aliases": ["var", "variable", "переменная"],
                },
                "definition": "Новое определение (игнорируется)",
            }
        ]

        processor._update_concept_dictionary(update_concepts)

        # Проверяем, что aliases обновились, но definition остался прежним
        assert len(processor.concept_dictionary["concepts"]) == 1
        concept = processor.concept_dictionary["concepts"][0]
        assert set(concept["term"]["aliases"]) == {"var", "variable", "переменная"}
        assert concept["definition"] == "Область памяти"

    def test_process_chunk_nodes_new_node(self, processor):
        """Тест добавления новых узлов."""
        new_nodes = [
            {
                "id": "test:c:100",
                "type": "Chunk",
                "text": "Тестовый чанк",
                "local_start": 0,
            },
            {
                "id": "test:p:concept",
                "type": "Concept",
                "text": "Концепт",
                "definition": "Определение",
                "local_start": 0,  # ← ДОБАВЛЕНО
            },
        ]

        result = processor._process_chunk_nodes(new_nodes)

        assert len(result) == 2
        assert "test:c:100" in processor.known_node_ids
        assert "test:p:concept" in processor.known_node_ids

    def test_process_chunk_nodes_update_existing(self, processor):
        """Тест обновления существующего Chunk узла."""
        # Добавляем существующий узел
        processor.learning_graph["nodes"] = [
            {
                "id": "test:c:100",
                "type": "Chunk",
                "text": "Короткий текст",
                "local_start": 0,
            }
        ]
        processor.known_node_ids.add("test:c:100")

        # Пытаемся добавить более длинную версию
        new_nodes = [
            {
                "id": "test:c:100",
                "type": "Chunk",
                "text": "Это более длинный текст того же чанка",
                "local_start": 0,
            }
        ]

        result = processor._process_chunk_nodes(new_nodes)

        assert len(result) == 0  # Не добавляем, а обновляем
        assert (
            processor.learning_graph["nodes"][0]["text"]
            == "Это более длинный текст того же чанка"
        )

    def test_validate_edges_valid(self, processor):
        """Тест валидации корректных рёбер."""
        # Добавляем известные узлы
        processor.known_node_ids.update(["chunk1", "chunk2"])
        processor.concept_id_map["concept1"] = 0

        edges = [
            {
                "source": "chunk1",
                "target": "chunk2",
                "type": "PREREQUISITE",
                "weight": 0.8,
            },
            {
                "source": "chunk1",
                "target": "concept1",
                "type": "MENTIONS",
                "weight": 0.5,
            },
        ]

        result = processor._validate_edges(edges)
        assert len(result) == 2

    def test_validate_edges_invalid_reference(self, processor):
        """Тест отбрасывания рёбер с несуществующими узлами."""
        processor.known_node_ids.add("chunk1")

        edges = [
            {
                "source": "chunk1",
                "target": "unknown",
                "type": "PREREQUISITE",
                "weight": 0.8,
            }
        ]

        result = processor._validate_edges(edges)
        assert len(result) == 0

    def test_validate_edges_self_loop(self, processor):
        """Тест отбрасывания PREREQUISITE self-loops."""
        processor.known_node_ids.add("chunk1")

        edges = [
            {
                "source": "chunk1",
                "target": "chunk1",
                "type": "PREREQUISITE",
                "weight": 0.8,
            },
            {
                "source": "chunk1",
                "target": "chunk1",
                "type": "PARALLEL",
                "weight": 0.8,
            },  # OK
        ]

        result = processor._validate_edges(edges)
        assert len(result) == 1
        assert result[0]["type"] == "PARALLEL"

    def test_process_llm_response_valid(self, processor):
        """Тест обработки валидного ответа LLM."""
        response_text = json.dumps(
            {
                "concepts_added": {
                    "concepts": [
                        {
                            "concept_id": "test:p:var",
                            "term": {"primary": "Переменная"},
                            "definition": "Тест",
                        }
                    ]
                },
                "chunk_graph_patch": {
                    "nodes": [
                        {
                            "id": "test:c:100",
                            "type": "Chunk",
                            "text": "Текст",
                            "local_start": 0,  # ← ДОБАВЛЕНО
                        }
                    ],
                    "edges": [],
                },
            }
        )

        success, data = processor._process_llm_response(response_text, "slice_001")

        assert success is True
        assert data is not None
        assert len(data["concepts_added"]["concepts"]) == 1

    def test_process_llm_response_invalid_json(self, processor):
        """Тест обработки некорректного JSON."""
        response_text = "This is not JSON"

        success, data = processor._process_llm_response(response_text, "slice_001")

        assert success is False
        assert data is None

    def test_process_llm_response_missing_fields(self, processor):
        """Тест обработки ответа с отсутствующими полями."""
        response_text = json.dumps(
            {
                "concepts_added": {"concepts": []}
                # chunk_graph_patch отсутствует
            }
        )

        success, data = processor._process_llm_response(response_text, "slice_001")

        assert success is False
        assert data is None

    def test_load_slice_valid(self, processor, sample_slice_file):
        """Тест загрузки валидного слайса."""
        slice_data = processor._load_slice(sample_slice_file)

        assert slice_data.id == "slice_001"
        assert slice_data.order == 1
        assert slice_data.slug == "test"
        assert "Переменная" in slice_data.text

    def test_load_slice_invalid_json(self, processor, tmp_path):
        """Тест загрузки невалидного JSON."""
        bad_file = tmp_path / "bad.slice.json"
        bad_file.write_text("not json")

        with pytest.raises(ValueError):
            processor._load_slice(bad_file)

    def test_apply_patch(self, processor):
        """Тест применения патча к графу."""
        patch_data = {
            "concepts_added": {
                "concepts": [
                    {
                        "concept_id": "test:p:new",
                        "term": {"primary": "Новый концепт"},
                        "definition": "Определение",
                    }
                ]
            },
            "chunk_graph_patch": {
                "nodes": [
                    {
                        "id": "test:c:200",
                        "type": "Chunk",
                        "text": "Новый чанк",
                        "local_start": 0,
                    }
                ],
                "edges": [
                    {
                        "source": "test:c:200",
                        "target": "test:p:new",
                        "type": "MENTIONS",
                        "weight": 0.8,
                    }
                ],
            },
        }

        nodes_added, edges_added = processor._apply_patch(patch_data)

        assert nodes_added == 2  # 1 Chunk + 1 Concept
        assert (
            edges_added >= 1
        )  # Минимум 1 edge из патча, возможны автоматические MENTIONS
        assert len(processor.concept_dictionary["concepts"]) == 1
        assert processor.stats.total_nodes == 2
        # edges может быть больше из-за автоматических MENTIONS
        assert processor.stats.total_edges >= 1

    def test_save_bad_response(self, processor, tmp_path, monkeypatch):
        """Тест сохранения некорректного ответа."""
        monkeypatch.setattr("src.itext2kg.LOGS_DIR", tmp_path)

        processor._save_bad_response(
            "slice_001", "bad response", "Invalid JSON", "repair response"
        )

        bad_file = tmp_path / "slice_001_bad.json"
        assert bad_file.exists()

        saved_data = json.loads(bad_file.read_text())
        assert saved_data["slice_id"] == "slice_001"
        assert saved_data["original_response"] == "bad response"
        assert saved_data["validation_error"] == "Invalid JSON"
        assert saved_data["repair_response"] == "repair response"

    def test_process_single_slice_success(
        self, processor, sample_slice_file, sample_llm_response, monkeypatch
    ):
        """Тест успешной обработки слайса."""
        # Мокаем LLM клиент
        mock_llm = Mock()
        mock_llm.create_response.return_value = (
            sample_llm_response,
            "response_123",
            ResponseUsage(100, 50, 150, 0),
        )
        processor.llm_client = mock_llm

        # Обрабатываем слайс
        result = processor._process_single_slice(sample_slice_file)

        assert result is True
        assert processor.stats.total_tokens_used == 150
        assert len(processor.concept_dictionary["concepts"]) == 1
        assert len(processor.learning_graph["nodes"]) == 2  # 1 Chunk + 1 Concept

        # Проверяем edges - должен быть минимум 1 из патча
        assert len(processor.learning_graph["edges"]) >= 1

        # Если есть упоминание концепта в тексте чанка, должен быть и MENTIONS edge
        edges_types = {e["type"] for e in processor.learning_graph["edges"]}
        # В sample_llm_response уже есть MENTIONS edge, но может добавиться автоматический

    def test_process_single_slice_with_repair(
        self, processor, sample_slice_file, sample_llm_response, monkeypatch
    ):
        """Тест обработки слайса с repair."""
        # Мокаем LLM клиент
        mock_llm = Mock()
        # Первый вызов возвращает невалидный JSON
        mock_llm.create_response.return_value = (
            "invalid json",
            "response_123",
            ResponseUsage(100, 50, 150, 0),
        )
        # Repair возвращает валидный ответ
        mock_llm.repair_response.return_value = (
            sample_llm_response,
            "response_124",
            ResponseUsage(100, 60, 160, 0),
        )
        processor.llm_client = mock_llm

        # Обрабатываем слайс
        result = processor._process_single_slice(sample_slice_file)

        assert result is True
        assert mock_llm.repair_response.called
        assert processor.stats.total_tokens_used == 160  # Используется repair usage

    def test_process_single_slice_failed_repair(
        self, processor, sample_slice_file, tmp_path, monkeypatch
    ):
        """Тест неудачного repair."""
        monkeypatch.setattr("src.itext2kg.LOGS_DIR", tmp_path)

        # Мокаем LLM клиент
        mock_llm = Mock()
        mock_llm.create_response.return_value = (
            "invalid json",
            "response_123",
            ResponseUsage(100, 50, 150, 0),
        )
        mock_llm.repair_response.return_value = (
            "still invalid",
            "response_124",
            ResponseUsage(100, 60, 160, 0),
        )
        processor.llm_client = mock_llm

        # Обрабатываем слайс
        result = processor._process_single_slice(sample_slice_file)

        assert result is False
        # Проверяем, что сохранился bad response
        bad_file = tmp_path / "slice_001_bad.json"
        assert bad_file.exists()

    def test_process_single_slice_api_error(self, processor, sample_slice_file):
        """Тест обработки API ошибки."""
        # Мокаем LLM клиент
        mock_llm = Mock()
        mock_llm.create_response.side_effect = Exception("API Error")
        processor.llm_client = mock_llm

        # Обрабатываем слайс
        result = processor._process_single_slice(sample_slice_file)

        assert result is False

    def test_full_run_success(
        self,
        mock_config,
        mock_schemas,
        mock_prompt,
        tmp_path,
        monkeypatch,
        sample_llm_response,
    ):
        """Тест полного успешного прогона."""
        # Патчим пути
        monkeypatch.setattr("src.itext2kg.SCHEMAS_DIR", mock_schemas)
        monkeypatch.setattr("src.itext2kg.PROMPTS_DIR", mock_prompt)
        monkeypatch.setattr("src.itext2kg.STAGING_DIR", tmp_path / "staging")
        monkeypatch.setattr("src.itext2kg.OUTPUT_DIR", tmp_path / "out")
        monkeypatch.setattr("src.itext2kg.LOGS_DIR", tmp_path / "logs")

        # Создаем директории
        (tmp_path / "staging").mkdir()
        (tmp_path / "out").mkdir()
        (tmp_path / "logs").mkdir()

        # Создаем тестовый слайс
        slice_data = {
            "id": "slice_001",
            "order": 1,
            "source_file": "test.txt",
            "slug": "test",
            "text": "Test content",
            "slice_token_start": 0,
            "slice_token_end": 100,
        }
        slice_file = tmp_path / "staging" / "slice_001.slice.json"
        slice_file.write_text(json.dumps(slice_data))

        # Мокаем OpenAIClient
        with patch("src.itext2kg.OpenAIClient") as mock_client_class:
            mock_instance = Mock()
            mock_instance.create_response.return_value = (
                sample_llm_response,
                "response_123",
                ResponseUsage(100, 50, 150, 0),
            )
            mock_client_class.return_value = mock_instance

            processor = SliceProcessor(mock_config)
            result = processor.run()

            assert result == EXIT_SUCCESS

            # Проверяем, что файлы созданы
            assert (tmp_path / "out" / "ConceptDictionary.json").exists()
            assert (tmp_path / "out" / "LearningChunkGraph_raw.json").exists()

            # Проверяем содержимое сохраненных файлов
            graph_data = json.loads(
                (tmp_path / "out" / "LearningChunkGraph_raw.json").read_text(
                    encoding="utf-8"
                )
            )

            # Должны быть узлы и рёбра
            assert len(graph_data["nodes"]) >= 1
            assert len(graph_data["edges"]) >= 1  # Минимум из патча

    def test_add_mentions_edges_basic(self, processor):
        """Тест базового добавления MENTIONS edges."""
        # Добавляем концепты в словарь
        processor.concept_dictionary = {
            "concepts": [
                {
                    "concept_id": "test:p:stack",
                    "term": {"primary": "Стек", "aliases": ["stack", "LIFO"]},
                    "definition": "LIFO структура данных",
                }
            ]
        }
        processor.concept_id_map = {"test:p:stack": 0}

        # Создаем чанки для тестирования
        chunk_nodes = [
            {
                "id": "test:c:100",
                "type": "Chunk",
                "text": "Используем стек для хранения данных. Stack - это LIFO структура.",
                "local_start": 100,
            },
            {
                "id": "test:c:200",
                "type": "Chunk",
                "text": "Другой текст без упоминаний",
                "local_start": 200,
            },
        ]

        # Запускаем добавление MENTIONS
        edges_added = processor._add_mentions_edges(chunk_nodes)

        # Проверяем результат
        assert edges_added == 1  # Только первый чанк содержит упоминания

        mentions_edges = [
            e for e in processor.learning_graph["edges"] if e["type"] == "MENTIONS"
        ]
        assert len(mentions_edges) == 1

        edge = mentions_edges[0]
        assert edge["source"] == "test:c:100"
        assert edge["target"] == "test:p:stack"
        assert edge["weight"] == 1.0

    def test_add_mentions_edges_case_insensitive(self, processor):
        """Тест case-insensitive поиска."""
        processor.concept_dictionary = {
            "concepts": [
                {
                    "concept_id": "test:p:function",
                    "term": {"primary": "Функция", "aliases": ["function"]},
                    "definition": "Блок кода",
                }
            ]
        }
        processor.concept_id_map = {"test:p:function": 0}

        chunk_nodes = [
            {
                "id": "test:c:1",
                "type": "Chunk",
                "text": "ФУНКЦИЯ обрабатывает данные",  # Uppercase
                "local_start": 0,
            },
            {
                "id": "test:c:2",
                "type": "Chunk",
                "text": "Function processes data",  # Mixed case
                "local_start": 100,
            },
            {
                "id": "test:c:3",
                "type": "Chunk",
                "text": "функция возвращает результат",  # Lowercase
                "local_start": 200,
            },
        ]

        edges_added = processor._add_mentions_edges(chunk_nodes)

        assert edges_added == 3  # Все три чанка содержат упоминания
        mentions_edges = [
            e for e in processor.learning_graph["edges"] if e["type"] == "MENTIONS"
        ]
        assert len(mentions_edges) == 3

        # Проверяем что все чанки связаны с концептом
        sources = {e["source"] for e in mentions_edges}
        assert sources == {"test:c:1", "test:c:2", "test:c:3"}

    def test_add_mentions_edges_full_word_only(self, processor):
        """Тест full word match (не подстроки)."""
        processor.concept_dictionary = {
            "concepts": [
                {
                    "concept_id": "test:p:stack",
                    "term": {"primary": "стек", "aliases": []},
                    "definition": "LIFO структура",
                }
            ]
        }
        processor.concept_id_map = {"test:p:stack": 0}

        chunk_nodes = [
            {
                "id": "test:c:1",
                "type": "Chunk",
                "text": "Используем стек для данных",  # Полное слово - MATCH
                "local_start": 0,
            },
            {
                "id": "test:c:2",
                "type": "Chunk",
                "text": "Стековая память важна",  # Подстрока - NO MATCH
                "local_start": 100,
            },
            {
                "id": "test:c:3",
                "type": "Chunk",
                "text": "Стек, очередь и дерево",  # С пунктуацией - MATCH
                "local_start": 200,
            },
            {
                "id": "test:c:4",
                "type": "Chunk",
                "text": "настек не является словом",  # Часть слова - NO MATCH
                "local_start": 300,
            },
        ]

        edges_added = processor._add_mentions_edges(chunk_nodes)

        assert edges_added == 2  # Только полные слова
        mentions_edges = [
            e for e in processor.learning_graph["edges"] if e["type"] == "MENTIONS"
        ]

        sources = {e["source"] for e in mentions_edges}
        assert sources == {"test:c:1", "test:c:3"}

    def test_add_mentions_edges_no_duplicates(self, processor):
        """Тест что дубликаты MENTIONS edges не создаются."""
        processor.concept_dictionary = {
            "concepts": [
                {
                    "concept_id": "test:p:array",
                    "term": {"primary": "массив", "aliases": ["array"]},
                    "definition": "Структура данных",
                }
            ]
        }
        processor.concept_id_map = {"test:p:array": 0}

        # Добавляем существующий MENTIONS edge
        processor.learning_graph["edges"] = [
            {
                "source": "test:c:100",
                "target": "test:p:array",
                "type": "MENTIONS",
                "weight": 1.0,
            }
        ]

        chunk_nodes = [
            {
                "id": "test:c:100",
                "type": "Chunk",
                "text": "Массив содержит элементы. Array is useful.",
                "local_start": 100,
            }
        ]

        edges_added = processor._add_mentions_edges(chunk_nodes)

        assert edges_added == 0  # Не добавлено, так как уже существует
        mentions_edges = [
            e for e in processor.learning_graph["edges"] if e["type"] == "MENTIONS"
        ]
        assert len(mentions_edges) == 1  # Только оригинальный edge

    def test_add_mentions_edges_multiple_terms(self, processor):
        """Тест поиска по primary и всем aliases."""
        processor.concept_dictionary = {
            "concepts": [
                {
                    "concept_id": "test:p:tree",
                    "term": {
                        "primary": "дерево",
                        "aliases": ["tree", "древовидная структура", "Tree"],
                    },
                    "definition": "Иерархическая структура данных",
                }
            ]
        }
        processor.concept_id_map = {"test:p:tree": 0}

        chunk_nodes = [
            {
                "id": "test:c:1",
                "type": "Chunk",
                "text": "Используем дерево для иерархии",  # primary
                "local_start": 0,
            },
            {
                "id": "test:c:2",
                "type": "Chunk",
                "text": "Binary tree is efficient",  # alias
                "local_start": 100,
            },
            {
                "id": "test:c:3",
                "type": "Chunk",
                "text": "Древовидная структура подходит",  # alias phrase
                "local_start": 200,
            },
        ]

        edges_added = processor._add_mentions_edges(chunk_nodes)

        assert edges_added == 3
        mentions_edges = [
            e for e in processor.learning_graph["edges"] if e["type"] == "MENTIONS"
        ]

        sources = {e["source"] for e in mentions_edges}
        assert sources == {"test:c:1", "test:c:2", "test:c:3"}

        # Все указывают на один концепт
        targets = {e["target"] for e in mentions_edges}
        assert targets == {"test:p:tree"}


class TestProcessingStats:
    """Тесты для ProcessingStats."""

    def test_init(self):
        """Тест инициализации статистики."""
        stats = ProcessingStats()

        assert stats.total_slices == 0
        assert stats.processed_slices == 0
        assert stats.failed_slices == 0
        assert stats.total_concepts == 0
        assert stats.total_nodes == 0
        assert stats.total_edges == 0
        assert stats.total_tokens_used == 0
        assert stats.start_time is not None


class TestSliceData:
    """Тесты для SliceData."""

    def test_creation(self):
        """Тест создания SliceData."""
        slice_data = SliceData(
            id="slice_001",
            order=1,
            source_file="test.txt",
            slug="test",
            text="Тестовый текст",
            slice_token_start=0,
            slice_token_end=100,
        )

        assert slice_data.id == "slice_001"
        assert slice_data.order == 1
        assert slice_data.source_file == "test.txt"
        assert slice_data.slug == "test"
        assert slice_data.text == "Тестовый текст"
        assert slice_data.slice_token_start == 0
        assert slice_data.slice_token_end == 100
