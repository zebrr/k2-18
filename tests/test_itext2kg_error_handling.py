"""Тесты для обработки ошибок и восстановления в модуле itext2kg."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import json
import time
from datetime import datetime, timezone

from src.itext2kg import SliceProcessor, ProcessingStats, SliceData
from src.utils.exit_codes import (
    EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR, EXIT_API_LIMIT_ERROR, EXIT_IO_ERROR
)
from src.utils.validation import ValidationError, GraphInvariantError
from src.utils.llm_client import ResponseUsage


@pytest.fixture
def mock_config():
    """Тестовая конфигурация."""
    return {
        'itext2kg': {
            'model': 'gpt-4o',
            'api_key': 'test-key',
            'tpm_limit': 120000,
            'tpm_safety_margin': 0.15,
            'max_completion': 4096,
            'log_level': 'info',
            'temperature': 1.0,
            'timeout': 45,
            'max_retries': 3
        }
    }


@pytest.fixture
def setup_test_env(tmp_path, mock_config):
    """Подготовка тестового окружения."""
    # Создаем структуру каталогов
    (tmp_path / "prompts").mkdir()
    (tmp_path / "schemas").mkdir()
    (tmp_path / "data" / "staging").mkdir(parents=True)
    (tmp_path / "data" / "out").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    
    # Создаем промпт
    prompt_file = tmp_path / "prompts" / "itext2kg_extraction.md"
    prompt_file.write_text("Test prompt {concept_dictionary_schema} {learning_chunk_graph_schema}")
    
    # Создаем схемы
    concept_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"concepts": {"type": "array"}}
    }
    graph_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"nodes": {"type": "array"}, "edges": {"type": "array"}}
    }
    
    (tmp_path / "schemas" / "ConceptDictionary.schema.json").write_text(json.dumps(concept_schema))
    (tmp_path / "schemas" / "LearningChunkGraph.schema.json").write_text(json.dumps(graph_schema))
    
    return tmp_path


class TestErrorHandlingMethods:
    """Тесты для методов обработки ошибок."""
    
    def test_save_bad_response(self, setup_test_env, mock_config):
        """Тест сохранения некорректного ответа."""
        with patch('src.itext2kg.PROMPTS_DIR', setup_test_env / "prompts"), \
             patch('src.itext2kg.SCHEMAS_DIR', setup_test_env / "schemas"), \
             patch('src.itext2kg.LOGS_DIR', setup_test_env / "logs"):
            
            processor = SliceProcessor(mock_config)
            
            # Сохраняем bad response
            processor._save_bad_response(
                slice_id="slice_001",
                original_response='{"invalid": json',
                error="JSON decode error",
                repair_response='{"still": invalid}'
            )
            
            # Проверяем, что файл создан
            bad_file = setup_test_env / "logs" / "slice_001_bad.json"
            assert bad_file.exists()
            
            # Проверяем содержимое
            bad_data = json.loads(bad_file.read_text())
            assert bad_data["slice_id"] == "slice_001"
            assert bad_data["original_response"] == '{"invalid": json'
            assert bad_data["validation_error"] == "JSON decode error"
            assert bad_data["repair_response"] == '{"still": invalid}'
            assert "timestamp" in bad_data
    
    def test_save_temp_dumps(self, setup_test_env, mock_config):
        """Тест сохранения временных дампов."""
        with patch('src.itext2kg.PROMPTS_DIR', setup_test_env / "prompts"), \
             patch('src.itext2kg.SCHEMAS_DIR', setup_test_env / "schemas"), \
             patch('src.itext2kg.LOGS_DIR', setup_test_env / "logs"):
            
            processor = SliceProcessor(mock_config)
            
            # Добавляем данные
            processor.concept_dictionary = {
                "concepts": [
                    {"concept_id": "test:p:concept", "term": {"primary": "Test"}}
                ]
            }
            processor.learning_graph = {
                "nodes": [{"id": "test:c:0", "type": "Chunk", "text": "Test"}],
                "edges": [{"source": "test:c:0", "target": "test:p:concept", "type": "MENTIONS"}]
            }
            processor.stats.total_slices = 10
            processor.stats.processed_slices = 5
            processor.stats.failed_slices = 2
            
            # Сохраняем дампы
            processor._save_temp_dumps("test_reason")
            
            # Проверяем, что файлы созданы
            logs_dir = setup_test_env / "logs"
            temp_files = list(logs_dir.glob("*_temp_test_reason_*.json"))
            stats_files = list(logs_dir.glob("processing_stats_test_reason_*.json"))
            
            assert len(temp_files) == 2  # ConceptDictionary и LearningChunkGraph
            assert len(stats_files) == 1
            
            # Проверяем содержимое статистики
            stats_file = stats_files[0]
            stats_data = json.loads(stats_file.read_text())
            assert stats_data["reason"] == "test_reason"
            assert stats_data["stats"]["total_slices"] == 10
            assert stats_data["stats"]["processed_slices"] == 5
            assert stats_data["stats"]["failed_slices"] == 2
    
    def test_save_temp_dumps_empty_data(self, setup_test_env, mock_config):
        """Тест сохранения дампов при пустых данных."""
        with patch('src.itext2kg.PROMPTS_DIR', setup_test_env / "prompts"), \
             patch('src.itext2kg.SCHEMAS_DIR', setup_test_env / "schemas"), \
             patch('src.itext2kg.LOGS_DIR', setup_test_env / "logs"):
            
            processor = SliceProcessor(mock_config)
            
            # Пустые данные
            processor._save_temp_dumps("empty_data")
            
            # Проверяем, что сохранилась только статистика
            logs_dir = setup_test_env / "logs"
            temp_files = list(logs_dir.glob("*_temp_empty_data_*.json"))
            stats_files = list(logs_dir.glob("processing_stats_empty_data_*.json"))
            
            assert len(temp_files) == 0  # Пустые данные не сохраняются
            assert len(stats_files) == 1


class TestRepairLogic:
    """Тесты для repair логики."""
    
    def test_successful_repair(self, setup_test_env, mock_config):
        """Тест успешного repair после первой ошибки."""
        with patch('src.itext2kg.PROMPTS_DIR', setup_test_env / "prompts"), \
             patch('src.itext2kg.SCHEMAS_DIR', setup_test_env / "schemas"), \
             patch('src.itext2kg.STAGING_DIR', setup_test_env / "data" / "staging"), \
             patch('src.itext2kg.OUTPUT_DIR', setup_test_env / "data" / "out"), \
             patch('src.itext2kg.LOGS_DIR', setup_test_env / "logs"):
            
            # Создаем слайс
            slice_file = setup_test_env / "data" / "staging" / "test.slice.json"
            slice_data = {
                "id": "slice_001",
                "order": 1,
                "source_file": "test.txt",
                "slug": "test",
                "text": "Test text",
                "slice_token_start": 0,
                "slice_token_end": 100
            }
            slice_file.write_text(json.dumps(slice_data))
            
            processor = SliceProcessor(mock_config)
            
            # Мокаем LLM клиент
            mock_llm = Mock()
            processor.llm_client = mock_llm
            
            # Первый вызов возвращает невалидный JSON
            first_response = ResponseUsage(100, 50, 150, 0)
            mock_llm.create_response.return_value = ('{"invalid": json', 'resp_1', first_response)
            
            # Repair возвращает валидный JSON
            repair_response = ResponseUsage(100, 60, 160, 0)
            valid_response = {
                "concepts_added": {"concepts": []},
                "chunk_graph_patch": {
                    "nodes": [{"id": "test:c:0", "type": "Chunk", "text": "Test", "local_start": 0}],
                    "edges": []
                }
            }
            mock_llm.repair_response.return_value = (json.dumps(valid_response), 'resp_2', repair_response)
            
            # Обрабатываем слайс
            result = processor._process_single_slice(Path(slice_file))
            
            assert result is True
            assert processor.stats.processed_slices == 0  # Увеличивается в run()
            assert len(processor.learning_graph['nodes']) == 1
            
            # Проверяем, что repair был вызван
            assert mock_llm.repair_response.called
    
    def test_failed_repair(self, setup_test_env, mock_config):
        """Тест неудачного repair."""
        with patch('src.itext2kg.PROMPTS_DIR', setup_test_env / "prompts"), \
             patch('src.itext2kg.SCHEMAS_DIR', setup_test_env / "schemas"), \
             patch('src.itext2kg.STAGING_DIR', setup_test_env / "data" / "staging"), \
             patch('src.itext2kg.OUTPUT_DIR', setup_test_env / "data" / "out"), \
             patch('src.itext2kg.LOGS_DIR', setup_test_env / "logs"):
            
            # Создаем слайс
            slice_file = setup_test_env / "data" / "staging" / "test.slice.json"
            slice_data = {
                "id": "slice_001",
                "order": 1,
                "source_file": "test.txt",
                "slug": "test",
                "text": "Test text",
                "slice_token_start": 0,
                "slice_token_end": 100
            }
            slice_file.write_text(json.dumps(slice_data))
            
            processor = SliceProcessor(mock_config)
            
            # Мокаем LLM клиент
            mock_llm = Mock()
            processor.llm_client = mock_llm
            
            # Оба вызова возвращают невалидный JSON
            response_usage = ResponseUsage(100, 50, 150, 0)
            mock_llm.create_response.return_value = ('{"invalid": json', 'resp_1', response_usage)
            mock_llm.repair_response.return_value = ('{"still": invalid}', 'resp_2', response_usage)
            
            # Обрабатываем слайс
            result = processor._process_single_slice(Path(slice_file))
            
            assert result is False
            
            # Проверяем, что bad response был сохранен
            bad_file = setup_test_env / "logs" / "slice_001_bad.json"
            assert bad_file.exists()
            
            bad_data = json.loads(bad_file.read_text())
            assert bad_data["original_response"] == '{"invalid": json'
            assert bad_data["repair_response"] == '{"still": invalid}'


class TestGracefulDegradation:
    """Тесты для graceful degradation."""
    
    def test_partial_failure_handling(self, setup_test_env, mock_config):
        """Тест обработки частичных сбоев."""
        with patch('src.itext2kg.PROMPTS_DIR', setup_test_env / "prompts"), \
             patch('src.itext2kg.SCHEMAS_DIR', setup_test_env / "schemas"), \
             patch('src.itext2kg.STAGING_DIR', setup_test_env / "data" / "staging"), \
             patch('src.itext2kg.OUTPUT_DIR', setup_test_env / "data" / "out"), \
             patch('src.itext2kg.LOGS_DIR', setup_test_env / "logs"):
            
            # Создаем несколько слайсов
            for i in range(3):
                slice_file = setup_test_env / "data" / "staging" / f"test_{i}.slice.json"
                slice_data = {
                    "id": f"slice_{i:03d}",
                    "order": i,
                    "source_file": "test.txt",
                    "slug": "test",
                    "text": f"Test text {i}",
                    "slice_token_start": i * 100,
                    "slice_token_end": (i + 1) * 100
                }
                slice_file.write_text(json.dumps(slice_data))
            
            processor = SliceProcessor(mock_config)
            
            # Мокаем LLM клиент
            mock_llm = Mock()
            processor.llm_client = mock_llm
            
            # Первый и третий слайсы успешны, второй - ошибка
            valid_response = {
                "concepts_added": {"concepts": []},
                "chunk_graph_patch": {"nodes": [], "edges": []}
            }
            response_usage = ResponseUsage(100, 50, 150, 0)
            
            responses = [
                (json.dumps(valid_response), 'resp_1', response_usage),  # Успех
                ('{"invalid": json', 'resp_2', response_usage),         # Ошибка
                (json.dumps(valid_response), 'resp_3', response_usage)  # Успех
            ]
            mock_llm.create_response.side_effect = responses
            mock_llm.repair_response.return_value = ('{"still": invalid}', 'resp_repair', response_usage)
            
            # Запускаем обработку
            result = processor.run()
            
            # Проверяем, что процесс завершился успешно несмотря на ошибку
            assert result == EXIT_SUCCESS
            assert processor.stats.processed_slices == 2
            assert processor.stats.failed_slices == 1
    
    def test_all_slices_failed(self, setup_test_env, mock_config):
        """Тест когда все слайсы failed."""
        with patch('src.itext2kg.PROMPTS_DIR', setup_test_env / "prompts"), \
             patch('src.itext2kg.SCHEMAS_DIR', setup_test_env / "schemas"), \
             patch('src.itext2kg.STAGING_DIR', setup_test_env / "data" / "staging"), \
             patch('src.itext2kg.OUTPUT_DIR', setup_test_env / "data" / "out"), \
             patch('src.itext2kg.LOGS_DIR', setup_test_env / "logs"):
            
            # Создаем слайс
            slice_file = setup_test_env / "data" / "staging" / "test.slice.json"
            slice_data = {
                "id": "slice_001",
                "order": 1,
                "source_file": "test.txt",
                "slug": "test",
                "text": "Test text",
                "slice_token_start": 0,
                "slice_token_end": 100
            }
            slice_file.write_text(json.dumps(slice_data))
            
            processor = SliceProcessor(mock_config)
            
            # Мокаем LLM клиент - всегда возвращает ошибку
            mock_llm = Mock()
            processor.llm_client = mock_llm
            mock_llm.create_response.side_effect = Exception("API Error")
            
            # Запускаем обработку
            result = processor.run()
            
            # Проверяем, что процесс вернул RUNTIME_ERROR
            assert result == EXIT_RUNTIME_ERROR
            assert processor.stats.processed_slices == 0
            assert processor.stats.failed_slices == 1
            
            # Проверяем, что файл статистики создан
            logs_dir = setup_test_env / "logs"
            stats_files = list(logs_dir.glob("processing_stats_all_failed_*.json"))
            assert len(stats_files) > 0
    
    def test_keyboard_interrupt_handling(self, setup_test_env, mock_config):
        """Тест обработки прерывания пользователем."""
        with patch('src.itext2kg.PROMPTS_DIR', setup_test_env / "prompts"), \
             patch('src.itext2kg.SCHEMAS_DIR', setup_test_env / "schemas"), \
             patch('src.itext2kg.STAGING_DIR', setup_test_env / "data" / "staging"), \
             patch('src.itext2kg.OUTPUT_DIR', setup_test_env / "data" / "out"), \
             patch('src.itext2kg.LOGS_DIR', setup_test_env / "logs"):
            
            # Создаем несколько слайсов
            for i in range(3):
                slice_file = setup_test_env / "data" / "staging" / f"test_{i}.slice.json"
                slice_data = {
                    "id": f"slice_{i:03d}",
                    "order": i,
                    "source_file": "test.txt",
                    "slug": "test",
                    "text": f"Test text {i}",
                    "slice_token_start": i * 100,
                    "slice_token_end": (i + 1) * 100
                }
                slice_file.write_text(json.dumps(slice_data))
            
            processor = SliceProcessor(mock_config)
            
            # Мокаем обработку слайсов - прерывание на втором
            with patch.object(processor, '_process_single_slice') as mock_process:
                def side_effect(slice_file):
                    if "test_1" in str(slice_file):
                        raise KeyboardInterrupt()
                    return True
                
                mock_process.side_effect = side_effect
                
                # Запускаем обработку
                result = processor.run()
                
                # Проверяем, что процесс вернул RUNTIME_ERROR
                assert result == EXIT_RUNTIME_ERROR
                
                # Проверяем, что файл статистики создан
                logs_dir = setup_test_env / "logs"
                stats_files = list(logs_dir.glob("processing_stats_interrupted_*.json"))
                assert len(stats_files) > 0


class TestValidationErrorHandling:
    """Тесты для обработки ошибок валидации."""
    
    def test_validation_error_in_finalize(self, setup_test_env, mock_config):
        """Тест обработки ошибки валидации при финализации."""
        with patch('src.itext2kg.PROMPTS_DIR', setup_test_env / "prompts"), \
             patch('src.itext2kg.SCHEMAS_DIR', setup_test_env / "schemas"), \
             patch('src.itext2kg.STAGING_DIR', setup_test_env / "data" / "staging"), \
             patch('src.itext2kg.OUTPUT_DIR', setup_test_env / "data" / "out"), \
             patch('src.itext2kg.LOGS_DIR', setup_test_env / "logs"):
            
            processor = SliceProcessor(mock_config)
            
            # Добавляем невалидные данные, но с содержимым
            processor.concept_dictionary = {"concepts": [{"invalid": "structure"}]}
            processor.learning_graph = {"nodes": [{"invalid": "node"}], "edges": []}
            processor.stats.processed_slices = 1
            
            # Мокаем валидацию чтобы она падала
            with patch('src.itext2kg.validate_json') as mock_validate:
                mock_validate.side_effect = ValidationError("Invalid schema")
                
                # Вызываем финализацию
                result = processor._finalize_and_save()
                
                # Проверяем результат
                assert result == EXIT_RUNTIME_ERROR
                
                # Проверяем, что файлы созданы
                logs_dir = setup_test_env / "logs"
                # Файлы данных имеют _temp_ в названии
                data_temp_files = list(logs_dir.glob("*_temp_validation_failed_*.json"))
                # Файл статистики не имеет _temp_ в названии
                stats_files = list(logs_dir.glob("processing_stats_validation_failed_*.json"))
                
                # Проверяем что файл статистики создан
                assert len(stats_files) == 1
                # И должны быть 2 файла данных (ConceptDictionary и LearningChunkGraph)
                assert len(data_temp_files) == 2
    
    def test_io_error_in_finalize(self, setup_test_env, mock_config):
        """Тест обработки ошибки IO при сохранении."""
        with patch('src.itext2kg.PROMPTS_DIR', setup_test_env / "prompts"), \
             patch('src.itext2kg.SCHEMAS_DIR', setup_test_env / "schemas"), \
             patch('src.itext2kg.STAGING_DIR', setup_test_env / "data" / "staging"), \
             patch('src.itext2kg.OUTPUT_DIR', setup_test_env / "data" / "out"), \
             patch('src.itext2kg.LOGS_DIR', setup_test_env / "logs"):
            
            processor = SliceProcessor(mock_config)
            
            # Добавляем валидные данные
            processor.concept_dictionary = {"concepts": []}
            processor.learning_graph = {"nodes": [], "edges": []}
            processor.stats.processed_slices = 1
            
            # Делаем output директорию недоступной для записи
            output_dir = setup_test_env / "data" / "out"
            output_dir.chmod(0o444)  # Только чтение
            
            try:
                # Вызываем финализацию
                result = processor._finalize_and_save()
                
                # На Windows права могут не работать, проверяем альтернативно
                if result == EXIT_SUCCESS:
                    pytest.skip("Filesystem permissions not enforced on this platform")
                
                # Проверяем результат
                assert result == EXIT_IO_ERROR
                
                # Проверяем, что временные дампы созданы
                logs_dir = setup_test_env / "logs"
                temp_files = list(logs_dir.glob("*_temp_io_error_*.json"))
                assert len(temp_files) > 0
            finally:
                # Восстанавливаем права
                output_dir.chmod(0o755)