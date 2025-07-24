"""
Тесты для обработки дубликатов узлов в itext2kg.
"""

import unittest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.itext2kg import SliceProcessor


class TestNodeDeduplication(unittest.TestCase):
    """Тесты обработки дубликатов узлов."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        self.config = {
            'itext2kg': {
                'api_key': 'test-key',
                'model': 'test-model',
                'tpm_limit': 100000,
                'tpm_safety_margin': 0.15,
                'max_completion': 4096,
                'log_level': 'info',
                'temperature': 0.7,
                'reasoning_effort': 'medium',
                'reasoning_summary': 'auto',
                'timeout': 30,
                'max_retries': 3
            }
        }
        
        # Мокаем LLM клиент
        with patch('src.itext2kg.OpenAIClient'):
            self.processor = SliceProcessor(self.config)
            self.processor.llm_client = Mock()
    
    def test_process_chunk_duplicate_longer_text(self):
        """Обновление Chunk узла если новый текст длиннее."""
        # Добавляем существующий узел
        existing_chunk = {
            "id": "test:c:100",
            "type": "Chunk", 
            "text": "Короткий текст",
            "local_start": 100
        }
        self.processor.learning_graph['nodes'].append(existing_chunk)
        self.processor.known_node_ids.add("test:c:100")
        
        # Новый узел с тем же ID но длиннее
        new_nodes = [{
            "id": "test:c:100",
            "type": "Chunk",
            "text": "Это гораздо более длинный текст чанка с дополнительной информацией",
            "local_start": 100
        }]
        
        # Обрабатываем
        nodes_to_add = self.processor._process_chunk_nodes(new_nodes)
        
        # Проверяем что узел обновлен
        self.assertEqual(len(nodes_to_add), 0)  # Не добавляем новый
        self.assertEqual(len(self.processor.learning_graph['nodes']), 1)
        self.assertEqual(
            self.processor.learning_graph['nodes'][0]['text'],
            "Это гораздо более длинный текст чанка с дополнительной информацией"
        )
    
    def test_process_chunk_duplicate_shorter_text(self):
        """Игнорирование Chunk узла если новый текст короче."""
        # Добавляем существующий узел
        existing_chunk = {
            "id": "test:c:100",
            "type": "Chunk",
            "text": "Это длинный текст чанка с полной информацией",
            "local_start": 100
        }
        self.processor.learning_graph['nodes'].append(existing_chunk)
        self.processor.known_node_ids.add("test:c:100")
        
        # Новый узел с тем же ID но короче
        new_nodes = [{
            "id": "test:c:100",
            "type": "Chunk",
            "text": "Короткий",
            "local_start": 100
        }]
        
        # Обрабатываем
        nodes_to_add = self.processor._process_chunk_nodes(new_nodes)
        
        # Проверяем что узел НЕ обновлен
        self.assertEqual(len(nodes_to_add), 0)
        self.assertEqual(
            self.processor.learning_graph['nodes'][0]['text'],
            "Это длинный текст чанка с полной информацией"
        )
    
    def test_process_assessment_duplicate(self):
        """Игнорирование дубликатов Assessment узлов."""
        # Добавляем существующий Assessment
        existing_assessment = {
            "id": "test:q:200:0",
            "type": "Assessment",
            "text": "Что такое алгоритм?",
            "local_start": 200
        }
        self.processor.learning_graph['nodes'].append(existing_assessment)
        self.processor.known_node_ids.add("test:q:200:0")
        
        # Пытаемся добавить дубликат
        new_nodes = [{
            "id": "test:q:200:0",
            "type": "Assessment",
            "text": "Что такое алгоритм?",
            "local_start": 200
        }]
        
        # Мокаем logger для проверки предупреждения
        with patch.object(self.processor.logger, 'warning') as mock_warning:
            nodes_to_add = self.processor._process_chunk_nodes(new_nodes)
            
            # Проверяем что узел игнорирован
            self.assertEqual(len(nodes_to_add), 0)
            self.assertEqual(len(self.processor.learning_graph['nodes']), 1)
            
            # Проверяем что было логирование
            mock_warning.assert_called_once()
            log_data = json.loads(mock_warning.call_args[0][0])
            self.assertEqual(log_data['event'], 'assessment_duplicate_ignored')
            self.assertEqual(log_data['node_id'], 'test:q:200:0')
    
    def test_process_concept_nodes_pass_through(self):
        """Concept узлы проходят без изменений через _process_chunk_nodes."""
        new_nodes = [
            {
                "id": "test:p:algorithm",
                "type": "Concept",
                "text": "Алгоритм",
                "definition": "Последовательность действий",
                "local_start": 0
            },
            {
                "id": "test:p:structure",
                "type": "Concept", 
                "text": "Структура",
                "definition": "Организация данных",
                "local_start": 0
            }
        ]
        
        # Обрабатываем
        nodes_to_add = self.processor._process_chunk_nodes(new_nodes)
        
        # Все узлы должны пройти
        self.assertEqual(len(nodes_to_add), 2)
        self.assertEqual(nodes_to_add[0]['id'], "test:p:algorithm")
        self.assertEqual(nodes_to_add[1]['id'], "test:p:structure")
        
        # ID должны быть добавлены в known
        self.assertIn("test:p:algorithm", self.processor.known_node_ids)
        self.assertIn("test:p:structure", self.processor.known_node_ids)
    
    def test_process_mixed_nodes(self):
        """Обработка смешанного набора узлов."""
        # Существующие узлы
        self.processor.learning_graph['nodes'] = [
            {"id": "test:c:100", "type": "Chunk", "text": "Старый чанк", "local_start": 100},
            {"id": "test:q:300:0", "type": "Assessment", "text": "Вопрос 1", "local_start": 300}
        ]
        self.processor.known_node_ids = {"test:c:100", "test:q:300:0"}
        
        # Новые узлы
        new_nodes = [
            # Дубликат Chunk с длинным текстом
            {"id": "test:c:100", "type": "Chunk", "text": "Обновленный более длинный чанк", "local_start": 100},
            # Новый Chunk
            {"id": "test:c:200", "type": "Chunk", "text": "Новый чанк", "local_start": 200},
            # Дубликат Assessment
            {"id": "test:q:300:0", "type": "Assessment", "text": "Вопрос 1", "local_start": 300},
            # Новый Assessment
            {"id": "test:q:400:0", "type": "Assessment", "text": "Вопрос 2", "local_start": 400},
            # Новый Concept
            {"id": "test:p:concept", "type": "Concept", "text": "Концепт", "local_start": 0}
        ]
        
        # Обрабатываем
        nodes_to_add = self.processor._process_chunk_nodes(new_nodes)
        
        # Проверяем результаты
        self.assertEqual(len(nodes_to_add), 3)  # Новый Chunk, Assessment и Concept
        
        # Проверяем что Chunk обновлен
        self.assertEqual(
            self.processor.learning_graph['nodes'][0]['text'],
            "Обновленный более длинный чанк"
        )
        
        # Проверяем ID добавленных узлов
        added_ids = [node['id'] for node in nodes_to_add]
        self.assertIn("test:c:200", added_ids)
        self.assertIn("test:q:400:0", added_ids)
        self.assertIn("test:p:concept", added_ids)


class TestIncrementalValidation(unittest.TestCase):
    """Тесты инкрементальной валидации."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        self.config = {
            'itext2kg': {
                'api_key': 'test-key',
                'model': 'test-model',
                'tpm_limit': 100000,
                'tpm_safety_margin': 0.15,
                'max_completion': 4096,
                'log_level': 'info',
                'temperature': 0.7,
                'reasoning_effort': 'medium', 
                'reasoning_summary': 'auto',
                'timeout': 30,
                'max_retries': 3
            }
        }
    
    @patch('src.itext2kg.validate_graph_invariants_intermediate')
    @patch('src.itext2kg.validate_concept_dictionary_invariants')
    def test_incremental_validation_called(self, mock_dict_val, mock_graph_val):
        """Инкрементальная валидация вызывается после применения патча."""
        with patch('src.itext2kg.OpenAIClient'):
            processor = SliceProcessor(self.config)
            processor.llm_client = Mock()
            
            # Мокаем успешный ответ LLM
            processor.llm_client.create_response.return_value = (
                json.dumps({
                    "concepts_added": {"concepts": []},
                    "chunk_graph_patch": {
                        "nodes": [{
                            "id": "test:c:100",
                            "type": "Chunk",
                            "text": "Тест",
                            "local_start": 100
                        }],
                        "edges": []
                    }
                }),
                "response_id_123",
                Mock(total_tokens=100, input_tokens=80, output_tokens=20, reasoning_tokens=0)
            )
            
            # Создаем тестовый слайс
            slice_file = Path("test_slice.json")
            with patch.object(Path, 'read_text') as mock_read:
                mock_read.return_value = json.dumps({
                    "id": "slice_001",
                    "order": 1,
                    "source_file": "test.txt",
                    "slug": "test",
                    "text": "Test content",
                    "slice_token_start": 0,
                    "slice_token_end": 100
                })
                
                # Обрабатываем слайс
                result = processor._process_single_slice(slice_file)
                
                # Проверяем что валидация была вызвана
                self.assertTrue(result)
                mock_graph_val.assert_called_once()
                mock_dict_val.assert_called_once()
    
    @patch('src.itext2kg.validate_graph_invariants_intermediate')
    def test_validation_error_handling(self, mock_graph_val):
        """Обработка ошибок инкрементальной валидации."""
        # Настраиваем mock для генерации ошибки
        from src.utils.validation import GraphInvariantError
        mock_graph_val.side_effect = GraphInvariantError("Дублированный ID узла (Chunk): test:c:100")
        
        with patch('src.itext2kg.OpenAIClient'):
            processor = SliceProcessor(self.config)
            processor.llm_client = Mock()
            
            # Мокаем методы
            processor._save_temp_dumps = Mock()
            
            # Мокаем успешный ответ LLM
            processor.llm_client.create_response.return_value = (
                json.dumps({
                    "concepts_added": {"concepts": []},
                    "chunk_graph_patch": {
                        "nodes": [{
                            "id": "test:c:100",
                            "type": "Chunk",
                            "text": "Тест",
                            "local_start": 100
                        }],
                        "edges": []
                    }
                }),
                "response_id_123",
                Mock(total_tokens=100, input_tokens=80, output_tokens=20, reasoning_tokens=0)
            )
            
            # Создаем тестовый слайс
            slice_file = Path("test_slice.json")
            with patch.object(Path, 'read_text') as mock_read:
                mock_read.return_value = json.dumps({
                    "id": "slice_001",
                    "order": 1,
                    "source_file": "test.txt",
                    "slug": "test",
                    "text": "Test content",
                    "slice_token_start": 0,
                    "slice_token_end": 100
                })
                
                # Обрабатываем слайс
                result = processor._process_single_slice(slice_file)
                
                # Проверяем что слайс помечен как failed
                self.assertFalse(result)
                
                # Проверяем что временный дамп был сохранен
                processor._save_temp_dumps.assert_called_once()
                call_args = processor._save_temp_dumps.call_args[0][0]
                self.assertIn("validation_error_slice_slice_001", call_args)


class TestEdgeDeduplication(unittest.TestCase):
    """Тесты дедупликации рёбер в _validate_edges."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        self.config = {
            'itext2kg': {
                'api_key': 'test-key',
                'model': 'test-model',
                'tpm_limit': 100000,
                'tpm_safety_margin': 0.15,
                'max_completion': 4096,
                'log_level': 'info',
                'temperature': 0.7,
                'reasoning_effort': 'medium',
                'reasoning_summary': 'auto',
                'timeout': 30,
                'max_retries': 3
            }
        }
        
        # Мокаем LLM клиент
        with patch('src.itext2kg.OpenAIClient'):
            self.processor = SliceProcessor(self.config)
            self.processor.llm_client = Mock()
    
    def test_validate_edges_filters_duplicates(self):
        """Фильтрация дублированных рёбер в _validate_edges."""
        # Настраиваем существующие узлы и концепты
        self.processor.known_node_ids = {"chunk1", "chunk2", "assessment1"}
        self.processor.concept_id_map = {"concept1": 0, "concept2": 1}
        
        # Добавляем существующие рёбра в граф
        self.processor.learning_graph['edges'] = [
            {
                "source": "chunk1",
                "target": "concept1",
                "type": "MENTIONS",
                "weight": 1.0
            },
            {
                "source": "chunk2",
                "target": "concept2",
                "type": "PREREQUISITE",
                "weight": 0.8
            }
        ]
        
        # Создаем список рёбер для валидации с дубликатами
        edges_to_validate = [
            # Дублирует существующее MENTIONS ребро - должно быть отфильтровано
            {
                "source": "chunk1",
                "target": "concept1",
                "type": "MENTIONS",
                "weight": 1.0
            },
            # Новое валидное ребро - должно пройти
            {
                "source": "chunk1",
                "target": "concept2",
                "type": "MENTIONS",
                "weight": 1.0
            },
            # Дублирует существующее PREREQUISITE ребро - должно быть отфильтровано
            {
                "source": "chunk2",
                "target": "concept2",
                "type": "PREREQUISITE",
                "weight": 0.9  # Другой вес, но все равно дубликат
            },
            # Новое валидное ребро - должно пройти
            {
                "source": "assessment1",
                "target": "chunk1",
                "type": "TESTS",
                "weight": 0.7
            },
            # Дубликат внутри патча - должно быть отфильтровано
            {
                "source": "assessment1",
                "target": "chunk1",
                "type": "TESTS",
                "weight": 0.7
            }
        ]
        
        # Выполняем валидацию
        valid_edges = self.processor._validate_edges(edges_to_validate)
        
        # Проверяем результаты
        self.assertEqual(len(valid_edges), 2, f"Expected 2 valid edges, got {len(valid_edges)}")
        
        # Проверяем, что прошли только новые валидные рёбра
        valid_edge_keys = [(e['source'], e['target'], e['type']) for e in valid_edges]
        self.assertIn(("chunk1", "concept2", "MENTIONS"), valid_edge_keys)
        self.assertIn(("assessment1", "chunk1", "TESTS"), valid_edge_keys)
        
        # Проверяем, что дубликаты отфильтрованы
        self.assertNotIn(("chunk1", "concept1", "MENTIONS"), valid_edge_keys)
        self.assertNotIn(("chunk2", "concept2", "PREREQUISITE"), valid_edge_keys)
    
    def test_validate_edges_duplicate_logging(self):
        """Логирование при фильтрации дублированных рёбер."""
        # Настраиваем существующие узлы
        self.processor.known_node_ids = {"chunk1"}
        self.processor.concept_id_map = {"concept1": 0}
        
        # Добавляем существующее ребро
        self.processor.learning_graph['edges'] = [
            {
                "source": "chunk1",
                "target": "concept1",
                "type": "MENTIONS",
                "weight": 1.0
            }
        ]
        
        # Пытаемся добавить дубликат
        edges_to_validate = [
            {
                "source": "chunk1",
                "target": "concept1",
                "type": "MENTIONS",
                "weight": 1.0
            }
        ]
        
        # Мокаем logger.info для проверки логирования
        with patch.object(self.processor.logger, 'info') as mock_info:
            self.processor._validate_edges(edges_to_validate)
            
            # Проверяем, что был вызов логирования
            mock_info.assert_called()
            
            # Проверяем содержимое лога
            log_call_found = False
            for call in mock_info.call_args_list:
                log_message = call[0][0]
                try:
                    log_data = json.loads(log_message)
                    if (log_data.get('event') == 'edge_dropped' and 
                        log_data.get('reason') == 'duplicate_edge'):
                        log_call_found = True
                        self.assertEqual(log_data.get('source'), 'chunk1')
                        self.assertEqual(log_data.get('target'), 'concept1')
                        self.assertEqual(log_data.get('type'), 'MENTIONS')
                        break
                except json.JSONDecodeError:
                    continue
            
            self.assertTrue(log_call_found, "Expected duplicate edge log message not found")
    
    def test_mentions_edges_duplication_scenario(self):
        """Сценарий с MENTIONS рёбрами через previous_response_id."""
        # Настройка: у нас есть chunk из предыдущего слайса
        self.processor.known_node_ids = {"test:c:0", "test:c:100"}
        self.processor.concept_id_map = {"test:p:fibonachchi": 0, "test:p:recursion": 1}
        
        # В графе уже есть MENTIONS от старого чанка
        self.processor.learning_graph['edges'] = [
            {
                "source": "test:c:0",
                "target": "test:p:fibonachchi", 
                "type": "MENTIONS",
                "weight": 1.0
            }
        ]
        
        # LLM возвращает патч с дублированным MENTIONS (видит старый чанк через контекст)
        edges_from_llm = [
            # Дубликат - должен быть отфильтрован
            {
                "source": "test:c:0",
                "target": "test:p:fibonachchi",
                "type": "MENTIONS", 
                "weight": 1.0
            },
            # Новое ребро от нового чанка - должно пройти
            {
                "source": "test:c:100",
                "target": "test:p:recursion",
                "type": "MENTIONS",
                "weight": 1.0
            }
        ]
        
        # Валидируем
        valid_edges = self.processor._validate_edges(edges_from_llm)
        
        # Проверяем
        self.assertEqual(len(valid_edges), 1)
        self.assertEqual(valid_edges[0]['source'], "test:c:100")
        self.assertEqual(valid_edges[0]['target'], "test:p:recursion")
    
    def test_validate_edges_preserves_other_validations(self):
        """Проверка, что остальные валидации работают вместе с дедупликацией."""
        # Настраиваем узлы
        self.processor.known_node_ids = {"chunk1", "chunk2"}
        self.processor.concept_id_map = {"concept1": 0}
        
        edges_to_validate = [
            # Битая ссылка - должна быть отфильтрована
            {
                "source": "unknown_node",
                "target": "concept1",
                "type": "MENTIONS",
                "weight": 1.0
            },
            # PREREQUISITE self-loop - должна быть отфильтрована
            {
                "source": "chunk1",
                "target": "chunk1", 
                "type": "PREREQUISITE",
                "weight": 0.5
            },
            # Неверный вес - должен быть отфильтрован
            {
                "source": "chunk1",
                "target": "concept1",
                "type": "MENTIONS",
                "weight": 1.5
            },
            # Валидное ребро - должно пройти
            {
                "source": "chunk2",
                "target": "concept1",
                "type": "PREREQUISITE",
                "weight": 0.8
            }
        ]
        
        # Валидируем
        valid_edges = self.processor._validate_edges(edges_to_validate)
        
        # Только одно валидное ребро должно пройти
        self.assertEqual(len(valid_edges), 1)
        self.assertEqual(valid_edges[0]['source'], "chunk2")
        self.assertEqual(valid_edges[0]['type'], "PREREQUISITE")


if __name__ == '__main__':
    unittest.main()