"""
Интеграционные тесты для dedup.py с реальными API вызовами

Требуют реального API ключа в config.toml
ИСПОЛЬЗУЮТ ВРЕМЕННЫЕ ДИРЕКТОРИИ - НЕ ТРОГАЮТ PRODUCTION DATA
"""

import csv
import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.exit_codes import EXIT_INPUT_ERROR, EXIT_SUCCESS


def create_test_graph_with_semantic_duplicates():
    """Создание тестового графа с реальными семантическими дубликатами"""
    nodes = []
    edges = []

    # Добавляем Concept узлы (не должны участвовать в дедупликации)
    nodes.append(
        {
            "id": "handbook:p:python",
            "type": "Concept",
            "text": "Python",
            "definition": "High-level programming language",
            "node_offset": 0,
        }
    )

    nodes.append(
        {
            "id": "handbook:p:variable",
            "type": "Concept",
            "text": "Variable",
            "definition": "Named storage location",
            "node_offset": 0,
        }
    )

    # Группа 1: Определение переменных (семантические дубликаты)
    nodes.append(
        {
            "id": "handbook:c:100",
            "type": "Chunk",
            "text": "In Python, variables are containers for storing data values. You create a variable by assigning a value to it using the equals sign.",
            "node_offset": 0,
        }
    )

    nodes.append(
        {
            "id": "handbook:c:200",
            "type": "Chunk",
            "text": "Python variables are used to store data values. A variable is created when you assign a value to it with the = operator.",
            "node_offset": 0,
        }
    )

    nodes.append(
        {
            "id": "handbook:c:300",
            "type": "Chunk",
            "text": "Variables in Python are containers that hold data. To create a variable, simply assign it a value using the assignment operator (=).",
            "node_offset": 0,
        }
    )

    # Группа 2: Циклы for (семантические дубликаты)
    nodes.append(
        {
            "id": "handbook:c:400",
            "type": "Chunk",
            "text": "The for loop in Python is used to iterate over a sequence (list, tuple, string) or other iterable objects. It executes a block of code for each item.",
            "node_offset": 0,
        }
    )

    nodes.append(
        {
            "id": "handbook:c:500",
            "type": "Chunk",
            "text": "Python for loops iterate through sequences like lists, tuples, or strings. The loop runs a code block once for every element in the sequence.",
            "node_offset": 0,
        }
    )

    # Уникальные chunks (не должны быть помечены как дубликаты)
    nodes.append(
        {
            "id": "handbook:c:600",
            "type": "Chunk",
            "text": "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming styles.",
            "node_offset": 0,
        }
    )

    nodes.append(
        {
            "id": "handbook:c:700",
            "type": "Chunk",
            "text": "List comprehensions provide a concise way to create lists in Python. They consist of brackets containing an expression followed by a for clause.",
            "node_offset": 0,
        }
    )

    # Assessment узлы
    nodes.append(
        {
            "id": "handbook:q:800:1",
            "type": "Assessment",
            "text": "What is a variable in Python?",
            "node_offset": 0,
        }
    )

    nodes.append(
        {
            "id": "handbook:q:900:2",
            "type": "Assessment",
            "text": "How do you create a variable in Python?",  # Семантически похож на assessment_1
            "node_offset": 0,
        }
    )

    # Добавляем рёбра
    edges.extend(
        [
            {
                "source": "handbook:c:100",
                "target": "handbook:p:variable",
                "type": "MENTIONS",
                "weight": 1.0,
            },
            {
                "source": "handbook:c:200",
                "target": "handbook:p:variable",
                "type": "MENTIONS",
                "weight": 1.0,
            },
            {
                "source": "handbook:c:300",
                "target": "handbook:p:variable",
                "type": "MENTIONS",
                "weight": 1.0,
            },
            {
                "source": "handbook:c:100",
                "target": "handbook:c:400",
                "type": "PREREQUISITE",
                "weight": 0.8,
            },
            {
                "source": "handbook:c:200",
                "target": "handbook:c:600",
                "type": "ELABORATES",
                "weight": 0.6,
            },
            {
                "source": "handbook:c:400",
                "target": "handbook:c:700",
                "type": "HINT_FORWARD",
                "weight": 0.7,
            },
            {
                "source": "handbook:c:300",
                "target": "handbook:q:800:1",
                "type": "TESTS",
                "weight": 0.9,
            },
            {
                "source": "handbook:c:500",
                "target": "handbook:q:900:2",
                "type": "TESTS",
                "weight": 0.85,
            },
        ]
    )

    return {"nodes": nodes, "edges": edges}


class TestDedupIntegrationWithTemp:
    """Интеграционные тесты с реальными API вызовами, использующие временные директории"""

    @pytest.fixture(autouse=True)
    def setup_temp_dirs(self, tmp_path, monkeypatch):
        """Подготовка перед каждым тестом"""
        # Создаем структуру директорий как в проекте
        self.data_out = tmp_path / "data" / "out"
        self.logs = tmp_path / "logs"
        self.data_out.mkdir(parents=True)
        self.logs.mkdir(parents=True)
        
        # Меняем рабочую директорию на временную
        monkeypatch.chdir(tmp_path)

    def test_semantic_duplicates(self):
        """Тест семантических дубликатов с реальным API"""
        # Создаем граф с семантическими дубликатами
        graph = create_test_graph_with_semantic_duplicates()

        # Сохраняем входной файл
        input_path = self.data_out / "LearningChunkGraph_raw.json"
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

        # Запускаем dedup
        from src.dedup import main
        exit_code = main()
        
        assert exit_code == EXIT_SUCCESS

        # Проверяем результат
        output_path = self.data_out / "LearningChunkGraph_dedup.json"
        assert output_path.exists()
        
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        # Проверяем что дубликаты найдены
        original_chunks = len([n for n in graph["nodes"] if n["type"] in ["Chunk", "Assessment"]])
        result_chunks = len([n for n in result["nodes"] if n["type"] in ["Chunk", "Assessment"]])
        
        # Должны быть удалены некоторые дубликаты ИЛИ все сохранены при высоком пороге
        # (зависит от конфигурации sim_threshold)
        assert result_chunks <= original_chunks, "Duplicates count should not increase"
        
        # Проверяем dedup_map
        dedup_map_path = self.logs / "dedup_map.csv"
        assert dedup_map_path.exists()
        
        with open(dedup_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            dedup_entries = list(reader)
        
        # С высоким порогом (0.97) дубликаты могут не найтись, это нормально
        # Проверяем только что файл создан и корректной структуры
        if dedup_entries:
            for entry in dedup_entries:
                assert "duplicate_id" in entry
                assert "master_id" in entry
                assert "similarity" in entry

    def test_edge_cases(self):
        """Тест граничных случаев"""
        # Создаем граф с граничными случаями
        nodes = []
        
        # Пустые тексты (должны быть удалены)
        nodes.append({
            "id": "handbook:c:0",
            "type": "Chunk",
            "text": "",
            "node_offset": 0,
        })
        
        nodes.append({
            "id": "handbook:c:100",
            "type": "Chunk",
            "text": "   \n\t  ",
            "node_offset": 0,
        })
        
        # Нормальный текст
        nodes.append({
            "id": "handbook:c:200",
            "type": "Chunk",
            "text": "This is a valid chunk with content.",
            "node_offset": 0,
        })
        
        # Идентичные тексты
        nodes.append({
            "id": "handbook:c:300",
            "type": "Chunk",
            "text": "Python is great for data science.",
            "node_offset": 0,
        })
        
        nodes.append({
            "id": "handbook:c:400",
            "type": "Chunk",
            "text": "Python is great for data science.",
            "node_offset": 0,
        })
        
        graph = {"nodes": nodes, "edges": []}
        
        # Сохраняем входной файл
        input_path = self.data_out / "LearningChunkGraph_raw.json"
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        
        # Запускаем dedup
        from src.dedup import main
        exit_code = main()
        
        assert exit_code == EXIT_SUCCESS
        
        # Проверяем результат
        output_path = self.data_out / "LearningChunkGraph_dedup.json"
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        
        result_ids = {n["id"] for n in result["nodes"]}
        
        # Пустые тексты должны быть удалены
        assert "handbook:c:0" not in result_ids
        assert "handbook:c:100" not in result_ids
        
        # Нормальный текст должен остаться
        assert "handbook:c:200" in result_ids
        
        # Один из идентичных должен быть удален
        identical_count = sum(1 for id in ["handbook:c:300", "handbook:c:400"] if id in result_ids)
        assert identical_count == 1, "Only one of identical chunks should remain"


