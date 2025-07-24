#!/usr/bin/env python3
"""
dedup.py - Утилита для удаления дубликатов узлов из графа знаний

Удаляет дубликаты узлов типа Chunk и Assessment, которые появились из-за overlap,
некорректного разреза или разных границ Chunk-ов. Использует векторные эмбеддинги
и FAISS для поиска похожих узлов.

Использование:
    python -m src.dedup
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime
import csv

# Добавляем корень проекта в PYTHONPATH для корректных импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import faiss

# Импорт утилит проекта
from src.utils.config import load_config
from src.utils.validation import validate_json
from src.utils.llm_embeddings import get_embeddings
from src.utils.exit_codes import (
    EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR, 
    EXIT_RUNTIME_ERROR, EXIT_API_LIMIT_ERROR, EXIT_IO_ERROR
)

# Установка UTF-8 кодировки для Windows консоли
from src.utils.console_encoding import setup_console_encoding
setup_console_encoding()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class UnionFind:
    """Union-Find структура данных для кластеризации дубликатов"""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x: str) -> str:
        """Найти корень элемента с path compression"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: str, y: str):
        """Объединить два элемента"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
    
    def get_clusters(self) -> Dict[str, List[str]]:
        """Получить все кластеры"""
        clusters = {}
        for x in self.parent:
            root = self.find(x)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(x)
        return clusters


def filter_nodes_for_dedup(nodes: List[Dict]) -> List[Dict]:
    """
    Фильтрация узлов для дедупликации
    
    Обрабатываем только узлы типа Chunk и Assessment с непустым текстом
    """
    filtered = []
    for node in nodes:
        if node.get('type') in ['Chunk', 'Assessment']:
            text = node.get('text')
            if text is not None and text.strip():
                filtered.append(node)
    
    logger.info(f"Отфильтровано {len(filtered)} узлов из {len(nodes)} для дедупликации")
    return filtered


def build_faiss_index(embeddings: np.ndarray, config: Dict) -> faiss.IndexHNSWFlat:
    """
    Создание FAISS индекса для быстрого поиска похожих векторов
    """
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(
        dim, 
        config['faiss_M'],
        faiss.METRIC_INNER_PRODUCT  # Используем inner product для нормализованных векторов
    )
    index.hnsw.efConstruction = config['faiss_efC']
    index.add(embeddings)
    
    logger.debug(f"Создан FAISS индекс с {embeddings.shape[0]} векторами")
    return index


def find_duplicates(
    nodes: List[Dict], 
    embeddings: np.ndarray, 
    index: faiss.IndexHNSWFlat,
    config: Dict
) -> List[Tuple[str, str, float]]:
    """
    Поиск кандидатов-дубликатов через FAISS
    
    Returns:
        Список кортежей (master_id, duplicate_id, similarity)
    """
    duplicates = []
    k_neighbors = min(config['k_neighbors'] + 1, len(nodes))  # +1 так как сам узел тоже в результатах
    
    # Поиск ближайших соседей для всех узлов сразу
    similarities, indices = index.search(embeddings, k_neighbors)
    
    for i, node in enumerate(nodes):
        node_text_len = len(node['text'])
        
        for j in range(1, k_neighbors):  # Пропускаем j=0 (сам узел)
            neighbor_idx = indices[i, j]
            if neighbor_idx == -1:  # Нет больше соседей
                break
                
            similarity = similarities[i, j]
            if similarity < config['sim_threshold']:
                continue
            
            neighbor = nodes[neighbor_idx]
            neighbor_text_len = len(neighbor['text'])
            
            # Проверка length ratio
            len_ratio = min(node_text_len, neighbor_text_len) / max(node_text_len, neighbor_text_len)
            if len_ratio < config['len_ratio_min']:
                continue
            
            # Определение master и duplicate по local_start, затем по id
            if node['local_start'] < neighbor['local_start']:
                master, duplicate = node, neighbor
            elif node['local_start'] > neighbor['local_start']:
                master, duplicate = neighbor, node
            else:  # local_start равны
                if node['id'] < neighbor['id']:
                    master, duplicate = node, neighbor
                else:
                    master, duplicate = neighbor, node
            
            # Избегаем дублирования пар
            if i < neighbor_idx:
                duplicates.append((master['id'], duplicate['id'], similarity))
    
    logger.info(f"Найдено {len(duplicates)} потенциальных дубликатов")
    return duplicates


def cluster_duplicates(duplicates: List[Tuple[str, str, float]]) -> Dict[str, str]:
    """
    Кластеризация дубликатов через Union-Find
    
    Returns:
        Словарь {duplicate_id: master_id}
    """
    if not duplicates:
        return {}
    
    uf = UnionFind()
    
    # Сохраняем информацию о том, кто был master в исходных парах
    # В duplicates первый элемент - всегда master (определен по local_start)
    initial_masters = {}
    for master_id, duplicate_id, _ in duplicates:
        uf.union(master_id, duplicate_id)
        # Запоминаем, что duplicate_id должен указывать на master_id
        initial_masters[duplicate_id] = master_id
    
    # Получаем кластеры
    clusters = uf.get_clusters()
    dedup_map = {}
    
    # Для каждого кластера определяем финальный master
    for cluster_nodes in clusters.values():
        if len(cluster_nodes) > 1:
            # Находим узел, который был master в исходных парах
            # Если таких несколько, берем минимальный
            masters_in_cluster = set()
            for node in cluster_nodes:
                if node not in initial_masters:
                    # Этот узел был master в какой-то паре
                    masters_in_cluster.add(node)
            
            # Выбираем финальный master
            if masters_in_cluster:
                master_id = min(masters_in_cluster)
            else:
                # Все узлы были duplicates, берем минимальный
                master_id = min(cluster_nodes)
            
            # Создаем маппинг для всех не-master узлов
            for node_id in cluster_nodes:
                if node_id != master_id:
                    dedup_map[node_id] = master_id
    
    logger.info(f"Сформировано {len(clusters)} кластеров, {len(dedup_map)} узлов помечены как дубликаты")
    return dedup_map


def rewrite_graph(graph: Dict, dedup_map: Dict[str, str]) -> Dict:
    """
    Перезапись графа с заменой ID дубликатов на master ID
    и удалением узлов с пустым текстом
    """
    # Создаем новый граф
    new_graph = {
        'nodes': [],
        'edges': []
    }
    
    # Фильтруем узлы
    removed_duplicates = 0
    removed_empty = 0
    for node in graph['nodes']:
        # Пропускаем дубликаты
        if node['id'] in dedup_map:
            removed_duplicates += 1
            continue
            
        # Удаляем узлы с пустым текстом (только Chunk и Assessment)
        if node.get('type') in ['Chunk', 'Assessment']:
            text = node.get('text', '')
            if not text.strip():
                removed_empty += 1
                continue
        
        # Добавляем узел в новый граф
        new_graph['nodes'].append(node)
    
    logger.info(f"Удалено {removed_duplicates} узлов-дубликатов, {removed_empty} пустых узлов")
    
    # Обновляем рёбра
    seen_edges = set()  # Для удаления дублированных рёбер
    updated_edges_count = 0
    
    for edge in graph['edges']:
        # Заменяем ID на master если это дубликат
        source = dedup_map.get(edge['source'], edge['source'])
        target = dedup_map.get(edge['target'], edge['target'])
        
        # Проверяем, что узлы существуют (включая удаленные пустые)
        node_ids = {n['id'] for n in new_graph['nodes']}
        if source not in node_ids or target not in node_ids:
            logger.debug(f"Отброшено висячее ребро: {source} -> {target}")
            continue
        
        # Создаем ключ для проверки дубликатов рёбер
        edge_key = (source, target, edge['type'])
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        
        # Обновляем ребро если изменились ID
        if source != edge['source'] or target != edge['target']:
            updated_edges_count += 1
        
        new_edge = edge.copy()
        new_edge['source'] = source
        new_edge['target'] = target
        new_graph['edges'].append(new_edge)
    
    logger.info(f"Обновлено {updated_edges_count} рёбер, финальное количество: {len(new_graph['edges'])}")
    
    return new_graph


def save_dedup_map(dedup_map: Dict[str, str], duplicates: List[Tuple[str, str, float]]):
    """
    Сохранение маппинга дубликатов в CSV файл
    """
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    csv_path = logs_dir / 'dedup_map.csv'
    
    # Создаем словарь similarity для быстрого доступа
    similarity_map = {(master, dup): sim for master, dup, sim in duplicates}
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['duplicate_id', 'master_id', 'similarity'])
        
        for duplicate_id, master_id in sorted(dedup_map.items()):
            # Ищем similarity
            sim = similarity_map.get((master_id, duplicate_id), 
                                    similarity_map.get((duplicate_id, master_id), 0.0))
            writer.writerow([duplicate_id, master_id, f"{sim:.4f}"])
    
    logger.info(f"Сохранён маппинг дубликатов в {csv_path}")


def main():
    """Основная функция"""
    start_time = time.time()
    
    try:
        # Загрузка конфигурации
        config = load_config()
        dedup_config = config['dedup']
        
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        return EXIT_CONFIG_ERROR
    
    # Пути к файлам
    input_path = Path('data/out/LearningChunkGraph_raw.json')
    output_path = Path('data/out/LearningChunkGraph_dedup.json')
    
    # Проверка входного файла
    if not input_path.exists():
        logger.error(f"Входной файл не найден: {input_path}")
        return EXIT_INPUT_ERROR
    
    try:
        # Загрузка графа
        logger.info("Загрузка графа знаний...")
        with open(input_path, 'r', encoding='utf-8') as f:
            graph = json.load(f)
        
        # Валидация входного графа
        try:
            validate_json(graph, 'LearningChunkGraph')
        except Exception as e:
            logger.error(f"Входной граф не соответствует схеме: {e}")
            return EXIT_INPUT_ERROR
        
        # Фильтрация узлов для дедупликации
        nodes_to_dedup = filter_nodes_for_dedup(graph['nodes'])
        
        if len(nodes_to_dedup) < 2:
            logger.info("Недостаточно узлов для дедупликации, копируем граф без изменений")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            # Сохраняем пустой dedup_map
            save_dedup_map({}, [])
            return EXIT_SUCCESS
        
        # Сортировка узлов по local_start для детерминированности
        nodes_to_dedup.sort(key=lambda n: (n.get('local_start', 0), n['id']))
        
        # Получение эмбеддингов
        logger.info(f"Получение эмбеддингов для {len(nodes_to_dedup)} узлов...")
        texts = [node['text'] for node in nodes_to_dedup]
        
        try:
            embeddings = get_embeddings(texts, dedup_config)
        except Exception as e:
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                logger.error(f"Превышен лимит API: {e}")
                return EXIT_API_LIMIT_ERROR
            else:
                logger.error(f"Ошибка получения эмбеддингов: {e}")
                return EXIT_RUNTIME_ERROR
        
        # Построение FAISS индекса
        logger.info("Построение FAISS индекса...")
        index = build_faiss_index(embeddings, dedup_config)
        
        # Поиск дубликатов
        logger.info("Поиск дубликатов...")
        duplicates = find_duplicates(nodes_to_dedup, embeddings, index, dedup_config)
        
        if not duplicates:
            logger.info("Дубликаты не найдены, копируем граф без изменений")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            # Сохраняем пустой dedup_map
            save_dedup_map({}, [])
            return EXIT_SUCCESS
        
        # Кластеризация дубликатов
        logger.info("Кластеризация дубликатов...")
        dedup_map = cluster_duplicates(duplicates)
        
        # Перезапись графа
        logger.info("Перезапись графа...")
        new_graph = rewrite_graph(graph, dedup_map)
        
        # Валидация выходного графа
        try:
            validate_json(new_graph, 'LearningChunkGraph')
        except Exception as e:
            logger.error(f"Выходной граф не соответствует схеме: {e}")
            return EXIT_RUNTIME_ERROR
        
        # Сохранение результатов
        logger.info("Сохранение результатов...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_graph, f, ensure_ascii=False, indent=2)
        
        # Сохранение маппинга дубликатов
        save_dedup_map(dedup_map, duplicates)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Дедупликация завершена за {elapsed_time:.2f} секунд")
        logger.info(f"Узлов было: {len(graph['nodes'])}, стало: {len(new_graph['nodes'])}")
        logger.info(f"Рёбер было: {len(graph['edges'])}, стало: {len(new_graph['edges'])}")
        
        return EXIT_SUCCESS
        
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
        return EXIT_RUNTIME_ERROR


if __name__ == '__main__':
    sys.exit(main())