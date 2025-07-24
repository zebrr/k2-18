#!/usr/bin/env python
"""
refiner.py - Утилита для добавления дальних связей в граф знаний.

Ищет пропущенные связи между узлами, которые не встречались в одном контексте
при первичной обработке. Использует семантическое сходство для поиска кандидатов
и LLM для анализа типов связей.
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Добавляем корень проекта в PYTHONPATH для корректных импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import faiss

from src.utils.config import load_config
from src.utils.validation import validate_json, validate_graph_invariants
from src.utils.exit_codes import (
    EXIT_SUCCESS, EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR, EXIT_API_LIMIT_ERROR, EXIT_IO_ERROR
)
from src.utils.llm_embeddings import get_embeddings
from src.utils.llm_client import OpenAIClient, TPMBucket

# Установка UTF-8 кодировки для Windows консоли
from src.utils.console_encoding import setup_console_encoding
setup_console_encoding()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("refiner")


def setup_json_logging(config: Dict) -> logging.Logger:
    """
    Настройка JSON Lines логирования для refiner.
    
    Args:
        config: Конфигурация refiner
        
    Returns:
        Настроенный logger
    """
    from datetime import datetime, timezone
    import json
    import logging
    
    # Создаем каталог для логов
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Генерируем имя файла лога
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"refiner_{timestamp}.log"
    
    # Создаем кастомный форматтер для JSON Lines
    class JSONLineFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "event": getattr(record, 'event', 'log'),
            }
            
            # Добавляем дополнительные поля из record
            for key in ['slice_id', 'node_id', 'concept_id', 'action', 'source', 
                       'target', 'type', 'weight', 'conditions', 'pairs_count',
                       'tokens_used', 'duration_ms', 'edges_added', 'error']:
                if hasattr(record, key):
                    log_data[key] = getattr(record, key)
            
            # Добавляем сообщение если есть
            if record.getMessage():
                log_data['message'] = record.getMessage()
            
            # Добавляем данные для DEBUG уровня
            if record.levelname == 'DEBUG':
                for key in ['prompt', 'response', 'raw_response', 'new_aliases', 
                           'old_len', 'new_len', 'similarity']:
                    if hasattr(record, key):
                        log_data[key] = getattr(record, key)
            
            return json.dumps(log_data, ensure_ascii=False)
    
    # Создаем logger
    logger = logging.getLogger("refiner")
    logger.setLevel(logging.DEBUG if config.get('log_level', 'info').lower() == 'debug' else logging.INFO)
    
    # Удаляем существующие handlers
    logger.handlers = []
    
    # File handler для JSON Lines
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(JSONLineFormatter())
    logger.addHandler(file_handler)
    
    # Console handler для обычного вывода (не JSON)
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(logging.Formatter(
    #    '[%(asctime)s] %(levelname)-8s | %(message)s',
    # datefmt='%H:%M:%S'
    # ))
    # console_handler.setLevel(logging.INFO)  # Только важные сообщения в консоль
    # logger.addHandler(console_handler)
    
    # Логируем начало работы
    logger.info("Refiner started", extra={"event": "refiner_start", "config": {
        "model": config["model"],
        "tpm_limit": config["tpm_limit"],
        "sim_threshold": config["sim_threshold"], 
        "max_pairs_per_node": config["max_pairs_per_node"]
    }})
    
    return logger


def log_edge_operation(logger: logging.Logger, operation: str, edge: Dict, **kwargs):
    """
    Логирование операций с рёбрами в структурированном виде.
    
    Args:
        logger: Logger
        operation: Тип операции (added, updated, replaced, removed)
        edge: Данные ребра
        **kwargs: Дополнительные параметры для логирования
    """
    extra = {
        "event": f"edge_{operation}",
        "source": edge.get("source"),
        "target": edge.get("target"),
        "type": edge.get("type"),
        "weight": edge.get("weight")
    }
    extra.update(kwargs)
    
    message = f"Edge {operation}: {edge.get('source')} -> {edge.get('target')} ({edge.get('type')})"
    
    if operation in ["updated", "replaced"]:
        logger.info(message, extra=extra)
    else:
        logger.debug(message, extra=extra)


def validate_refiner_config(config: Dict) -> None:
    """
    Валидация параметров конфигурации refiner.
    
    Args:
        config: Секция [refiner] из конфига
        
    Raises:
        ValueError: Если параметры некорректны
    """
    # Проверка обязательных параметров
    required = [
        "embedding_model", "sim_threshold", "max_pairs_per_node",
        "model", "api_key", "tpm_limit", "max_completion",
        "weight_low", "weight_mid", "weight_high",
        "faiss_M", "faiss_metric"
    ]
    
    for param in required:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
    
    # Проверка api_key
    if not config["api_key"].strip():
        raise ValueError("api_key cannot be empty")
    
    # Проверка диапазонов
    if not 0 <= config["sim_threshold"] <= 1:
        raise ValueError(f"sim_threshold must be in [0,1], got {config['sim_threshold']}")
    
    if config["max_pairs_per_node"] <= 0:
        raise ValueError(f"max_pairs_per_node must be > 0, got {config['max_pairs_per_node']}")
    
    # Проверка весов
    if not (0 <= config["weight_low"] < config["weight_mid"] < config["weight_high"] <= 1):
        raise ValueError(
            f"Weights must satisfy: 0 <= weight_low < weight_mid < weight_high <= 1, "
            f"got {config['weight_low']}, {config['weight_mid']}, {config['weight_high']}"
        )
    
    # Проверка FAISS параметров
    if config["faiss_M"] <= 0:
        raise ValueError(f"faiss_M must be > 0, got {config['faiss_M']}")
    
    if config["faiss_metric"] not in ["INNER_PRODUCT", "L2"]:
        raise ValueError(f"faiss_metric must be INNER_PRODUCT or L2, got {config['faiss_metric']}")
    
    # Проверка reasoning параметров для o-моделей
    if config.get("model", "").startswith("o"):
        if config.get("reasoning_effort") not in ["low", "medium", "high", None]:
            raise ValueError(f"reasoning_effort must be low/medium/high, got {config.get('reasoning_effort')}")
        
        if config.get("reasoning_summary") not in ["auto", "concise", "detailed", None]:
            raise ValueError(f"reasoning_summary must be auto/concise/detailed, got {config.get('reasoning_summary')}")


def load_and_validate_graph(input_path: Path) -> Dict:
    """
    Загрузка и валидация графа.
    
    Args:
        input_path: Путь к файлу графа
        
    Returns:
        Загруженный граф
        
    Raises:
        Exception: При ошибках загрузки или валидации
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    # Валидация по схеме
    validate_json(graph, "LearningChunkGraph")
    
    return graph


def extract_target_nodes(graph: Dict) -> List[Dict]:
    """
    Извлечение узлов типа Chunk и Assessment.
    
    Args:
        graph: Граф знаний
        
    Returns:
        Список целевых узлов
    """
    target_types = {"Chunk", "Assessment"}
    return [
        node for node in graph.get("nodes", [])
        if node.get("type") in target_types
    ]


def build_edges_index(graph: Dict) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Построение индекса существующих рёбер для быстрого поиска.
    
    Args:
        graph: Граф знаний
        
    Returns:
        Словарь {source_id: {target_id: [edges]}}
    """
    edges_index = {}
    
    for edge in graph.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        
        if source not in edges_index:
            edges_index[source] = {}
        
        if target not in edges_index[source]:
            edges_index[source][target] = []
        
        edges_index[source][target].append(edge)
    
    return edges_index


def get_node_embeddings(nodes: List[Dict], config: Dict, logger: logging.Logger) -> Dict[str, np.ndarray]:
    """
    Получение embeddings для всех узлов.
    
    Args:
        nodes: Список узлов (Chunk/Assessment)
        config: Конфигурация refiner
        logger: Logger для вывода
        
    Returns:
        Словарь {node_id: embedding_vector}
    """
    logger.info(f"Getting embeddings for {len(nodes)} nodes")
    
    # Извлекаем тексты в том же порядке, что и узлы
    texts = []
    node_ids = []
    
    for node in nodes:
        if node.get("text", "").strip():
            texts.append(node["text"])
            node_ids.append(node["id"])
        else:
            logger.warning(f"Node {node['id']} has empty text, skipping")
    
    if not texts:
        logger.error("No texts to get embeddings for")
        return {}
    
    try:
        # Получаем embeddings батчами
        embeddings = get_embeddings(texts, config)
        
        # Создаем словарь для быстрого доступа
        embeddings_dict = {}
        for i, node_id in enumerate(node_ids):
            embeddings_dict[node_id] = embeddings[i]
            
        logger.info(f"Successfully obtained embeddings for {len(embeddings_dict)} nodes")
        return embeddings_dict
        
    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        raise


def build_similarity_index(embeddings_dict: Dict[str, np.ndarray], nodes: List[Dict], 
                          config: Dict, logger: logging.Logger) -> Tuple[faiss.Index, List[str]]:
    """
    Построение FAISS индекса для поиска похожих узлов.
    
    Args:
        embeddings_dict: Словарь {node_id: embedding}
        nodes: Список узлов в порядке возрастания local_start
        config: Конфигурация refiner
        logger: Logger
        
    Returns:
        (faiss_index, node_ids_list) - индекс и список ID в порядке добавления
    """
    # Сортируем узлы по local_start для детерминированности
    sorted_nodes = sorted(nodes, key=lambda n: (n.get("local_start", 0), n["id"]))
    
    # Собираем embeddings в правильном порядке
    embeddings_list = []
    node_ids_list = []
    
    for node in sorted_nodes:
        if node["id"] in embeddings_dict:
            embeddings_list.append(embeddings_dict[node["id"]])
            node_ids_list.append(node["id"])
    
    if not embeddings_list:
        raise ValueError("No embeddings to build index")
    
    # Конвертируем в numpy array
    embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
    dim = embeddings_matrix.shape[1]  # Должно быть 1536
    
    logger.info(f"Building FAISS index: dim={dim}, M={config['faiss_M']}, "
                f"metric={config['faiss_metric']}")
    
    # Создаем HNSW индекс
    if config["faiss_metric"] == "INNER_PRODUCT":
        metric = faiss.METRIC_INNER_PRODUCT
    else:
        # На случай если добавят другие метрики
        metric = faiss.METRIC_L2
        
    index = faiss.IndexHNSWFlat(dim, config["faiss_M"], metric)
    index.hnsw.efConstruction = config.get("faiss_efC", 200)
    
    # Добавляем векторы
    index.add(embeddings_matrix)
    
    logger.info(f"FAISS index built with {index.ntotal} vectors")
    return index, node_ids_list


def generate_candidate_pairs(nodes: List[Dict], embeddings_dict: Dict[str, np.ndarray],
                           index: faiss.Index, node_ids_list: List[str],
                           edges_index: Dict, config: Dict, logger: logging.Logger) -> List[Dict]:
    """
    Генерация пар узлов-кандидатов для анализа связей.
    
    Args:
        nodes: Список всех узлов
        embeddings_dict: Словарь embeddings
        index: FAISS индекс
        node_ids_list: Список ID в порядке индекса
        edges_index: Индекс существующих рёбер
        config: Конфигурация
        logger: Logger
        
    Returns:
        Список словарей с информацией о парах для анализа
    """
    # Создаем быстрый доступ к узлам по ID
    nodes_by_id = {node["id"]: node for node in nodes}
    
    # Параметры поиска
    k_neighbors = min(config["max_pairs_per_node"] + 1, len(nodes))  # +1 т.к. найдет себя
    sim_threshold = config["sim_threshold"]
    
    candidate_pairs = []
    processed_pairs = set()  # Чтобы избежать дубликатов (A,B) и (B,A)
    
    logger.info(f"Searching for candidates: k={k_neighbors-1}, threshold={sim_threshold}")
    
    # Для каждого узла ищем кандидатов
    for i, node_id_a in enumerate(node_ids_list):
        node_a = nodes_by_id[node_id_a]
        embedding_a = embeddings_dict[node_id_a]
        
        # Поиск ближайших соседей
        query = np.array([embedding_a], dtype=np.float32)
        similarities, indices = index.search(query, k_neighbors)
        
        candidates_for_a = []
        
        for j, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == i:  # Пропускаем сам узел
                continue
                
            if sim < sim_threshold:  # Отсекаем по порогу
                continue
                
            node_id_b = node_ids_list[idx]
            node_b = nodes_by_id[node_id_b]
            
            # Проверяем порядок local_start
            local_start_a = node_a.get("local_start", 0)
            local_start_b = node_b.get("local_start", 0)
            
            if local_start_a >= local_start_b:
                continue  # Обрабатываем только пары где A < B
            
            # Проверяем, что пара еще не обработана
            pair_key = (node_id_a, node_id_b)
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Собираем существующие рёбра между узлами
            existing_edges = []
            
            # A -> B
            if node_id_a in edges_index and node_id_b in edges_index[node_id_a]:
                for edge in edges_index[node_id_a][node_id_b]:
                    existing_edges.append(edge)
            
            # B -> A
            if node_id_b in edges_index and node_id_a in edges_index[node_id_b]:
                for edge in edges_index[node_id_b][node_id_a]:
                    existing_edges.append(edge)
            
            candidates_for_a.append({
                "node_id": node_id_b,
                "text": node_b["text"],
                "similarity": float(sim),
                "existing_edges": existing_edges
            })
        
        # Сортируем кандидатов по убыванию similarity и берем top-K
        candidates_for_a.sort(key=lambda x: x["similarity"], reverse=True)
        candidates_for_a = candidates_for_a[:config["max_pairs_per_node"]]
        
        if candidates_for_a:
            candidate_pairs.append({
                "source_node": {
                    "id": node_id_a,
                    "text": node_a["text"]
                },
                "candidates": candidates_for_a
            })
        
        # Логирование прогресса
        if (i + 1) % 10 == 0:
            logger.debug(f"Processed {i + 1}/{len(node_ids_list)} nodes")
    
    logger.info(f"Generated {len(candidate_pairs)} nodes with candidates, "
                f"total {sum(len(p['candidates']) for p in candidate_pairs)} pairs")
    
    return candidate_pairs


def load_refiner_prompt(config: Dict) -> str:
    """
    Загрузка и подготовка промпта для анализа связей.
    
    Args:
        config: Конфигурация refiner с весами
        
    Returns:
        Подготовленный промпт
        
    Raises:
        FileNotFoundError: Если файл промпта не найден
    """
    prompt_path = Path(__file__).parent / "prompts" / "refiner_relation.md"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    # Загружаем промпт
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # Подставляем веса из конфига
    prompt = prompt_template.replace("{weight_low}", str(config["weight_low"]))
    prompt = prompt.replace("{weight_mid}", str(config["weight_mid"]))  # Теперь работаем с prompt, а не prompt_template
    prompt = prompt.replace("{weight_high}", str(config["weight_high"]))  # И здесь тоже
    
    return prompt


def analyze_candidate_pairs(candidate_pairs: List[Dict], graph: Dict,
                          config: Dict, logger: logging.Logger) -> List[Dict]:
    """
    Анализ пар кандидатов через LLM для определения типов связей.
    
    Args:
        candidate_pairs: Список пар узлов для анализа
        graph: Исходный граф (для контекста)
        config: Конфигурация refiner
        logger: Logger
        
    Returns:
        Список новых/обновленных рёбер
    """
    # Загружаем промпт
    try:
        prompt = load_refiner_prompt(config)
        logger.info("Loaded refiner prompt with weight substitutions")
    except FileNotFoundError as e:
        logger.error(f"Failed to load prompt: {e}")
        raise
    
    # Инициализируем LLM клиент с правильным форматом конфига
    llm_config = {
        'api_key': config["api_key"],
        'model': config["model"],
        'tpm_limit': config["tpm_limit"],
        'tpm_safety_margin': config.get("tpm_safety_margin", 0.15),
        'max_completion': config["max_completion"],
        'timeout': config.get("timeout", 45),
        'max_retries': config.get("max_retries", 3)
    }
    
    # Добавляем параметры для обычных моделей
    if not config.get("model", "").startswith("o"):
        llm_config['temperature'] = config.get("temperature", 0.6)
    
    # Добавляем параметры для reasoning моделей
    if config.get("reasoning_effort"):
        llm_config['reasoning_effort'] = config["reasoning_effort"]
    if config.get("reasoning_summary"):
        llm_config['reasoning_summary'] = config["reasoning_summary"]
    
    llm_client = OpenAIClient(llm_config)
    
    all_new_edges = []
    previous_response_id = None
    
    # Вывод в терминал START
    from datetime import datetime, timezone, timedelta
    import time
    
    start_time = time.time()
    utc3_tz = timezone(timedelta(hours=3))
    start_timestamp = datetime.now(utc3_tz).strftime("%H:%M:%S")
    
    print(f"[{start_timestamp}] START    | {len(candidate_pairs)} nodes | "
          f"model={config['model']} | tpm={config['tpm_limit']//1000}k")
    
    logger.info(f"Starting LLM analysis of {len(candidate_pairs)} nodes")
    
    # Обрабатываем каждый узел последовательно
    for i, pair_data in enumerate(candidate_pairs):
        source_node = pair_data["source_node"]
        candidates = pair_data["candidates"]
        
        # Формируем input для LLM
        input_data = {
            "source_node": source_node,
            "candidates": candidates
        }
        
        logger.debug(f"Processing node {i+1}/{len(candidate_pairs)}: {source_node['id']} "
                    f"with {len(candidates)} candidates")
        
        request_start = time.time()
        
        # Логируем промпт в DEBUG режиме
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("LLM request", extra={
                "event": "llm_request",
                "node_id": source_node['id'],
                "prompt": prompt[:1000] + "..." if len(prompt) > 1000 else prompt,
                "input_data": json.dumps(input_data, ensure_ascii=False)[:1000] + "..."
            })
        
        try:
            # Отправляем запрос к LLM
            response_text, response_id, usage = llm_client.create_response(
                instructions=prompt,
                input_data=json.dumps(input_data, ensure_ascii=False, indent=2),
                previous_response_id=previous_response_id
            )
            
            # Логируем ответ в DEBUG режиме
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("LLM response", extra={
                    "event": "llm_response",
                    "node_id": source_node['id'],
                    "response": response_text[:1000] + "..." if len(response_text) > 1000 else response_text,
                    "usage": usage
                })
            
            # Сохраняем response_id для следующего вызова
            previous_response_id = response_id
            
            # Парсим ответ
            try:
                edges_response = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response for node {source_node['id']}: {e}")
                logger.debug(f"Raw response: {response_text}")
                
                # Один repair-retry
                logger.info("Attempting repair retry with clarification")
                repair_prompt = prompt + "\n\nPLEASE RETURN ONLY VALID JSON ARRAY, NO OTHER TEXT."
                
                # Обновляем инструкции в клиенте для repair
                llm_client.last_instructions = repair_prompt
                response_text, response_id, usage = llm_client.repair_response(
                    instructions=repair_prompt,
                    input_data=json.dumps(input_data, ensure_ascii=False, indent=2)
                )
                
                try:
                    edges_response = json.loads(response_text)
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to parse repaired response: {e2}")
                    
                    # Сохраняем плохой ответ
                    bad_response_path = Path(f"logs/{source_node['id']}_bad.json")
                    bad_response_path.parent.mkdir(exist_ok=True)
                    
                    with open(bad_response_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "node_id": source_node['id'],
                            "original_response": response_text,
                            "error": str(e2)
                        }, f, ensure_ascii=False, indent=2)
                    
                    logger.error(f"Saved bad response to {bad_response_path}")
                    continue  # Пропускаем этот узел
            
            # Валидация и обработка ответа
            valid_edges = validate_llm_edges(edges_response, source_node['id'], 
                                           candidates, graph, logger)
            
            all_new_edges.extend(valid_edges)
            
            # Логирование прогресса
            added_count = len([e for e in valid_edges if e.get("type")])
            request_time_ms = int((time.time() - request_start) * 1000)
            
            # Вывод в терминал NODE  
            node_timestamp = datetime.now(utc3_tz).strftime("%H:%M:%S")
            tokens_used = usage.get('total_tokens', 0) if usage else 0
            
            print(f"[{node_timestamp}] NODE     | ✅ {i+1:03d}/{len(candidate_pairs):03d} | "
                  f"pairs={len(candidates)} | tokens={tokens_used} | {request_time_ms}ms | "
                  f"edges_added={added_count}")
            
            logger.info(f"[{i+1}/{len(candidate_pairs)}] Node {source_node['id']}: "
                       f"{added_count} new edges from {len(candidates)} candidates")
            
        except Exception as e:
            # Вывод ошибки в терминал
            error_timestamp = datetime.now(utc3_tz).strftime("%H:%M:%S")
            
            if "RateLimitError" in str(e):
                print(f"[{error_timestamp}] ERROR    | ⚠️ RateLimitError | will retry...")
            else:
                print(f"[{error_timestamp}] ERROR    | ⚠️ {type(e).__name__}: {str(e)[:50]}...")
            
            logger.error(f"Error processing node {source_node['id']}: {e}")
            continue
    
    # Вывод END
    end_time = time.time()
    elapsed = int(end_time - start_time)
    minutes = elapsed // 60
    seconds = elapsed % 60
    
    end_timestamp = datetime.now(utc3_tz).strftime("%H:%M:%S")
    total_added = len([e for e in all_new_edges if e.get("type")])
    
    print(f"[{end_timestamp}] END      | Done | nodes={len(candidate_pairs)} | "
          f"edges_added={total_added} | time={minutes}m {seconds}s")
    
    logger.info(f"LLM analysis complete: {len(all_new_edges)} total edges to process")
    return all_new_edges


def validate_llm_edges(edges_response: List[Dict], source_id: str,
                      candidates: List[Dict], graph: Dict, 
                      logger: logging.Logger) -> List[Dict]:
    """
    Валидация рёбер, полученных от LLM.
    
    Args:
        edges_response: Ответ LLM (список рёбер)
        source_id: ID исходного узла
        candidates: Список кандидатов
        graph: Граф для проверки существования узлов
        logger: Logger
        
    Returns:
        Список валидных рёбер
    """
    # Собираем все ID узлов для проверки
    all_node_ids = {node["id"] for node in graph["nodes"]}
    candidate_ids = {c["node_id"] for c in candidates}
    
    # Допустимые типы рёбер
    valid_edge_types = {
        "PREREQUISITE", "ELABORATES", "EXAMPLE_OF", "HINT_FORWARD",
        "REFER_BACK", "PARALLEL", "TESTS", "REVISION_OF", "MENTIONS"
    }
    
    valid_edges = []
    
    for edge_data in edges_response:
        # Пропускаем записи с type: null
        if edge_data.get("type") is None:
            continue
        
        source = edge_data.get("source")
        target = edge_data.get("target")
        edge_type = edge_data.get("type")
        weight = edge_data.get("weight", 0.5)
        conditions = edge_data.get("conditions", "")
        
        # Базовая валидация
        if not all([source, target, edge_type]):
            logger.warning(f"Incomplete edge data: {edge_data}")
            continue
        
        # Проверка source (должен быть текущий узел)
        if source != source_id:
            logger.warning(f"Invalid source: expected {source_id}, got {source}")
            continue
        
        # Проверка target (должен быть в кандидатах)
        if target not in candidate_ids:
            logger.warning(f"Target {target} not in candidates")
            continue
        
        # Проверка типа
        if edge_type not in valid_edge_types:
            logger.warning(f"Invalid edge type: {edge_type}")
            continue
        
        # Проверка веса
        try:
            weight = float(weight)
            if not 0 <= weight <= 1:
                logger.warning(f"Weight out of range: {weight}")
                weight = 0.5
        except (ValueError, TypeError):
            logger.warning(f"Invalid weight: {weight}, using 0.5")
            weight = 0.5
        
        # Проверка PREREQUISITE self-loops
        if edge_type == "PREREQUISITE" and source == target:
            logger.warning(f"PREREQUISITE self-loop detected: {source}")
            continue
        
        # Добавляем валидное ребро
        valid_edges.append({
            "source": source,
            "target": target,
            "type": edge_type,
            "weight": weight,
            "conditions": conditions
        })
    
    return valid_edges


def update_graph_with_new_edges(graph: Dict, new_edges: List[Dict], 
                                logger: logging.Logger) -> Dict[str, int]:
    """
    Обновление графа новыми рёбрами от LLM согласно логике из ТЗ.
    
    Args:
        graph: Исходный граф
        new_edges: Список новых рёбер от LLM
        logger: Logger для отслеживания изменений
        
    Returns:
        Статистика изменений: {added, updated, replaced, total_processed}
    """
    stats = {
        "added": 0,
        "updated": 0,
        "replaced": 0,
        "self_loops_removed": 0,
        "total_processed": 0
    }
    
    # Создаем индекс существующих рёбер для быстрого поиска
    # Ключ: (source, target) -> список индексов рёбер
    edge_index = {}
    for i, edge in enumerate(graph["edges"]):
        key = (edge["source"], edge["target"])
        if key not in edge_index:
            edge_index[key] = []
        edge_index[key].append(i)
    
    # Обрабатываем каждое новое ребро
    for new_edge in new_edges:
        stats["total_processed"] += 1
        
        source = new_edge["source"]
        target = new_edge["target"]
        edge_type = new_edge["type"]
        weight = new_edge["weight"]
        
        key = (source, target)
        
        # Сценарий 1: Новое ребро (нет такого source+target)
        if key not in edge_index:
            # Добавляем с пометкой
            new_edge["conditions"] = "added_by=refiner_v1"
            graph["edges"].append(new_edge)
            stats["added"] += 1
            
            logger.debug(f"Added new edge: {source} -> {target} ({edge_type}, w={weight:.2f})")
            log_edge_operation(logger, "added", new_edge)
            # Обновляем индекс
            if key not in edge_index:
                edge_index[key] = []
            edge_index[key].append(len(graph["edges"]) - 1)
            
        else:
            # Есть существующее ребро/рёбра с таким source+target
            existing_indices = edge_index[key]
            
            # Ищем ребро с таким же типом
            same_type_idx = None
            for idx in existing_indices:
                if graph["edges"][idx]["type"] == edge_type:
                    same_type_idx = idx
                    break
            
            if same_type_idx is not None:
                # Сценарий 2: Дубликат (same source+target+type)
                existing_edge = graph["edges"][same_type_idx]
                old_weight = existing_edge.get("weight", 0.5)
                
                if weight > old_weight:
                    # Обновляем вес на больший
                    existing_edge["weight"] = weight
                    stats["updated"] += 1
                    
                    logger.debug(f"Updated weight: {source} -> {target} ({edge_type}), "
                               f"old={old_weight:.2f}, new={weight:.2f}")
                    log_edge_operation(logger, "updated", existing_edge, old_weight=old_weight)
                else:
                    logger.debug(f"Kept existing weight: {source} -> {target} ({edge_type}), "
                               f"existing={old_weight:.2f} >= new={weight:.2f}")
                    
            else:
                # Сценарий 3: Замена типа (same source+target, разный type)
                # Находим ребро с максимальным весом среди существующих
                max_weight_idx = None
                max_weight = -1
                
                for idx in existing_indices:
                    edge_weight = graph["edges"][idx].get("weight", 0.5)
                    if edge_weight > max_weight:
                        max_weight = edge_weight
                        max_weight_idx = idx
                
                # Заменяем только если новый вес >= старого
                if weight >= max_weight:
                    # Сохраняем старые рёбра для логирования
                    removed_edges = []
                    
                    # Удаляем все старые рёбра между этими узлами
                    # (сортируем индексы в обратном порядке для корректного удаления)
                    for idx in sorted(existing_indices, reverse=True):
                        old_edge = graph["edges"].pop(idx)
                        removed_edges.append(old_edge)
                        logger.debug(f"Removed old edge: {source} -> {target} "
                                   f"({old_edge['type']}, w={old_edge.get('weight', 0.5):.2f})")
                    # Добавляем новое ребро с пометкой
                    new_edge["conditions"] = "fixed_by=refiner_v1"
                    graph["edges"].append(new_edge)
                    stats["replaced"] += 1
                    
                    logger.debug(f"Replaced edge type: {source} -> {target}, "
                               f"new type={edge_type}, w={weight:.2f}")
                    log_edge_operation(logger, "replaced", new_edge, old_types=[e['type'] for e in removed_edges])
                    
                    # Пересоздаем индекс после изменений
                    edge_index = {}
                    for i, edge in enumerate(graph["edges"]):
                        key = (edge["source"], edge["target"])
                        if key not in edge_index:
                            edge_index[key] = []
                        edge_index[key].append(i)
                else:
                    logger.debug(f"Kept existing edge: {source} -> {target}, "
                               f"max existing weight={max_weight:.2f} > new={weight:.2f}")
    
    # Финальная очистка: удаляем PREREQUISITE self-loops
    edges_before = len(graph["edges"])
    graph["edges"] = [
        edge for edge in graph["edges"]
        if not (edge["type"] == "PREREQUISITE" and edge["source"] == edge["target"])
    ]
    
    self_loops_removed = edges_before - len(graph["edges"])
    if self_loops_removed > 0:
        stats["self_loops_removed"] = self_loops_removed
        logger.info(f"Removed {self_loops_removed} PREREQUISITE self-loops")
    
    # Логируем итоговую статистику
    logger.info(f"Graph update complete: added={stats['added']}, "
               f"updated={stats['updated']}, replaced={stats['replaced']}, "
               f"self-loops removed={stats['self_loops_removed']}")
    
    return stats


def main():
    """Основная функция."""
    logger = None  # Инициализируем переменную
    
    try:
        # Загрузка конфигурации
        config = load_config()
        refiner_config = config["refiner"]
        
        # Проверка run флага ДО настройки логирования
        if not refiner_config.get("run", True):
            # Простое логирование для случая run=false
            print("Refiner is disabled (run=false), copying file without changes")
            
            input_path = Path("data/out/LearningChunkGraph_dedup.json")
            output_path = Path("data/out/LearningChunkGraph.json")
            
            if not input_path.exists():
                print(f"ERROR: Input file not found: {input_path}")
                return EXIT_INPUT_ERROR
            
            shutil.copy2(input_path, output_path)
            print(f"Copied {input_path} to {output_path}")
            return EXIT_SUCCESS
        
        # Настройка JSON логирования только если run=true
        logger = setup_json_logging(refiner_config)
        
        # Валидация конфигурации
        try:
            validate_refiner_config(refiner_config)
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            return EXIT_CONFIG_ERROR
        
        # Пути к файлам
        input_path = Path("data/out/LearningChunkGraph_dedup.json")
        output_path = Path("data/out/LearningChunkGraph.json")
        
        # Загрузка и валидация графа
        try:
            graph = load_and_validate_graph(input_path)
            logger.info(f"Loaded graph with {len(graph.get('nodes', []))} nodes "
                       f"and {len(graph.get('edges', []))} edges", extra={
                "event": "graph_loaded",
                "nodes_count": len(graph.get('nodes', [])),
                "edges_count": len(graph.get('edges', []))
            })
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_path}")
            return EXIT_INPUT_ERROR
        except Exception as e:
            logger.error(f"Failed to load/validate graph: {e}")
            return EXIT_INPUT_ERROR
        
        # Извлечение целевых узлов
        target_nodes = extract_target_nodes(graph)
        logger.info(f"Found {len(target_nodes)} Chunk/Assessment nodes", extra={
            "event": "nodes_extracted",
            "target_nodes_count": len(target_nodes)
        })
        
        if not target_nodes:
            logger.warning("No Chunk/Assessment nodes found, saving graph without changes")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            return EXIT_SUCCESS
        
        # Построение индекса рёбер
        edges_index = build_edges_index(graph)
        
        # Генерация кандидатов для анализа
        try:
            # Получаем embeddings для всех узлов
            embeddings_dict = get_node_embeddings(target_nodes, refiner_config, logger)
            
            if not embeddings_dict:
                logger.warning("No embeddings obtained, saving graph without changes")
                validate_json(graph, "LearningChunkGraph")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(graph, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved graph to {output_path} (no changes)")
                return EXIT_SUCCESS
            
            # Строим FAISS индекс
            faiss_index, node_ids_list = build_similarity_index(
                embeddings_dict, target_nodes, refiner_config, logger
            )
            
            # Генерируем пары кандидатов
            candidate_pairs = generate_candidate_pairs(
                target_nodes, embeddings_dict, faiss_index, node_ids_list,
                edges_index, refiner_config, logger
            )
            
            if not candidate_pairs:
                logger.info("No candidate pairs found above similarity threshold")
            else:
                logger.info(f"Found {len(candidate_pairs)} nodes with candidates for analysis")
                
                # Анализ через LLM
                try:
                    new_edges = analyze_candidate_pairs(
                        candidate_pairs, graph, refiner_config, logger
                    )
                    
                    # Обновление графа с новыми рёбрами
                    if new_edges:
                        logger.info(f"LLM analysis returned {len(new_edges)} edges to process")
                        update_stats = update_graph_with_new_edges(graph, new_edges, logger)
                        logger.info(f"Graph updated: {update_stats['added']} added, "
                                   f"{update_stats['updated']} updated, "
                                   f"{update_stats['replaced']} replaced", extra={
                            "event": "graph_updated",
                            "stats": update_stats
                        })
                    else:
                        logger.info("No new edges found by LLM analysis")
                        
                except Exception as e:
                    logger.error(f"LLM analysis failed: {e}")
                    if "RateLimitError" in str(e):
                        return EXIT_API_LIMIT_ERROR
                    return EXIT_RUNTIME_ERROR
            
        except Exception as e:
            logger.error(f"Error during candidate generation: {e}")
            return EXIT_RUNTIME_ERROR
        
        # Временно: сохраняем граф без изменений
        # TODO: После анализа LLM здесь будет обновленный граф
        
        # Финальная валидация
        try:
            validate_json(graph, "LearningChunkGraph")
            validate_graph_invariants(graph)
        except Exception as e:
            logger.error(f"Graph validation failed: {e}")
            # Сохраняем проблемный граф для анализа
            failed_path = Path("data/out/LearningChunkGraph_refiner_failed.json")
            with open(failed_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            logger.error(f"Saved failed graph to {failed_path}")
            return EXIT_RUNTIME_ERROR
        
        # Сохранение результата
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully saved refined graph to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            return EXIT_IO_ERROR
        
        # Логирование завершения
        logger.info("Refiner completed successfully", extra={
            "event": "refiner_complete",
            "edges_added": update_stats.get('added', 0) if 'update_stats' in locals() else 0,
            "edges_updated": update_stats.get('updated', 0) if 'update_stats' in locals() else 0,
            "edges_replaced": update_stats.get('replaced', 0) if 'update_stats' in locals() else 0
        })
        
        return EXIT_SUCCESS
        
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error: {e}", exc_info=True, extra={
                "event": "refiner_error",
                "error": str(e)
            })
        else:
            # Если logger не инициализирован, используем print
            print(f"ERROR: Unexpected error: {e}")
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())