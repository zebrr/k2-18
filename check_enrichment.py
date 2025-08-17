#!/usr/bin/env python3
import json

# Проверим demo_path в графе
with open('viz/data/out/LearningChunkGraph_wow.json', 'r', encoding='utf-8') as f:
    graph = json.load(f)

print("=== ПРОВЕРКА ENRICHMENT ===\n")

# Проверяем demo_path
if '_meta' in graph and 'demo_path' in graph['_meta']:
    demo_path = graph['_meta']['demo_path']
    print(f"✅ Demo path найден: {len(demo_path)} узлов")
    print(f"   Первые 5 узлов: {demo_path[:5]}")
    
    # Проверяем конфигурацию генерации
    if 'demo_generation_config' in graph['_meta']:
        config = graph['_meta']['demo_generation_config']
        print(f"   Стратегия: {config.get('strategy')}, max_nodes: {config.get('max_nodes')}")
else:
    print("❌ Demo path не найден в _meta")

# Проверяем concepts поле в узлах
nodes_with_concepts = [n for n in graph['nodes'] if 'concepts' in n and n['concepts']]
print(f"\n✅ Узлов с заполненным полем concepts: {len(nodes_with_concepts)}/{len(graph['nodes'])}")
if nodes_with_concepts:
    print(f"   Пример: узел '{nodes_with_concepts[0]['id']}' связан с {nodes_with_concepts[0]['concepts']}")

# Проверим mention_index в словаре концептов
with open('viz/data/out/ConceptDictionary_wow.json', 'r', encoding='utf-8') as f:
    concepts = json.load(f)

if '_meta' in concepts and 'mention_index' in concepts['_meta']:
    mention_index = concepts['_meta']['mention_index']
    print(f"\n✅ Mention index найден: {len(mention_index)} концептов упоминаются")
    # Показываем топ-3 по упоминаниям
    sorted_mentions = sorted(mention_index.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
    print("   Топ-3 упоминаемых концептов:")
    for concept_id, data in sorted_mentions:
        print(f"   - {concept_id}: {data['count']} упоминаний")
else:
    print("\n❌ Mention index не найден в _meta словаря")

# Проверка handle_large_graph - если граф был большой
if '_meta' in graph and 'graph_metadata' in graph['_meta']:
    metadata = graph['_meta']['graph_metadata']
    print(f"\n✅ Граф был отфильтрован:")
    print(f"   Оригинал: {metadata.get('original_nodes')} узлов")
    print(f"   Отфильтровано до: {metadata.get('displayed_nodes')} узлов")
else:
    print(f"\n✅ Граф не требовал фильтрации ({len(graph['nodes'])} узлов)")
