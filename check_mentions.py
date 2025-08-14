#!/usr/bin/env python
import json

# Загрузить граф
with open('/Users/askold.romanov/code/k2-18/data/out/LearningChunkGraph_longrange.json', 'r') as f:
    graph = json.load(f)

# Проверить веса MENTIONS edges
mentions_edges = [e for e in graph["edges"] if e["type"] == "MENTIONS"]
mentions_weights = [e["weight"] for e in mentions_edges]

# Получить уникальные веса
unique_weights = set(mentions_weights)

print(f"Total MENTIONS edges: {len(mentions_edges)}")
print(f"Unique weights: {sorted(unique_weights)}")
print(f"Min weight: {min(mentions_weights)}")
print(f"Max weight: {max(mentions_weights)}")

# Показать несколько примеров
print("\nПримеры MENTIONS edges:")
for edge in mentions_edges[:5]:
    print(f"  {edge['source']} -> {edge['target']}, weight={edge['weight']}")

# Проверить автосгенерированные
auto_mentions = [e for e in mentions_edges if e.get('conditions') == 'auto_generated']
print(f"\nAuto-generated MENTIONS: {len(auto_mentions)}")
