#!/usr/bin/env python3
"""Quick test to verify metrics were added to graph nodes."""

import json
from pathlib import Path

# Load the enriched graph
graph_file = Path("viz/data/out/LearningChunkGraph_wow.json")
if not graph_file.exists():
    print(f"ERROR: File not found: {graph_file}")
    exit(1)

with open(graph_file, "r", encoding="utf-8") as f:
    graph_data = json.load(f)

# Define expected metrics
metrics = ['degree_in', 'degree_out', 'degree_centrality', 
           'pagerank', 'betweenness_centrality', 'closeness_centrality']

# Check first node
print("=== First Node Check ===")
first_node = graph_data['nodes'][0]
print(f"ID: {first_node['id']}")
print(f"Type: {first_node['type']}")
print("\nMetrics:")
for metric in metrics:
    if metric in first_node:
        value = first_node[metric]
        if isinstance(value, float):
            print(f"  ✓ {metric}: {value:.6f}")
        else:
            print(f"  ✓ {metric}: {value}")
    else:
        print(f"  ✗ {metric}: MISSING")

# Overall statistics
print("\n=== Overall Statistics ===")
nodes_with_all = sum(1 for node in graph_data['nodes'] 
                     if all(m in node for m in metrics))
print(f"Total nodes: {len(graph_data['nodes'])}")
print(f"Nodes with all metrics: {nodes_with_all}")

if nodes_with_all == len(graph_data['nodes']):
    print("✓ SUCCESS: All nodes have all metrics!")
else:
    print(f"✗ WARNING: {len(graph_data['nodes']) - nodes_with_all} nodes missing metrics")

# PageRank analysis
pageranks = [node.get('pagerank', 0) for node in graph_data['nodes']]
if pageranks:
    print(f"\n=== PageRank Analysis ===")
    print(f"Min: {min(pageranks):.6f}")
    print(f"Max: {max(pageranks):.6f}")
    print(f"Avg: {sum(pageranks)/len(pageranks):.6f}")
    
    # Top 3 nodes
    top_nodes = sorted(graph_data['nodes'], 
                      key=lambda x: x.get('pagerank', 0), 
                      reverse=True)[:3]
    print("\nTop-3 nodes by PageRank:")
    for i, node in enumerate(top_nodes, 1):
        node_id = node['id']
        if len(node_id) > 50:
            node_id = node_id[:47] + "..."
        print(f"  {i}. {node_id} (PR={node['pagerank']:.4f})")

# Check for NaN/inf values
print("\n=== Data Quality Check ===")
import math
issues = []
for node in graph_data['nodes']:
    for metric in metrics:
        if metric in node:
            value = node[metric]
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    issues.append(f"Node {node['id']}: {metric}={value}")

if issues:
    print(f"✗ Found {len(issues)} NaN/inf values:")
    for issue in issues[:5]:  # Show first 5
        print(f"  - {issue}")
else:
    print("✓ No NaN/inf values found")

print("\n=== Test Complete ===")
