import json
import networkx as nx
import math
import torch

# 读取JSON文件
with open('../Data/1022.json', 'r') as file:
    data = json.load(file)

# 创建一个无向图
G = nx.Graph()

# 解析节点和边
for node in data["nodes"]:
    node_id = int(node["id"])  # 确保节点ID是整数
    x, y = node["coordinate"]["x"], node["coordinate"]["y"]

    # 添加节点到图中
    G.add_node(node_id, pos=(x, y))

    # 添加边到图中
    for edge in node["edges"]:
        destination = int(edge["destination"])  # 确保目标节点ID是整数
        weight = edge["weight"]
        G.add_edge(node_id, destination, weight=weight)

# 获取并排序节点列表
nodes = sorted(G.nodes)
print("Sorted Nodes:", nodes)

# 创建节点到索引的映射，从 0 开始
node_to_index = {node: idx for idx, node in enumerate(nodes)}
print("Node to Index Mapping:", node_to_index)

# 使用映射生成 edge_index
edges = [(node_to_index[u], node_to_index[v]) for u, v in G.edges]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(edge_index)
# 打印 edge_index 的最大值和形状
print("Edge Index Shape:", edge_index.shape)
print("Max Index in Edge Index:", edge_index.max().item())
print("Number of Nodes:", len(nodes))

# 计算所有边的距离
distances = []
for u, v in G.edges():
    x1, y1 = G.nodes[u]['pos']
    x2, y2 = G.nodes[v]['pos']
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distances.append(distance)

# 找到最大距离用于归一化
max_distance = max(distances)

# 更新权重为归一化后的距离
for u, v in G.edges():
    x1, y1 = G.nodes[u]['pos']
    x2, y2 = G.nodes[v]['pos']
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    normalized_weight = (distance / max_distance) * 100  # 归一化
    G[u][v]['weight'] = normalized_weight

# 输出每条边的权重
# for u, v, data in G.edges(data=True):
#     print(f"Edge from {u} to {v} has weight {data['weight']}")