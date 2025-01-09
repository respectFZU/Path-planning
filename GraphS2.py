import time
import torch
from torch import nn
from torch_geometric.nn import SAGEConv
import networkx as nx
from Actor.data_small import G, edge_index
import torch.nn.functional as F
import random
from tqdm import tqdm
import json
from Actor.embeding import node_features


def TransToObs(start, end, obs_goals, obs_positions):  # start\end为点，goals/positions是矩阵，维度为input_dim*node_num
    start_feature = torch.zeros_like(node_features)
    end_feature = torch.zeros_like(node_features)
    start_feature[start] = node_features[start]
    end_feature[end] = node_features[end]
    Observation = torch.cat((start_feature.unsqueeze(0),
                             end_feature.unsqueeze(0),
                             obs_goals.unsqueeze(0),
                             obs_positions.unsqueeze(0)), dim=0)
    return Observation
# start_feature_extracted = Observation[0]
# end_feature_extracted = Observation[1]
# obs_goals_extracted = Observation[2]
# obs_positions_extracted = Observation[3]

class PathPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PathPredictor, self).__init__()
        self.conv1 = SAGEConv(input_dim * 5, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim * 2, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # 初始化权重
        nn.init.kaiming_normal_(self.conv1.lin_l.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.lin_l.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.lin_l.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')

    def forward(self, x, observation):

        # 创建与 x 形状相同的零张量
        start_feature_masked = observation[0]
        target_feature_masked = observation[1]
        other_goals = observation[2]
        other_pos = observation[3]
        x = torch.cat([x, start_feature_masked, target_feature_masked, other_goals, other_pos], dim=1)
        x = self.conv1(x, edge_index).relu()
        x = self.ln1(x)

        target = (target_feature_masked != 0).any(dim=1).nonzero(as_tuple=True)[0].item()
        target_node_feature = x[target].unsqueeze(0).repeat(x.size(0), 1)
        x = torch.cat([x, target_node_feature], dim=1)
        x = self.conv2(x, edge_index).relu()
        x = self.ln2(x)
        x = self.conv3(x, edge_index).relu()
        x = self.fc(x)
        probabilities = F.softmax(x, dim=1)

        masked_probabilities = torch.zeros_like(probabilities)
        for node in range(probabilities.size(0)):
            if node in G:
                neighbors = list(G.neighbors(node))
                neighbors.append(node)
                neighbors = [int(n) for n in neighbors if int(n) < probabilities.size(1)]
                mask = torch.zeros_like(probabilities[node])
                mask[neighbors] = 1.0
                masked_probabilities[node] = probabilities[node] * mask
                if masked_probabilities[node].sum() > 0:
                    masked_probabilities[node] /= masked_probabilities[node].sum()

        return masked_probabilities


# 初始化模型
other_targets = torch.zeros_like(node_features, requires_grad=True)
other_positions = torch.zeros_like(node_features, requires_grad=True)
model = PathPredictor(input_dim=node_features.size(1), hidden_dim=128, output_dim=G.number_of_nodes())

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# 确保 edge_index 的形状正确
print(edge_index.shape)

# 记录训练开始时间
start_time = time.time()

# 训练模型
num_epochs = 30
node_pairs = [(i, j) for i in range(G.number_of_nodes()) for j in range(G.number_of_nodes()) if i != j]
random.shuffle(node_pairs)

# 保存每个 epoch 的准确率指标
accuracy_metrics = []

for epoch in range(num_epochs):
    error_paths = []
    for start_node, target_node in tqdm(node_pairs, desc=f"Epoch {epoch + 1}"):
        try:
            # 使用整数索引
            shortest_path = nx.dijkstra_path(G, source=start_node, target=target_node)

            model.train()
            current_node = start_node
            valid_path = True

            while current_node != target_node:
                optimizer.zero_grad()
                observation = TransToObs(current_node, target_node, other_targets, other_positions)
                output = model(node_features, observation)
                # print(output)
                next_node = output[current_node].argmax().item()

                correct_node = shortest_path[shortest_path.index(current_node) + 1]
                if next_node == correct_node:
                    current_node = next_node
                else:
                    valid_path = False
                    error_paths.append((start_node, target_node, current_node))

                    num_classes = output.size(1)

                    # 获取当前节点的输出
                    outputs = output[current_node].unsqueeze(0)

                    # 创建 one-hot 编码的标签
                    one_hot_labels = torch.zeros(num_classes)

                    # 确保 correct_node 在范围内
                    if correct_node < num_classes:
                        one_hot_labels[correct_node] = 1.0
                    else:
                        print(f"Error: Correct node {correct_node} is out of bounds.")
                        break

                    # 将 one-hot 标签转换为整数标签
                    integer_labels = torch.argmax(one_hot_labels).unsqueeze(0)

                    # 使用整数标签计算交叉熵损失
                    step_loss = F.cross_entropy(outputs, integer_labels)

                    # 反向传播和优化
                    step_loss.backward()
                    optimizer.step()
                    break

            if valid_path:
                print(f"Epoch {epoch + 1}, Start {start_node}, Target {target_node}, Path completed")

        except nx.NetworkXNoPath:
            print(f"No path from {start_node} to {target_node}")
        except nx.NodeNotFound as e:
            print(e)

    scheduler.step()

    # 保存模型
    torch.save(model.state_dict(), 'GraphS2.pth')

    # 从第十个 epoch 开始运行测试代码
    if epoch >= 11:

        # 加载模型
        model.load_state_dict(torch.load('GraphS2.pth'))
        model.eval()

        # 测试模型并计算准确率
        correct_count = 0
        total_count = 0
        shortest_path_count = 0
        non_shortest_path_count = 0
        invalid_path_count = 0
        incomplete_path_count = 0

        results = []

        for start_node in tqdm(range(G.number_of_nodes()), desc="Testing"):
            for target_node in range(G.number_of_nodes()):
                if start_node == target_node:
                    continue

                try:
                    correct_path = nx.dijkstra_path(G, source=start_node, target=target_node)

                    with torch.no_grad():
                        predicted_path = [start_node]
                        current_node = start_node
                        valid_path = True
                        visited = set()

                        while current_node != target_node:
                            observation = TransToObs(current_node, target_node, other_targets, other_positions)
                            output = model(node_features, observation)
                            next_node = output[current_node].argmax().item()

                            if current_node in correct_path and next_node == correct_path[
                                correct_path.index(current_node) + 1]:
                                current_node = next_node
                                predicted_path.append(current_node)
                            else:
                                if G.has_edge(current_node, next_node):
                                    if next_node in visited:
                                        valid_path = False
                                        print(f"循环检测失败: {current_node} -> {next_node}")
                                        break
                                    predicted_path.append(next_node)
                                    visited.add(next_node)
                                    current_node = next_node
                                    if len(predicted_path) - len(correct_path) > 3:
                                        print(f"路径过长失败: {predicted_path}")
                                        break
                                else:
                                    valid_path = False
                                    invalid_path_count += 1
                                    print(f"无效边失败: {current_node} -> {next_node}")
                                    break

                        if current_node != target_node:
                            valid_path = False
                            incomplete_path_count += 1
                            print(f"路径不完整失败: {predicted_path}")

                    if valid_path and current_node == target_node:
                        correct_count += 1
                        if predicted_path == correct_path:
                            shortest_path_count += 1
                            print(f"起点 {start_node} 到终点 {target_node} 最短路径预测成功")
                        else:
                            non_shortest_path_count += 1
                            print(f"起点 {start_node} 到终点 {target_node} 非最短路径预测成功")

                    total_count += 1

                    results.append({
                        "start_node": start_node,
                        "target_node": target_node,
                        "correct_path": correct_path,
                        "predicted_path": predicted_path,
                        "valid_path": valid_path
                    })

                    print(f"Target node: {target_node}")
                    print(f"Correct path: {correct_path}")
                    print(f"Predicted path: {predicted_path}")

                except nx.NetworkXNoPath:
                    print(f"No path from {start_node} to {target_node}")
                except nx.NodeNotFound as e:
                    print(e)

        accuracy = correct_count / total_count
        print(f"Epoch {epoch + 1} Accuracy: {accuracy:.2f}")
        print(f"Number of correct paths: {correct_count}")
        print(f"Number of shortest paths: {shortest_path_count}")
        print(f"Number of non-shortest paths: {non_shortest_path_count}")
        print(f"Number of invalid paths: {invalid_path_count}")
        print(f"Number of incomplete paths: {incomplete_path_count}")

        # 保存当前 epoch 的准确率
        accuracy_metrics.append({
            "epoch": epoch + 1,
            "accuracy": accuracy,
            "correct_paths": correct_count,
            "shortest_paths": shortest_path_count,
            "non_shortest_paths": non_shortest_path_count,
            "invalid_paths": invalid_path_count,
            "incomplete_paths": incomplete_path_count
        })
        if shortest_path_count == 650:
            break

# 记录训练结束时间
end_time = time.time()

# 计算并打印总耗时
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")

# 保存所有 epoch 的准确率指标
with open('accuracy_metrics.json', 'w') as f:
    json.dump(accuracy_metrics, f, indent=4)
