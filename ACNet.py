import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from Actor.data_small import G, edge_index

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择


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
        x = torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x
        start_feature_masked = torch.tensor(start_feature_masked, dtype=torch.float32) if isinstance(
            start_feature_masked, np.ndarray) else start_feature_masked
        target_feature_masked = torch.tensor(target_feature_masked, dtype=torch.float32) if isinstance(
            target_feature_masked, np.ndarray) else target_feature_masked
        other_goals = torch.tensor(other_goals, dtype=torch.float32) if isinstance(other_goals,
                                                                                   np.ndarray) else other_goals
        other_pos = torch.tensor(other_pos, dtype=torch.float32) if isinstance(other_pos, np.ndarray) else other_pos
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


# Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=256):
        super(Critic, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim  # 单个智能体状态维度
        self.action_dim = action_dim  # 单个智能体动作维度

        # 全连接层
        # 输入维度为所有智能体的状态和动作的拼接
        self.fc1 = nn.Linear(state_dim * num_agents * 4 + action_dim * num_agents * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出Q值

    def forward(self, states, actions):
        """
        前向传播
        states: [batch_size, num_agents, state_dim]
        actions: [batch_size, num_agents, action_dim]
        """
        # 展平状态和动作
        states = states.view(states.size(0), -1)  # [batch_size, num_agents * state_dim]
        actions = actions.view(actions.size(0), -1)  # [batch_size, num_agents * action_dim]
        # 拼接状态和动作
        x = torch.cat([states, actions], dim=1)  # [batch_size, num_agents*(state_dim + action_dim)]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)  # [batch_size, 1]
        return q


# Actor网络，继承自PathPredictor
class Actor(PathPredictor):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__(input_dim, hidden_dim, output_dim)
        # 可以在这里添加额外的层或修改网络结构

    def forward_actor(self, node_features, observation):
        """
        Actor的前向传播，用于生成动作
        node_features: [num_nodes, feature_dim]
        observation: [num_agents, 4, num_nodes]
        """
        # 确保Actor处于训练模式
        self.train()
        # 转换为张量并移动到设备
        node_features = node_features.to(DEVICE)
        # observation = torch.tensor(observation, dtype=torch.float32).to(DEVICE)

        # 使用PathPredictor的前向传播
        action_probs = self.forward(node_features, observation)  # [num_nodes, output_dim]
        # 选择每个智能体的动作
        # actions = torch.argmax(action_probs, dim=1)  # [num_nodes]
        return action_probs
