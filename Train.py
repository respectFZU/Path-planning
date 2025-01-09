import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from Actor.data_small import G, edge_index
from Actor.embeding import node_features
from MAPF_dl.ACNet import Actor, Critic
from GraphEnv_new import SimpleMAPFEnv

# 超参数设置
NUM_EPISODES = 1000  # 训练的回合数
MAX_STEPS = 100  # 每个回合的最大步数
GAMMA = 0.99  # 折扣因子
LR_ACTOR = 1e-4  # Actor的学习率
LR_CRITIC = 1e-3  # Critic的学习率
BUFFER_SIZE = 100000  # 经验回放缓冲区大小
BATCH_SIZE = 100  # 批量大小
TAU = 1e-3  # 软更新参数
JUMP_K = 10  # 观测时的跳跃参数

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)  # 使用双端队列存储经验
        self.batch_size = batch_size

    def push(self, state, actions, rewards, next_state, done):
        """将一个经验元组添加到缓冲区"""
        self.memory.append((state, actions, rewards, next_state, done))

    def sample(self):
        """随机采样一个批次的经验元组"""
        batch = random.sample(self.memory, self.batch_size)
        state, actions, rewards, next_state, done = map(np.array, zip(*batch))
        return state, actions, rewards, next_state, done

    def __len__(self):
        """返回缓冲区中当前的经验数量"""
        return len(self.memory)


# 初始化Actor和Critic
def initialize_networks():
    # 假设node_features和edge_index已经加载
    num_nodes = G.number_of_nodes()
    input_dim = node_features.size(1)
    hidden_dim = 128
    output_dim = num_nodes  # 输出维度等于节点数量

    # 创建Actor，加载预训练的PathPredictor权重
    actor = Actor(input_dim, hidden_dim, output_dim).to(DEVICE)
    actor.load_state_dict(torch.load('GraphS2.pth'))  # 加载预训练模型
    actor.train()  # 设置为训练模式

    # 定义Actor优化器
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)

    # 创建Critic
    state_dim = 4 * num_nodes  # 观测空间维度（4个通道，每个节点4个特征）
    action_dim = num_nodes  # 动作空间维度（节点数量）
    num_agents = 3  # 智能体数量

    critic = Critic(state_dim, action_dim, num_agents).to(DEVICE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # 创建目标Critic网络并初始化权重
    critic_target = Critic(state_dim, action_dim, num_agents).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.eval()  # 设置为评估模式

    return actor, actor_optimizer, critic, critic_optimizer, critic_target


# 初始化网络
actor, actor_optimizer, critic, critic_optimizer, critic_target = initialize_networks()

# 初始化经验回放缓冲区
memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

# 环境初始化
# AGV_data_bag = [
#     {'name': 1, 'start_id': 3, 'goal_id': 22},
#     {'name': 2, 'start_id': 2, 'goal_id': 0},
#     {'name': 3, 'start_id': 6, 'goal_id': 13}
# ]
AGV_data_bag = [
    {'name': 1, 'start_id': 0, 'goal_id': 22},
    {'name': 2, 'start_id': 17, 'goal_id': 3},
    {'name': 3, 'start_id': 16, 'goal_id': 19}
]

env = SimpleMAPFEnv(graph=G, AGV_bag=AGV_data_bag, num_nodes=26, num_agents=3)


# 工具函数
def preprocess_states(states, node_features):
    """
    将环境观测转换为适合GraphSAGE的输入格式
    参数：
        states (numpy.ndarray): 环境观测，形状为 [num_agents, 4, 26]
        node_features (torch.Tensor): 节点特征，形状为 [26, feature_dim]
    返回：
        processed_states (torch.Tensor): 处理后的观测，形状为 [num_agents, 4, 26, feature_dim]
    """
    # 将环境观测转换为浮点型张量并移动到设备
    states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)  # [num_agents, 4, 26]

    # 确保 node_features 在设备上
    node_features = node_features.to(DEVICE)  # [26, feature_dim]

    # 初始化 processed_states 为零张量
    num_agents, num_channels, num_nodes = states_tensor.shape
    feature_dim = node_features.shape[1]
    processed_states = torch.zeros((num_agents, num_channels, num_nodes, feature_dim), dtype=torch.float32).to(DEVICE)

    # 遍历每个智能体和通道
    for agent_idx in range(num_agents):
        for channel_idx in range(num_channels):
            for node_idx in range(num_nodes):
                if states_tensor[agent_idx, channel_idx, node_idx] != 0:
                    # 提取 node_features 中的特征并赋值
                    processed_states[agent_idx, channel_idx, node_idx] = node_features[node_idx]

    return processed_states


def preprocess_states_batch(states, node_features):
    """
    将环境观测转换为适合GraphSAGE的输入格式
    参数：
        states (numpy.ndarray): 环境观测，形状为 [batch_size, num_agents, 4, 26]
        node_features (torch.Tensor): 节点特征，形状为 [26, feature_dim]
    返回：
        processed_states (torch.Tensor): 处理后的观测，形状为 [batch_size, num_agents, 4, 26, feature_dim]
    """
    # 将环境观测转换为浮点型张量并移动到设备
    states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)  # [batch_size, num_agents, 4, 26]

    # 确保 node_features 在设备上
    node_features = node_features.to(DEVICE)  # [26, feature_dim]

    # 初始化 processed_states 为零张量
    batch_size, num_agents, num_channels, num_nodes = states_tensor.shape
    feature_dim = node_features.shape[1]
    processed_states = torch.zeros((batch_size, num_agents, num_channels, num_nodes, feature_dim),
                                   dtype=torch.float32).to(DEVICE)

    # 遍历每个批次、智能体和通道
    for batch_idx in range(batch_size):
        for agent_idx in range(num_agents):
            for channel_idx in range(num_channels):
                for node_idx in range(num_nodes):
                    if states_tensor[batch_idx, agent_idx, channel_idx, node_idx] != 0:
                        # 提取 node_features 中的特征并赋值
                        processed_states[batch_idx, agent_idx, channel_idx, node_idx] = node_features[node_idx]

    return processed_states


def preprocess_actions(actions_one_hot, node_features):
    """
    将动作的 One-hot 编码转换为适合 GraphSAGE 的输入格式
    参数：
        actions_one_hot (torch.Tensor): 动作的 One-hot 编码，形状为 [batch_size, num_agents, num_nodes]
        node_features (torch.Tensor): 节点特征，形状为 [num_nodes, feature_dim]
    返回：
        processed_actions (torch.Tensor): 处理后的动作，形状为 [batch_size, num_agents, num_nodes, feature_dim]
    """
    # 确保 actions_one_hot 和 node_features 在设备上
    actions_one_hot = actions_one_hot.to(DEVICE)
    node_features = node_features.to(DEVICE)

    # 获取维度信息
    batch_size, num_agents, num_nodes = actions_one_hot.shape
    feature_dim = node_features.shape[1]

    # 初始化 processed_actions 为零张量
    processed_actions = torch.zeros((batch_size, num_agents, num_nodes, feature_dim), dtype=torch.float32).to(DEVICE)

    # 遍历每个批次和智能体
    for batch_idx in range(batch_size):
        for agent_idx in range(num_agents):
            # 获取当前智能体的动作 One-hot 编码
            action_one_hot = actions_one_hot[batch_idx, agent_idx]

            # 找到非零索引
            non_zero_indices = torch.nonzero(action_one_hot, as_tuple=True)[0]

            # 为每个非零索引提取对应的节点特征
            for node_idx in non_zero_indices:
                processed_actions[batch_idx, agent_idx, node_idx] = node_features[node_idx]

    return processed_actions


# 训练循环
for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()  # 重置环境，获取初始观测
    state = preprocess_states(state, node_features)  # 转换为张量
    total_rewards = np.zeros(env.num_agents)  # 记录总奖励

    for step in range(MAX_STEPS):
        actions = []
        # 对每个智能体分别选择动作
        for agent_idx in range(env.num_agents):
            # 获取当前智能体的观测 [4, 26, 4]
            agent_observation = state[agent_idx].cpu().numpy()
            # 使用Actor选择动作
            action_probs = actor.forward_actor(node_features, agent_observation)
            agent_pos = env.world.getPos(agent_idx + 1)
            valid_action = env.get_valid_actions(agent_idx + 1)
            EPSILON = 0.2
            # ε-greedy 策略
            if random.random() < EPSILON:
                # 随机选择一个动作（探索）
                if random.random() < EPSILON / 2:
                    action = agent_pos
                else:
                    action = random.choice(valid_action)
            else:
                # 选择最优动作（利用）
                action = action_probs[agent_pos].argmax().item()

            # action = action_probs[agent_pos].argmax().item()
            actions.append(action)

        # 执行动作，与环境交互
        next_state, rewards, done, infos = env.step(actions)
        total_rewards += rewards  # 累积奖励
        Total_R = total_rewards

        # 预处理下一状态
        next_state_tensor = preprocess_states(next_state, node_features)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(DEVICE)

        # 将经验元组存入经验回放缓冲区
        memory.push(state.cpu().numpy(), actions, rewards, next_state, done)
        if done:
            state = env.reset()
            state = preprocess_states(state, node_features)
            total_rewards = np.zeros(env.num_agents)
        else:
            state = next_state_tensor  # 更新当前状态

        # 如果缓冲区中的经验超过批量大小，进行学习
        if len(memory) > BATCH_SIZE:
            # 采样一个批次的经验
            states, actions_batch, rewards_batch, next_states, dones = memory.sample()
            states = torch.tensor(states, dtype=torch.float32).to(
                DEVICE)  # [batch_size, num_agents, 4, num_nodes,input_feature]
            actions_batch = torch.tensor(actions_batch, dtype=torch.long).to(DEVICE)  # [batch_size, num_agents]
            rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(DEVICE)  # [batch_size, num_agents]
            next_states = torch.tensor(next_states, dtype=torch.float32).to(
                DEVICE)  # [batch_size, num_agents, 4, num_nodes]
            dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)  # [batch_size]

            # 对动作进行One-hot编码
            actions_one_hot = F.one_hot(actions_batch.view(-1), num_classes=G.number_of_nodes()).float()
            actions_one_hot = actions_one_hot.view(BATCH_SIZE, env.num_agents, -1).to(DEVICE)
            actions_one_hot = preprocess_actions(actions_one_hot, node_features)

            # ---------------------------- Critic 更新 ---------------------------- #
            with torch.no_grad():
                # 使用Actor选择下一状态的动作
                next_states = preprocess_states_batch(next_states,
                                                      node_features)  # [batch_size, num_agents, 4, 26, feature_dim]

                # 初始化 next_actions
                next_actions = torch.zeros((BATCH_SIZE, env.num_agents), dtype=torch.long).to(DEVICE)

                # 遍历每个智能体
                for agent_idx in range(env.num_agents):
                    # 提取当前智能体的状态
                    agent_next_states = next_states[:, agent_idx, :, :, :]  # [batch_size, 4, 26, feature_dim]

                    # 对每个批次样本计算动作
                    for batch_idx in range(BATCH_SIZE):
                        # 获取当前批次样本的智能体状态
                        agent_observation = agent_next_states[batch_idx].cpu().numpy()

                        # 使用Actor选择动作
                        action_probs = actor.forward_actor(node_features, agent_observation)
                        action = action_probs[env.world.getPos(agent_idx + 1)].argmax().item()

                        # 存储动作
                        next_actions[batch_idx, agent_idx] = action

                # 对 next_actions 进行 One-hot 编码
                next_actions_one_hot = F.one_hot(next_actions, num_classes=G.number_of_nodes()).float().to(DEVICE)
                next_actions_one_hot = preprocess_actions(next_actions_one_hot, node_features)

                # 使用目标Critic网络计算下一状态的Q值
                q_next = critic_target(next_states, next_actions_one_hot)  # [batch_size, 1]

                # 计算目标Q值
                rewards_sum = rewards_batch.sum(dim=1, keepdim=True)  # [batch_size, 1]
                q_target = rewards_sum + GAMMA * q_next * (1 - dones.view(-1, 1))  # [batch_size, 1]

            # 使用当前Critic网络计算现有的Q值
            q_current = critic(states, actions_one_hot)  # [batch_size, 1]
            # 计算Critic的损失（均方误差）
            critic_loss = F.mse_loss(q_current, q_target)

            # 反向传播并优化Critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # ---------------------------- Actor 更新 ---------------------------- #
            # 初始化 actions_pred
            actions_pred = torch.zeros((BATCH_SIZE, env.num_agents), dtype=torch.long).to(DEVICE)

            # 遍历每个智能体
            for agent_idx in range(env.num_agents):
                # 提取当前智能体的状态
                agent_states = states[:, agent_idx, :, :, :]  # [batch_size, 4, 26, feature_dim]

                # 对每个批次样本计算动作
                for batch_idx in range(BATCH_SIZE):
                    # 获取当前批次样本的智能体状态
                    agent_observation = agent_states[batch_idx].cpu().numpy()

                    # 使用Actor选择动作
                    action_probs = actor.forward_actor(node_features, agent_observation)
                    action = action_probs[env.world.getPos(agent_idx + 1)].argmax().item()

                    # 存储动作
                    actions_pred[batch_idx, agent_idx] = action

            # 对 actions_pred 进行 One-hot 编码
            actions_pred_one_hot = F.one_hot(actions_pred, num_classes=G.number_of_nodes()).float().to(DEVICE)
            actions_pred_one_hot = preprocess_actions(actions_pred_one_hot, node_features)

            # 计算Actor的损失（最大化Critic的Q值）
            actor_loss = -critic(states, actions_pred_one_hot).mean()

            # 反向传播并优化Actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # ------------------------ 软更新 Critic 目标网络 -------------------- #
            for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        # 输出每个回合的总奖励
        print(f"第 {episode} 回合 | 总奖励: {Total_R.sum()}")
        Total_R = 0
        episode = episode + 1

        # 每100个回合保存一次模型
        if episode % 100 == 0:
            torch.save(actor.state_dict(), f'actor_episode_{episode}.pth')
            torch.save(critic.state_dict(), f'critic_episode_{episode}.pth')

    # 训练结束后保存最终模型
    torch.save(actor.state_dict(), 'actor_final.pth')
    torch.save(critic.state_dict(), 'critic_final.pth')
