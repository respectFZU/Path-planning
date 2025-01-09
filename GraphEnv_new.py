import json
import math
import random
import gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gym import spaces

# 奖励塑造
Move_Cost, Stay_Cost, Collision_Cost, Wrong_Cost, Blocking_Cost = -0.3, -0.5, -1, -2, -1
Goal_Arrive_Reward, Finish_Reward = 0.01, 10
JOINT = False  # 是否使用联合奖励计算

# 自定义初始状态
AGV_data_bag = [
    {'name': 1, 'start_id': 3, 'goal_id': 22},
    {'name': 2, 'start_id': 2, 'goal_id': 0},
    {'name': 3, 'start_id': 6, 'goal_id': 13}
]

jump_k = 10


def graph_json(data):
    """
    根据 JSON 图信息构造 NetworkX 图
    """
    G = nx.Graph()
    for node in data["nodes"]:
        node_id = int(node["id"])
        x, y = node["coordinate"]["x"], node["coordinate"]["y"]
        G.add_node(node_id, pos=(x, y))
        for edge in node["edges"]:
            destination = int(edge["destination"])
            weight = edge["weight"]
            G.add_edge(node_id, destination, weight=weight)

    # 归一化边权重
    distances = [math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) for u, v in G.edges()]
    max_distance = max(distances) if distances else 1
    for u, v in G.edges():
        distance = math.dist(G.nodes[u]['pos'], G.nodes[v]['pos'])
        normalized_weight = (distance / max_distance) * 10
        G[u][v]['weight'] = normalized_weight

    return G


def construct_adjacency_matrix(G):
    """
    构建图的邻接矩阵
    """
    nodes = sorted(G.nodes())
    adj_matrix = np.zeros((len(nodes), len(nodes)))
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if G.has_edge(node1, node2):
                adj_matrix[i, j] = G[node1][node2]['weight']
            else:
                adj_matrix[i, j] = 0
    return adj_matrix


class State:
    """
    状态管理类，管理智能体的位置和目标
    """

    def __init__(self, world0, goals, adj_matrix, num_nodes=26, num_agents=3):
        assert world0.shape == (2, num_nodes) and world0.shape == goals.shape, "world0 和 goals 的形状不匹配"
        self.state = world0.copy()
        self.goals = goals.copy()
        self.adj_matrix = adj_matrix
        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.agents_pos, self.agents_goal = self.scanForAgents()
        assert len(self.agents_pos) == num_agents, "智能体数量与初始化不匹配"

    def scanForAgents(self):
        """
        扫描当前世界状态，获取所有智能体的位置和目标
        """
        agents_pos = [-1 for _ in range(self.num_agents)]
        agents_goal = [-1 for _ in range(self.num_agents)]

        # 获取智能体的位置
        for i in range(self.state.shape[1]):
            agent_name = self.state[1, i]
            if agent_name > 0:
                agents_pos[agent_name - 1] = self.state[0, i]

        # 获取智能体的目标
        for i in range(self.goals.shape[1]):
            agent_name = self.goals[1, i]
            if agent_name > 0:
                agents_goal[agent_name - 1] = self.goals[0, i]

        return agents_pos, agents_goal

    def getPos(self, agent_name):
        """
        获取指定智能体的位置
        """
        return self.agents_pos[agent_name - 1]

    def getGoal(self, agent_name):
        """
        获取指定智能体的目标位置
        """
        return self.agents_goal[agent_name - 1]

    def getNeighbors(self, agent_name):
        """
        获取指定智能体当前节点的邻居节点
        """
        neighbors = []
        node_id = self.getPos(agent_name)
        for i in range(self.num_nodes):
            if self.adj_matrix[node_id][i] != 0:
                neighbors.append(i)
        return neighbors

    def act(self, action, agent_name):
        """
        验证智能体的动作，并返回动作状态
        动作状态:
            2: 到达目标点或停留在目标点
            1: 在非目标点停留
            0: 正常移动到邻居节点
            -1: 无效动作
            -2: 碰撞
        """
        neighbors = self.getNeighbors(agent_name)
        agent_position = self.getPos(agent_name)
        agent_goal = self.getGoal(agent_name)

        # 动作验证
        if action not in neighbors and action != agent_position:
            print(f"Agent {agent_name}: 位置{agent_position}无效动作 {action}")
            return -1
        elif action == agent_position:
            if agent_position == agent_goal:
                print(f"Agent {agent_name}: 位置{agent_position}在终点{agent_goal}停留")
                return 2
            else:
                print(f"Agent {agent_name}: 位置{agent_position}在非终点{action}停留")
                return 1
        elif action in neighbors:
            if self.state[1, action] != 0:
                print(f"Agent {agent_name}: 位置{agent_position}动作 {action} 与其他Agent，碰撞")
                return -2
            elif action == agent_goal:
                print(f"Agent {agent_name}: 位置{agent_position}到达终点{agent_goal}")
                return 2
            else:
                print(f"Agent {agent_name}: 位置{agent_position}正常移动到 {action}")
                return 0

    def done(self):
        """
        判断是否所有智能体都到达了各自的目标点
        """
        return all(pos == goal for pos, goal in zip(self.agents_pos, self.agents_goal))


class SimpleMAPFEnv(gym.Env):
    """
    简单的多智能体路径规划环境
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, graph, AGV_bag=None, num_nodes=26, num_agents=3):
        super(SimpleMAPFEnv, self).__init__()
        self.graph = graph
        self.nodes = sorted(list(graph.nodes))
        self.Agv_bag = AGV_bag
        self.num_agents = num_agents
        self.num_nodes = num_nodes
        # 初始化个体奖励为0，并且设置为累加
        self.individual_rewards = [0.0 for _ in range(num_agents)]
        self.max_steps = 100
        self.current_step = 0

        # 构建邻接矩阵
        self.adj_matrix = construct_adjacency_matrix(self.graph)

        # 定义动作空间和观测空间
        # 动作空间：每个智能体选择一个节点进行移动，num_nodes 个节点（包括自己）
        self.action_space = spaces.MultiDiscrete([self.num_nodes for _ in range(self.num_agents)])

        # 观测空间：每个智能体的观测由自己的目标位置、其他智能体的目标和位置组成，形状为 (num_agents, 4, num_nodes)
        self.observation_space = spaces.Box(low=0, high=self.num_agents, shape=(self.num_agents, 4, self.num_nodes),
                                            dtype=np.float32)

        # 初始化世界状态
        self.reset()

    def setWorld(self):
        """
        初始化或重置世界状态，包括智能体的位置和目标
        """
        world = np.zeros((2, self.num_nodes), dtype=int)
        goals = np.zeros((2, self.num_nodes), dtype=int)
        world[0] = np.arange(self.num_nodes)
        goals[0] = np.arange(self.num_nodes)

        if self.Agv_bag is not None:
            for agv in self.Agv_bag:
                name = agv['name']
                start_id = agv['start_id']
                goal_id = agv['goal_id']
                if start_id >= self.num_nodes or goal_id >= self.num_nodes:
                    raise ValueError(f"Agent {name} 的 start_id 或 goal_id 超出节点范围")
                world[1, start_id] = name
                goals[1, goal_id] = name
        else:
            print("未提供初始AGV信息，随机初始化智能体位置和目标")
            for i in range(self.num_agents):
                # 随机选择起始位置，确保无冲突
                start_id = random.randint(0, self.num_nodes - 1)
                while world[1, start_id] != 0:
                    start_id = random.randint(0, self.num_nodes - 1)
                world[1, start_id] = i + 1

                # 随机选择目标位置，确保不同于起始位置且无冲突
                goal_id = random.randint(0, self.num_nodes - 1)
                while goal_id == start_id or goals[1, goal_id] != 0:
                    goal_id = random.randint(0, self.num_nodes - 1)
                goals[1, goal_id] = i + 1

        self.world = State(world, goals, self.adj_matrix, self.num_nodes, self.num_agents)

    # def getDistance(self, node1, node2):
    #     """
    #     获取两个节点之间的最短距离，基于边权重
    #     """
    #     assert node1 in self.graph and node2 in self.graph, "节点不存在于图中"
    #     try:
    #         distance = nx.shortest_path_length(self.graph, source=node1, target=node2, weight='weight')
    #         return distance
    #     except nx.NetworkXNoPath:
    #         return None

    def get_all_observations(self, jump_k):
        """
        获取所有智能体的观测
        """
        observations = []
        for agent_name in range(1, self.num_agents + 1):
            obs = self.getObservation(agent_name)
            observations.append(obs)
        return np.array(observations)

    def getObservation(self, agent_name):
        """
        获取指定智能体的观测
        """
        position_id = self.world.getPos(agent_name)
        goal_id = self.world.getGoal(agent_name)

        our_position = np.zeros(self.num_nodes)
        our_goal = np.zeros(self.num_nodes)
        other_goals = np.zeros(self.num_nodes)
        other_positions = np.zeros(self.num_nodes)

        our_position[position_id] = agent_name
        our_goal[goal_id] = agent_name

        for other_agent in range(1, self.num_agents + 1):
            if other_agent == agent_name: continue
            other_pos = self.world.getPos(other_agent)
            other_goal = self.world.getGoal(other_agent)
            path1 = self.getDistanceWeightOne(position_id, other_pos, None)
            path2 = self.getDistanceWeightOne(position_id, other_goal, None)
            if path1 is not None and len(path1) <= jump_k:
                other_positions[other_pos] = other_agent
            if path2 is not None and len(path2) <= jump_k:
                other_goals[other_goal] = other_agent

        # 合并观测信息
        combined_obs = np.concatenate([our_position, our_goal, other_goals, other_positions])
        # Reshape 为 (4, num_nodes)
        combined_obs = combined_obs.reshape(4, self.num_nodes)
        return combined_obs

    def reset(self):
        """
        重置环境到初始状态
        """
        self.current_step = 0
        # 重置个体奖励为0
        self.individual_rewards = [0.0 for _ in range(self.num_agents)]
        self.setWorld()
        observations = self.get_all_observations(jump_k)
        return observations

    def complete(self):
        """
        检查所有智能体是否都完成了目标
        """
        return self.world.done()

    def getDistanceWeightOne(self, node1, node2, occupied_nodes):
        """
        计算两个节点之间的最短路径列表，排除被占据的节点
        """
        temp_graph = self.graph.copy()
        if occupied_nodes:
            temp_graph.remove_nodes_from(occupied_nodes)

        try:
            path = nx.shortest_path(temp_graph, source=node1, target=node2, weight='weight')
            return path
        except (nx.NodeNotFound, nx.NetworkXNoPath) as e:
            # 打印调试信息，帮助识别问题
            print(f"Path not found from {node1} to {node2}: {e}")
            return None

    def getOtherAgentInVisual(self, agent_name):
        """
        获取指定智能体视野内的其他智能体
        """
        combined_obs = self.getObservation(agent_name)
        _, _, _, other_positions = combined_obs
        other_agents = other_positions[other_positions != 0]
        other_agents_list = other_agents.astype(int).tolist()
        return other_agents_list

    def getBlockingCost(self, agent_name):
        """
        计算智能体阻止其他机器人到达目标的数量，并返回相应的惩罚
        """
        inflation = 5
        other_agents = self.getOtherAgentInVisual(agent_name)
        num_blocking = 0
        for other_agent in other_agents:
            other_pos = self.world.getPos(other_agent)
            other_goal = self.world.getGoal(other_agent)
            node = self.world.getPos(agent_name)
            path_before = self.getDistanceWeightOne(other_pos, other_goal, [node])
            path_after = self.getDistanceWeightOne(other_pos, other_goal, None)
            if (path_before is None and path_after is not None) or (
                    path_before is not None and path_after is not None and len(path_before) > len(
                path_after) + inflation
            ):
                num_blocking = num_blocking + 1

        return num_blocking * Blocking_Cost

    def step(self, actions):
        """
        执行动作并更新环境状态
        参数:
        - actions: 一个包含每个智能体动作的列表，例如 [action_agent1, action_agent2, ...]
        返回:
        - observations: 所有智能体的观测
        - rewards: 每个智能体的奖励列表
        - done: 是否完成
        - info: 其他信息字典
        """
        self.current_step += 1
        rewards = [0 for _ in range(self.num_agents)]
        infos = {}

        # # 生成智能体列表并随机打乱顺序
        # agent_order = list(range(1, self.num_agents + 1))
        # random.shuffle(agent_order)
        # print(f"Random agent execution order: {agent_order}")

        # 临时记录已选择的目标节点，避免冲突
        target_nodes = {}

        for agent_name in range(1,self.num_agents+1):
            action = actions[agent_name - 1]
            print(f"Processing Agent {agent_name} with action {action}")

            # 获取动作状态
            action_status = self.world.act(action, agent_name)

            # 检查冲突：如果多个智能体选择同一节点
            if action in target_nodes and action_status == 0:
                # 冲突发生，将当前智能体视为碰撞，撤销动作
                print(f"Conflict detected: Agent {agent_name} tried to move to already targeted {action}")
                action_status = -2  # 碰撞
                # 不修改状态，保持当前智能体位置不变
                # 可选择通知先前智能体有冲突发生
            else:
                # 标记目标节点
                if action != self.world.getPos(agent_name):  # 不考虑原地不移动
                    target_nodes[action] = agent_name

            # 计算奖励
            if action_status == 2:  # 达到目标或停留在目标
                reward = Goal_Arrive_Reward  # =0.01
                # 更新状态
                if action == self.world.getGoal(agent_name):
                    self.world.state[1, self.world.getPos(agent_name)] = 0  # 清除当前位置
                    self.world.state[1, action] = agent_name  # 更新新位置
                    self.world.agents_pos[agent_name - 1] = action  # 更新位置
            elif action_status == 1:  # 在非目标点停留
                reward = Stay_Cost  # =-0.5
                # 保持当前状态
            elif action_status == 0:  # 正常移动
                reward = Move_Cost  # =-0.3
                # 更新状态
                self.world.state[1, self.world.getPos(agent_name)] = 0
                self.world.state[1, action] = agent_name
                self.world.agents_pos[agent_name - 1] = action
            elif action_status == -1:  # 无效动作
                reward = Wrong_Cost  # =-3
                # 不修改状态
            elif action_status == -2:  # 碰撞
                reward = Collision_Cost  # =-2
                # 不修改状态
            else:
                reward = 0
                print(f"Agent {agent_name}: 未知的动作状态 {action_status}")

            # 计算阻塞惩罚
            blocking_cost = self.getBlockingCost(agent_name)
            reward += blocking_cost
            rewards[agent_name - 1] += reward
            self.individual_rewards[agent_name - 1] += reward

            # # 处理联合奖励
            # if JOINT:
            #     other_agents = self.getOtherAgentInVisual(agent_name)
            #     v = len(other_agents)
            #     if v > 0:
            #         adjusted_reward = self.individual_rewards[agent_name - 1] / 2
            #         for agent in other_agents:
            #             adjusted_reward += self.individual_rewards[agent - 1] / (v * 2)
            #         rewards[agent_name - 1] = adjusted_reward

            # 记录信息
            on_goal = self.world.getPos(agent_name) == self.world.getGoal(agent_name)
            if agent_name not in infos:
                infos[agent_name] = {}
            infos[agent_name].update({
                'blocking': blocking_cost < 0,
                'valid_action': action_status >= 0,
                'on_goal': on_goal
            })

        # 检查是否完成
        done = self.complete() or self.current_step >= self.max_steps

        # 如果所有智能体完成任务，给予 Finish_Reward 并终止
        if self.complete():
            print("所有智能体已完成目标，给予额外的 Finish_Reward")
            for i in range(self.num_agents):
                self.individual_rewards[i] += Finish_Reward
                rewards[i] += Finish_Reward

        # 如果达到最大步数且任务未完成，打印信息
        if self.current_step >= self.max_steps and not self.complete():
            print("任务没完成")
            total_reward = sum(self.individual_rewards)
            print(f"Total rewards: {total_reward}")

        # 如果达到最大步数，给予超时惩罚
        if self.current_step >= self.max_steps:
            for i in range(self.num_agents):
                self.individual_rewards[i] -= 5  # 超时惩罚
                rewards[i] -= 5
            print("环境超时，所有智能体获得超时惩罚 -5")

        # 获取观测
        observations = self.get_all_observations(jump_k)

        # 将总奖励添加到 infos 字典中
        infos['total_rewards'] = self.individual_rewards.copy()

        return observations, rewards, done, infos

    def get_valid_actions(self, agent_name):
        """
        获取指定智能体的合法动作列表。
        合法动作包括当前节点（停留）和相邻节点（移动）。
        """
        neighbors = self.world.getNeighbors(agent_name)
        current_position = self.world.getPos(agent_name)
        valid_actions = neighbors.copy()
        valid_actions.append(current_position)  # 允许停留
        return valid_actions

    def render(self, mode='human'):
        plt.figure(figsize=(8, 8))
        pos = nx.get_node_attributes(self.graph, 'pos')

        # 绘制节点和边
        nx.draw(self.graph, pos, with_labels=True, node_size=500, node_color='lightblue')

        # 为不同智能体分配不同的颜色
        agent_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        if self.num_agents > len(agent_colors):
            # 动态生成更多颜色
            agent_colors = plt.cm.get_cmap('hsv', self.num_agents)

        for agent_idx, agent_name in enumerate(range(1, self.num_agents + 1)):
            color = agent_colors[agent_idx % len(agent_colors)]
            agent_pos = self.world.getPos(agent_name)
            agent_goal = self.world.getGoal(agent_name)

            # 为每个智能体添加标签
            label_pos = f'Agent {agent_name} Pos'
            label_goal = f'Agent {agent_name} Goal'

            # 绘制位置和目标
            plt.scatter(*pos[agent_pos], color=color, s=200, label=label_pos)
            plt.scatter(*pos[agent_goal], color=color, s=200, marker='*', label=label_goal)

            # 添加文本标注，调整位置和大小
            plt.text(pos[agent_pos][0] + 0.1, pos[agent_pos][1] + 0.1, f'A{agent_name}',
                     fontsize=10, ha='center', va='center', color=color,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            plt.text(pos[agent_goal][0] + 0.1, pos[agent_goal][1] + 0.1, f'G{agent_name}',
                     fontsize=10, ha='center', va='center', color=color,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 去重标签
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
        plt.title('AGV Positions and Goals')
        plt.show()
        plt.close()  # 关闭当前图形，释放内存

    def close(self):
        """
        关闭环境
        """
        plt.close()


# 示例用法
if __name__ == "__main__":
    # 加载图数据
    try:
        with open('../data/1022.json', 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("JSON 文件未找到，请确保路径 '../data/1022.json' 正确并且文件存在。")
        exit(1)

    # 构建图
    G = graph_json(data)

    # 打印AGV数据包（假设AGV_data_bag在其他地方定义）
    print("AGV_data_bag:", AGV_data_bag)

    # 初始化环境
    env = SimpleMAPFEnv(graph=G, AGV_bag=AGV_data_bag, num_nodes=26, num_agents=3)

    # 重置环境
    observations = env.reset()

    # 渲染初始状态
    env.render()

    # 主循环
    for step in range(env.max_steps):
        actions = []
        for agent_idx in range(1, env.num_agents + 1):
            current_pos = env.world.getPos(agent_idx)
            neighbors = env.world.getNeighbors(agent_idx)
            possible_actions = neighbors + [current_pos]  # 邻居节点加上停留
            action = random.choice(possible_actions)  # 随机选择一个有效动作
            actions.append(action)

        print(f"\nStep {step + 1}:")
        print(f"Actions: {actions}")
        observations, rewards, done, infos = env.step(actions)
        print(f"Rewards: {rewards}")
        print(f"Total Rewards So Far: {infos.get('total_rewards', [])}")
        print(f"Done: {done}")
        print(f"Infos: {infos}")
        env.render()

        if done:
            if env.complete():
                print("所有智能体已完成目标。")
            else:
                print("任务在最大轮次内未完成。")
            break

    # 如果任务在所有步数后仍未完成
    if not env.complete():
        print("任务未完成。")
        print(f"总奖励: {env.individual_rewards}")
    else:
        print(f"所有智能体的总奖励: {env.individual_rewards}")

    env.close()
