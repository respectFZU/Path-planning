import random
import numpy as np
import torch
from node2vec import Node2Vec
from Actor.data_small import G

# 设置随机种子
seed = 66
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 假设 G 是你的 NetworkX 图
node2vec = Node2Vec(G, dimensions=4, walk_length=5, num_walks=10, workers=1, seed=seed)
model = node2vec.fit(window=3, min_count=1, batch_words=4)

# 获取节点嵌入
node_embeddings = [model.wv[str(node)] for node in G.nodes()]
node_features = torch.tensor(np.array(node_embeddings), dtype=torch.float)
# node_features = torch.tensor([model.wv[str(node)] for node in G.nodes()], dtype=torch.float)
# print(node_features.shape)
# print(node_features)

# G: NetworkX 图对象
# 定义：G 是一个 NetworkX 图对象，表示节点和边的集合。
# 作用：Node2Vec 使用这个图来执行随机游走，以生成节点的嵌入。
# dimensions=4: 嵌入向量的维度
# 定义：每个节点嵌入向量的维度。
# 作用：决定了嵌入向量的大小。较高的维度可能捕捉更多信息，但也增加了计算复杂度。
# walk_length=10: 每次随机游走的步长
# 定义：每次随机游走的最大步数。
# 作用：影响嵌入的局部性。较短的步长关注局部结构，较长的步长可能捕捉更全局的图信息。
# num_walks=100: 每个节点的随机游走次数
# 定义：从每个节点开始的随机游走次数。
# 作用：更多的游走次数可以提高嵌入的稳定性和准确性，但也增加了计算时间。
# workers=1: 使用的线程数
# 定义：用于并行计算的线程数。
# 作用：增加线程数可以加速计算，特别是在大图上，但受限于硬件资源。
# window=5: 上下文窗口大小
# 定义：在训练嵌入时，考虑的节点上下文窗口大小。
# 作用：决定了在随机游走序列中，目标节点的上下文节点数量。较大的窗口可能捕捉更多的上下文信息。
# min_count=1: 忽略出现次数低于此值的节点
# 定义：在训练过程中，忽略出现次数低于该值的节点。
# 作用：用于过滤掉不常见的节点，减少噪声。设置为 1 表示不忽略任何节点。
# batch_words=4: 每批次处理的单词数
# 定义：每次训练批次中处理的“单词”（节点）数量。
# 作用：影响训练的批处理大小。较小的批次可能更稳定，但训练速度较慢。
# 这些参数影响 Node2Vec 的训练过程和生成的嵌入质量。根据具体需求，可以调整这些参数以优化模型性能。
