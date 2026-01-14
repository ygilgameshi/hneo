"""
Task Graph神经网络

使用GNN学习任务之间的关系，生成task embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np
import networkx as nx
from pathlib import Path


class TaskGNN(nn.Module):
    """
    Task Graph神经网络

    在Task Graph上进行消息传递，学习每个task的嵌入表示

    Args:
        n_tasks: 任务数量
        node_feat_dim: 节点初始特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出嵌入维度
        n_layers: GNN层数
        gnn_type: 'gcn' 或 'gat'
        dropout: Dropout比例

    Example:
        >>> task_gnn = TaskGNN(n_tasks=50, output_dim=64)
        >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 边索引
        >>> edge_weight = torch.tensor([0.8, 0.9, 0.7])  # 边权重
        >>> embeddings = task_gnn(edge_index, edge_weight)
        >>> print(embeddings.shape)  # (50, 64)
    """

    def __init__(self,
                 n_tasks,
                 node_feat_dim=64,
                 hidden_dim=128,
                 output_dim=64,
                 n_layers=3,
                 gnn_type='gcn',
                 dropout=0.1):
        super(TaskGNN, self).__init__()

        self.n_tasks = n_tasks
        self.output_dim = output_dim
        self.gnn_type = gnn_type
        self.n_layers = n_layers

        # 节点初始特征 (可学习的嵌入)
        self.node_embedding = nn.Embedding(n_tasks, node_feat_dim)

        # GNN层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_dim = node_feat_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else output_dim

            if gnn_type == 'gcn':
                self.convs.append(GCNConv(in_dim, out_dim))
            elif gnn_type == 'gat':
                # GAT with 4 heads
                self.convs.append(
                    GATConv(in_dim, out_dim, heads=4, concat=False, dropout=dropout)
                )
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")

            # 不在最后一层添加BN
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_dim))

            in_dim = out_dim

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index, edge_weight=None, batch=None):
        """
        前向传播

        Args:
            edge_index: (2, num_edges) 边索引
            edge_weight: (num_edges,) 边权重 (可选)
            batch: batch索引 (可选，用于图级别聚合)

        Returns:
            task_embeddings: (n_tasks, output_dim) 或 (batch_size, output_dim)
        """
        device = edge_index.device

        # 初始节点特征
        x = self.node_embedding(torch.arange(self.n_tasks, device=device))

        # GNN层
        for i, conv in enumerate(self.convs):
            if self.gnn_type == 'gcn':
                x = conv(x, edge_index, edge_weight)
            elif self.gnn_type == 'gat':
                x = conv(x, edge_index)

            # 激活和正则化 (除了最后一层)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = self.dropout(x)

        # 如果提供了batch索引，进行图级别聚合
        if batch is not None:
            x = global_mean_pool(x, batch)

        return x

    def get_task_embedding(self, task_idx):
        """
        获取特定task的嵌入 (不通过GNN传播)

        Args:
            task_idx: int 或 tensor

        Returns:
            embedding: (output_dim,) 或 (batch_size, output_dim)
        """
        device = self.node_embedding.weight.device

        if isinstance(task_idx, int):
            task_idx = torch.tensor([task_idx], device=device)
        elif not isinstance(task_idx, torch.Tensor):
            task_idx = torch.tensor(task_idx, device=device)
        else:
            task_idx = task_idx.to(device)

        return self.node_embedding(task_idx)


class TaskGraphWrapper:
    """
    Task Graph的PyTorch Geometric包装器

    加载NetworkX图并转换为PyG格式

    Args:
        graph_file: Task Graph文件路径 (.pkl)
        similarity_matrix_file: 相似性矩阵文件路径 (.npy)

    Example:
        >>> wrapper = TaskGraphWrapper('data/phase1_dataset/task_graph.pkl')
        >>> wrapper.to('cuda')
        >>> print(f"Nodes: {wrapper.n_tasks}, Edges: {wrapper.edge_index.shape[1]}")
    """

    def __init__(self,
                 graph_file='data/phase1_dataset/task_graph.pkl',
                 similarity_matrix_file='data/phase1_dataset/similarity_matrix.npy'):

        # 加载NetworkX图
        self.G = nx.read_gpickle(graph_file)
        self.n_tasks = self.G.number_of_nodes()

        # 加载相似性矩阵
        self.similarity_matrix = np.load(similarity_matrix_file)

        print(f"Loaded Task Graph:")
        print(f"  Nodes: {self.n_tasks}")
        print(f"  Edges: {self.G.number_of_edges()}")

        # 转换为PyG格式
        self.edge_index, self.edge_weight = self._to_pyg()

        # 获取节点属性
        self.task_names = [self.G.nodes[i]['hla'] for i in range(self.n_tasks)]

    def _to_pyg(self):
        """转换为PyG的edge_index和edge_weight"""
        edges = []
        weights = []

        for u, v, data in self.G.edges(data=True):
            # 无向图，添加双向边
            edges.append([u, v])
            edges.append([v, u])

            weight = data.get('weight', 1.0)
            weights.append(weight)
            weights.append(weight)

        if len(edges) == 0:
            # 空图，返回空tensor
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float)
        else:
            edge_index = torch.LongTensor(edges).t().contiguous()
            edge_weight = torch.FloatTensor(weights)

        return edge_index, edge_weight

    def to(self, device):
        """移动到指定设备"""
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)
        return self

    def get_task_name(self, task_idx):
        """获取task名称 (HLA allele)"""
        return self.task_names[task_idx]

    def get_neighbors(self, task_idx, k=5):
        """
        获取某个task的k个最相似的neighbors

        Args:
            task_idx: 任务索引
            k: 返回top-k个neighbors

        Returns:
            neighbor_indices, similarities
        """
        similarities = self.similarity_matrix[task_idx]

        # 排除自己
        similarities = similarities.copy()
        similarities[task_idx] = -1

        # 获取top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_sims = similarities[top_k_indices]

        return top_k_indices, top_k_sims


# 测试代码
if __name__ == "__main__":
    print("Testing TaskGNN...")

    # 创建简单的Task Graph
    n_tasks = 20

    # 随机生成边
    np.random.seed(42)
    n_edges = 50
    edges = []
    weights = []

    for _ in range(n_edges):
        u, v = np.random.choice(n_tasks, 2, replace=False)
        edges.append([u, v])
        edges.append([v, u])
        weight = np.random.uniform(0.3, 1.0)
        weights.extend([weight, weight])

    edge_index = torch.LongTensor(edges).t().contiguous()
    edge_weight = torch.FloatTensor(weights)

    print(f"✓ Created graph: {n_tasks} nodes, {len(edges) // 2} edges")

    # 创建TaskGNN
    task_gnn = TaskGNN(
        n_tasks=n_tasks,
        node_feat_dim=64,
        hidden_dim=128,
        output_dim=64,
        n_layers=3,
        gnn_type='gcn'
    )

    print(f"✓ Created TaskGNN with {sum(p.numel() for p in task_gnn.parameters()):,} parameters")

    # 前向传播
    embeddings = task_gnn(edge_index, edge_weight)

    print(f"✓ Task embeddings shape: {embeddings.shape}")
    print(f"✓ Sample embedding: {embeddings[0][:5]}")

    # 测试get_task_embedding
    single_embedding = task_gnn.get_task_embedding(0)
    print(f"✓ Single task embedding shape: {single_embedding.shape}")

    print("\n✓ TaskGNN test passed!")