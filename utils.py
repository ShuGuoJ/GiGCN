"""
Raw image -> Superpixel segmentation -> graph
"""
import numpy as np
import torch
import cv2 as cv
from torch_scatter import scatter
from torch_geometric.data import Data
import copy
from torch import nn


# Getting adjacent relationship among nodes
def get_edge_index(segment):
    if isinstance(segment, torch.Tensor):
        segment = segment.numpy()
    # 扩张
    img = segment.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    expansion = cv.dilate(img, kernel)
    mask = segment == expansion
    mask = np.invert(mask)
    # 构图
    h, w = segment.shape
    edge_index = set()
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    indices = list(zip(*np.nonzero(mask)))
    for x, y in indices:
        for dx, dy in directions:
            adj_x, adj_y = x + dx, y + dy
            if -1 < adj_x < h and -1 < adj_y < w:
                source, target = segment[x, y], segment[adj_x, adj_y]
                if source != target:
                    edge_index.add((source, target))
                    edge_index.add((target, source))
    return torch.tensor(list(edge_index), dtype=torch.long).T, edge_index


# Getting node features
def get_node(x, segment, mode='mean'):
    assert x.ndim == 3 and segment.ndim == 2
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(segment, np.ndarray):
        segment = torch.from_numpy(segment).to(torch.long)
    c = x.shape[2]
    x = x.reshape((-1, c))
    mask = segment.flatten()
    nodes = scatter(x, mask, dim=0, reduce=mode)
    return nodes.to(torch.float32)


# Constructing graphs by shifting
def get_grid_adj(grid):
    edge_index = list()
    # 上偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:-1] = grid[1:]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 下偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[1:] = grid[:-1]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 左偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:, :-1] = grid[:, 1:]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 右偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:, 1:] = grid[:, :-1]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    return edge_index


# Getting graph list
def get_graph_list(data, seg):
    graph_node_feature = []
    graph_edge_index = []
    for i in np.unique(seg):
        # 获取节点特征
        graph_node_feature.append(data[seg == i])
        # 获取邻接信息
        x, y = np.nonzero(seg == i)
        n = len(x)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        grid = np.full((x_max - x_min + 1, y_max - y_min + 1), -1, dtype=np.int32)
        x_hat, y_hat = x - x_min, y - y_min
        grid[x_hat, y_hat] = np.arange(n)
        graph_edge_index.append(get_grid_adj(grid))
    graph_list = []
    # 数据变换
    for node, edge_index in zip(graph_node_feature, graph_edge_index):
        node = torch.tensor(node, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        graph_list.append(Data(node, edge_index=edge_index))
    return graph_list


def split(graph_list, gt, mask):
    indices = np.nonzero(gt)
    ans = []
    number = mask[indices]
    gt = gt[indices]
    for i, n in enumerate(number):
        graph = copy.deepcopy(graph_list[n])
        graph.y = torch.tensor([gt[i]], dtype=torch.long)
        ans.append(graph)
    return ans


def summary(net: nn.Module):
    single_dotted_line = '-' * 40
    double_dotted_line = '=' * 40
    star_line = '*' * 40
    content = []
    def backward(m: nn.Module, chain: list):
        children = m.children()
        params = 0
        chain.append(m._get_name())
        try:
            child = next(children)
            params += backward(child, chain)
            for child in children:
                params += backward(child, chain)
            # print('*' * 40)
            # print('{:>25}{:>15,}'.format('->'.join(chain), params))
            # print('*' * 40)
            if content[-1] is not star_line:
                content.append(star_line)
            content.append('{:>25}{:>15,}'.format('->'.join(chain), params))
            content.append(star_line)
        except:
            for p in m.parameters():
                if p.requires_grad:
                    params += p.numel()
            # print('{:>25}{:>15,}'.format(chain[-1], params))
            content.append('{:>25}{:>15,}'.format(chain[-1], params))
        chain.pop()
        return params
    # print('-' * 40)
    # print('{:>25}{:>15}'.format('Layer(type)', 'Param'))
    # print('=' * 40)
    content.append(single_dotted_line)
    content.append('{:>25}{:>15}'.format('Layer(type)', 'Param'))
    content.append(double_dotted_line)
    params = backward(net, [])
    # print('=' * 40)
    # print('-' * 40)
    content.pop()
    content.append(single_dotted_line)
    print('\n'.join(content))
    return params


