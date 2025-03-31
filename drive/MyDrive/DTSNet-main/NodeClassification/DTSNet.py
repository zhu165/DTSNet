from typing import Optional
from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np


class DTS_prop(MessagePassing):
    def __init__(self, K, bias=True, alpha = 0.9, **kwargs):
        super(DTS_prop, self).__init__(aggr='add', **kwargs)#aggr所用的聚合方法
        # 当传入字典形式的参数时，就要使用 ** kwargs
        self.K = K
        self.alpha = alpha
        # 定义新的初始化变量。模型中的参数，它是Parameter()类，
        # 先转化为张量，再转化为可训练的Parameter对象
        # Parameter用于将参数自动加入到参数列表
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)#Fills self tensor with the specified value.

    def forward(self, x,  edge_index, edge_weight=None):

        #TEMP = self.temp
        TEMP = F.relu(self.temp)
        alpha = self.alpha

        # 计算拉普拉斯算子L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))

        # 初始化 tmp 列表
        tmp = [x]
        for i in range(self.K):
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            tmp.append(x)

        # 第一段多项式的输出
        out = (1 / math.factorial(self.K)) * TEMP[self.K] * tmp[self.K]

        # 第二段多项式的输出
        for i in range(self.K):
            x = tmp[self.K - i - 1]
            out += (1 / math.factorial(self.K - i - 1)) * TEMP[self.K - i - 1] * x

        # 计算第二段的输出，需重新计算 edge_index2 和 norm2 2I-L
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))

        # 清空 tmp 列表以重新计算
        tmp = [x]
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        # 第二段多项式的初始项
        out = alpha*out + (1-alpha)*(comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        # 合并第二段的项
        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)

            out = out + (1-alpha)*(comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)




