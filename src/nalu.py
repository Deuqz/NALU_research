from torch import nn
from torch.nn import Parameter
from torch.nn import init

import math
import torch
import torch.nn.functional as F


class NAC(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.W_hat = Parameter(torch.zeros((out_dim, in_dim)))
        self.M_hat = Parameter(torch.zeros((out_dim, in_dim)))

        self.register_parameter('W_hat', self.W_hat)
        self.register_parameter('M_hat', self.M_hat)
        self.register_parameter('bias', None)

        init.kaiming_uniform_(self.W_hat)
        init.kaiming_uniform_(self.M_hat)

    def forward(self, x):
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return F.linear(x, W, self.bias)


class NALU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, eps=1e-10):
        super().__init__()
        self.eps = eps
        self.nac = NAC(in_dim, out_dim)
        self.G = Parameter(torch.zeros((out_dim, in_dim)))

        self.register_parameter('bias', None)

        init.kaiming_uniform_(self.G, a=math.sqrt(5))

    def forward(self, x):
        g = torch.sigmoid(F.linear(x, self.G, self.bias))
        a = self.nac(x)
        x_log = torch.log(torch.abs(x) + self.eps)
        m = torch.exp(self.nac(x_log))

        out = g * a + (1 - g) * m
        return out


class INALU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, eps=1e-10, w=20, device='cuda'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.t = 20
        self.device = device

        self.low_bound = torch.fill(torch.zeros(1, in_dim), eps).to(device)
        self.up_bound = torch.fill(torch.zeros(1, out_dim), w).to(device)

        self.G = Parameter(torch.rand(out_dim))

        self.W_hat_a = Parameter(torch.zeros((out_dim, in_dim)))
        self.M_hat_a = Parameter(torch.zeros((out_dim, in_dim)))
        self.W_hat_m = Parameter(torch.zeros((out_dim, in_dim)))
        self.M_hat_m = Parameter(torch.zeros((out_dim, in_dim)))

        self.register_parameter('bias', None)

        init.kaiming_uniform_(self.W_hat_a)
        init.kaiming_uniform_(self.M_hat_a)
        init.kaiming_uniform_(self.W_hat_m)
        init.kaiming_uniform_(self.M_hat_m)

    def forward(self, x):
        W_a = torch.tanh(self.W_hat_a) * torch.sigmoid(self.M_hat_a)
        W_m = torch.tanh(self.W_hat_m) * torch.sigmoid(self.M_hat_m)

        a = F.linear(x, W_a, self.bias)

        x_log = torch.log(torch.max(x.abs(), self.low_bound))
        nac_res = torch.min(F.linear(x_log, W_m, self.bias), self.up_bound)
        m = torch.exp(nac_res)

        W_m_abs = W_m.abs()
        x_sign = torch.sign(x.view(x.shape[0], 1, -1))
        msm = x_sign * W_m_abs + 1 - W_m_abs
        msv = msm.prod(dim=-1)

        g = torch.sigmoid(self.G)

        out = g * a + (1 - g) * m * msv
        return out

    def reinitialize(self):
        self.G = Parameter(torch.rand(self.out_dim))

        self.W_hat_a = Parameter(torch.zeros((self.out_dim, self.in_dim)))
        self.M_hat_a = Parameter(torch.zeros((self.out_dim, self.in_dim)))
        self.W_hat_m = Parameter(torch.zeros((self.out_dim, self.in_dim)))
        self.M_hat_m = Parameter(torch.zeros((self.out_dim, self.in_dim)))

        init.xavier_uniform_(self.W_hat_a)
        init.xavier_uniform_(self.M_hat_a)
        init.xavier_uniform_(self.W_hat_m)
        init.xavier_uniform_(self.M_hat_m)

    def _reg_loss_var(self, var):
        return torch.mean(torch.max(torch.min(-var, var) + self.t, torch.zeros_like(var)) / self.t)

    def reg_loss(self):
        return self._reg_loss_var(self.G) \
            + self._reg_loss_var(self.W_hat_a) \
            + self._reg_loss_var(self.W_hat_m) \
            + self._reg_loss_var(self.M_hat_a) \
            + self._reg_loss_var(self.M_hat_m)
