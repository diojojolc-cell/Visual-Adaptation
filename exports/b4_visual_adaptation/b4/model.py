"""B4: state-conditioned visual reading and FFN-output adaptation."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def rms(x):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


class OutputAdapter(nn.Module):
    def __init__(self, condition, hidden, intermediate, *, device=None):
        super().__init__()
        self.width = round(2 * condition * intermediate / (condition + hidden))
        self.U = nn.Parameter(torch.empty(
            condition, self.width, device=device, dtype=torch.float32))
        self.V = nn.Parameter(torch.zeros(
            self.width, hidden, device=device, dtype=torch.float32))
        nn.init.xavier_uniform_(self.U)

    def forward(self, c):
        return F.silu(c @ self.U) @ self.V


class B4Adapter(nn.Module):
    """Four receiving layers; shared reader/conditioner; separate output writers."""

    def __init__(self, d=4096, m=12288, *, device=None):
        super().__init__()
        self.layers = (8, 17, 26, 35)
        self.heads, self.key_dim, self.condition, self.chunk = 4, 128, 128, 16
        options = dict(device=device, dtype=torch.float32)
        self.W_Q = nn.Parameter(torch.empty(d, self.heads * self.key_dim, **options))
        self.W_K = nn.Parameter(torch.empty(d, self.heads * self.key_dim, **options))
        self.E_layer = nn.Parameter(torch.zeros(4, self.heads * self.key_dim, **options))
        self.W_C = nn.Parameter(torch.empty(d * (1 + self.heads), self.condition, **options))
        self.b_C = nn.Parameter(torch.zeros(self.condition, **options))
        for weight in (self.W_Q, self.W_K, self.W_C):
            nn.init.xavier_uniform_(weight)
        self.output_adapters = nn.ModuleList([
            OutputAdapter(self.condition, d, m, device=device) for _ in self.layers
        ])

    def keys(self, memory):
        with torch.autocast(memory.values.device.type, enabled=False):
            raw = memory.normalized_values @ self.W_K
            split = raw.view(-1, self.heads, self.key_dim).transpose(0, 1)
            return rms(split + memory.positional[None])

    def output_delta(self, layer, x, memory, keys):
        index = self.layers.index(layer)
        with torch.autocast(x.device.type, enabled=False):
            q = rms(x) @ self.W_Q + self.E_layer[index]
            q = rms(q.view(-1, self.heads, self.key_dim).transpose(0, 1))
            chunks = []
            for start in range(0, len(x), self.chunk):
                stop = start + self.chunk
                logits = torch.einsum('htd,hnd->htn', q[:, start:stop], keys)
                logits = logits / math.sqrt(self.key_dim) + memory.prior[None, start:stop] / 0.1
                logits = logits.masked_fill(~memory.causal[None, start:stop], -torch.inf)
                response = logits.softmax(-1) @ memory.values
                z = torch.cat((
                    rms(x[start:stop]), rms(response).transpose(0, 1).flatten(1)
                ), dim=-1)
                c = F.silu(z @ self.W_C + self.b_C)
                chunks.append(self.output_adapters[index](c))
            return torch.cat(chunks)
