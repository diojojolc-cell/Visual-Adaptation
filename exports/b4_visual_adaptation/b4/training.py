"""Optimization and bounded activation storage for B4."""

import math
from contextlib import AbstractContextManager

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


def learning_rate(update, total, warmup, peak=1e-4):
    if update <= warmup:
        return peak * update / max(warmup, 1)
    return peak * 0.5 * (1 + math.cos(math.pi * (update - warmup) / (total - warmup)))


def answer_loss(logits, gold, option_ids, update, raw_steps=50):
    """Original answer-position CE/hinge, followed by differentiable RMS loss."""
    z = logits.float()
    full_lse = torch.logsumexp(z, dim=0)
    raw_ce = full_lse - z[gold]
    if update <= raw_steps:
        wrong = torch.cat((z[:gold], z[gold + 1:])).max()
        return raw_ce + 0.1 * F.relu(0.5 - (z[gold] - wrong)), raw_ce
    legal = z[option_ids]
    centered = legal - legal.mean()
    normalized = centered / centered.square().mean().sqrt()
    index = option_ids.index(gold)
    wrong = torch.cat((normalized[:index], normalized[index + 1:])).max()
    ce = torch.logsumexp(normalized, dim=0) - normalized[index]
    hinge = F.relu(0.5 - (normalized[index] - wrong))
    format_loss = full_lse - torch.logsumexp(legal, dim=0)
    return ce + 0.1 * hinge + format_loss, raw_ce


def selective_context():
    saved = {
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
    }

    def policy(context, operation, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE if operation in saved else CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy)


class FrozenHead(torch.autograd.Function):
    """Native BF16 logits, FP32 input-gradient accumulation, no head training."""

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        return F.linear(x.to(weight.dtype), weight).float()

    @staticmethod
    def backward(ctx, gradient):
        weight, = ctx.saved_tensors
        result = torch.zeros(weight.shape[1], device=gradient.device, dtype=torch.float32)
        for start in range(0, len(gradient), 8192):
            result.add_(gradient[start:start + 8192].float() @ weight[start:start + 8192].float())
        return result, None


class FrozenVideo(AbstractContextManager):
    """One video's frozen merger/DeepStack features, reused across its questions."""

    def __init__(self, model):
        self.model = model.model
        self.original = self.model.get_video_features
        self.value = None

    def __enter__(self):
        def features(*args, **kwargs):
            if self.value is None:
                with torch.no_grad():
                    self.value = self.original(*args, **kwargs)
            return self.value
        self.model.get_video_features = features
        return self

    def __exit__(self, *exc):
        del self.model.get_video_features
        self.value = None


def mean_gradients(adapter, count, distributed):
    parameters = list(adapter.parameters())
    flat = torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in parameters
    ])
    if distributed:
        torch.distributed.all_reduce(flat)
    flat.div_(count)
    offset = 0
    for p in parameters:
        p.grad = flat[offset:offset + p.numel()].view_as(p)
        offset += p.numel()
