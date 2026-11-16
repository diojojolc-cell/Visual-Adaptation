"""Visual memory from one prompt's native multimodal embeddings."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import rms


def positional_encoding(coordinates, media):
    frequency = 10000 ** (-2 * torch.arange(21, device=coordinates.device).float() / 42)
    phase = coordinates.float()[:, :, None] * frequency
    return torch.cat((
        torch.stack((phase.sin(), phase.cos()), -1).flatten(1), media.float()
    ), dim=-1)


@dataclass
class VisualMemory:
    values: torch.Tensor
    normalized_values: torch.Tensor
    positional: torch.Tensor
    active: torch.Tensor
    prior: torch.Tensor
    causal: torch.Tensor

    @classmethod
    @torch.no_grad()
    def from_prompt(cls, *, input_ids, inputs_embeds, attention_mask,
                    rope_positions, question_option_positions,
                    image_token_id, video_token_id):
        """Inputs have batch size one; token indices refer to the expanded prompt."""
        ids = input_ids[0]
        image, video = ids == image_token_id, ids == video_token_id
        visual = image | video
        positions = visual.nonzero().flatten()
        values = inputs_embeds[0, visual].detach().float()
        text_positions = (attention_mask[0].bool() & ~visual).nonzero().flatten()
        text = inputs_embeds[0, text_positions].detach().float()

        question_positions = torch.as_tensor(
            question_option_positions, dtype=torch.long, device=ids.device)
        active = torch.cat((question_positions, ids.new_tensor([len(ids) - 1])))
        active = active.unique(sorted=True)
        # Only queries with an earlier visual token receive adaptation.
        active = active[active > positions[0]]

        norm = F.normalize(values, dim=-1)
        running = torch.full((len(positions),), -torch.inf, device=ids.device)
        visible_counts = torch.searchsorted(text_positions, active, right=True).cpu().tolist()
        priors, cursor = [], 0
        for visible in visible_counts:
            for start in range(cursor, visible, 256):
                prefix = F.normalize(text[start:min(start + 256, visible)], dim=-1)
                scores = norm @ prefix.T
                running = torch.maximum(running, scores.max(-1).values)
            cursor = visible
            priors.append(running.clone())

        coordinates = rope_positions[:, 0, visual].T.contiguous()
        media = torch.stack((image[visual], video[visual]), -1)
        return cls(
            values=values,
            normalized_values=rms(values),
            positional=positional_encoding(coordinates, media),
            active=active,
            prior=torch.stack(priors),
            causal=positions[None, :] < active[:, None],
        )
