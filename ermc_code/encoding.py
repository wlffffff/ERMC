"""Semantic metadata encoder, independent of the host image encoder."""
import torch
from torch import nn
from ermc.goals import STATE_COUNT


class SemanticEncoding(nn.Module):
    """Input: raw channel-first image tensor [batch, 4, height, width].

    The first three channels alone go to the host visual encoder. Concatenate
    its output with this module's 16 features before the shared policy core.
    Do not feed the metadata channel into image convolutions. For preprocessed
    images, pass normalized=True (all metadata have been divided by 255).
    """

    def __init__(self):
        super().__init__()
        self.phase = nn.Embedding(8, 16)
        self.object = nn.Embedding.from_pretrained(torch.zeros(5, 8), freeze=False, padding_idx=0)
        self.predicate = nn.Embedding.from_pretrained(torch.zeros(5, 8), freeze=False, padding_idx=0)
        self.color = nn.Embedding.from_pretrained(torch.zeros(7, 16), freeze=False, padding_idx=0)

    def forward(self, observations, normalized=False):
        raw = observations[:, 3, 0, :4]
        if normalized:
            raw = raw * 255
        ids = raw.round().long()
        if not torch.allclose(raw.float(), ids.float(), atol=1e-4):
            raise ValueError('Metadata must encode integer IDs')
        for column, maximum in enumerate((STATE_COUNT-1, 4, 4, 6)):
            if ((ids[:, column] < 0) | (ids[:, column] > maximum)).any():
                raise ValueError('Invalid semantic metadata ID')
        phase, obj, predicate, color = ids.unbind(dim=1)
        goal = torch.cat((self.object(obj), self.predicate(predicate)), dim=-1)
        return self.phase(phase) + goal + self.color(color)
