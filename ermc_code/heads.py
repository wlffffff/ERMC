"""Shared plus zero-initialized RM-phase residual output layers."""
import torch
from torch import nn
from torch.nn import functional as F
from ermc.goals import STATE_COUNT

class PhaseHead(nn.Module):

    def __init__(self, source, phases=STATE_COUNT):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.phases = phases
        self.weight = nn.Parameter(source.weight.detach().repeat(phases, 1))
        self.bias = nn.Parameter(source.bias.detach().repeat(phases))
        self.phase = None

    def forward(self, latent):
        if self.phase is None or self.phase.shape != (latent.shape[0],):
            raise RuntimeError('RM phase must be set from this observation batch')
        all_outputs = F.linear(latent, self.weight, self.bias)
        all_outputs = all_outputs.reshape(-1, self.phases, self.out_features)
        return all_outputs[torch.arange(latent.shape[0], device=latent.device), self.phase]

class ResidualHead(nn.Module):

    def __init__(self, source, coefficient=0.2):
        super().__init__()
        self.shared = source
        self.residual = PhaseHead(source)
        with torch.no_grad():
            self.residual.weight.zero_()
            self.residual.bias.zero_()
        self.coefficient = coefficient
        self.phase = None

    def forward(self, latent):
        self.residual.phase = self.phase
        return self.shared(latent) + self.coefficient * self.residual(latent)
