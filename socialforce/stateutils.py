"""Utility functions to process state."""

import torch

def _desired_directions(state):
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 6:8] - state[:, 0:2]

    # support for prediction without given destination:
    # "desired_direction" is in the direction of the current velocity
    invalid_destination = torch.isnan(destination_vectors[:, 0])
    destination_vectors[invalid_destination] = state[invalid_destination, 2:4]

    norm_factors = torch.linalg.norm(destination_vectors, ord=2, dim=-1)
    norm_factors[norm_factors == 0.0] = 1.0
    return destination_vectors / norm_factors.unsqueeze(-1)


@torch.jit.script
def desired_directions(state: torch.Tensor) -> torch.Tensor:  # Added by Rashid Alyassi: faster version (tested)
    destination_vectors = state[:, 6:8] - state[:, 0:2]

    # Replace NaN goals with velocity (branchless)
    nan_mask = torch.isnan(destination_vectors)
    destination_vectors = torch.where(nan_mask, state[:, 2:4], destination_vectors)

    # Normalize safely
    norm = destination_vectors.square().sum(-1, keepdim=True).clamp_min(1e-8).sqrt()
    return destination_vectors / norm




def speeds(state):
    """Return the speeds corresponding to a given state."""
    return torch.linalg.norm(state[:, 2:4], ord=2, dim=-1)

