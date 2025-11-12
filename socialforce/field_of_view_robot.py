"""Field of view computation."""

import math
import torch


import math
import torch


class FieldOfView:
    """Compute field of view prefactors.

    The field of view angle twophi is given in degrees.
    out_of_view_factor is C in the paper.

    Changes by Rashid Alyassi
    - Unique FOV for the robot
    - Add GPU device
    """

    out_of_view_factor = 0.5

    def __init__(self, twophi=200.0, twophi_robot=220.0, robot_index=0, device='cpu'):
        self.robot_index = robot_index
        self.cosphi_human = math.cos(twophi / 2.0 / 180.0 * math.pi)
        self.cosphi_robot = twophi_robot  #math.cos(twophi_robot / 2.0 / 180.0 * math.pi)
        self.cosphi_dict = {}  # cache (N, device) → cosphi_self
        self.twophi_robot_std = None

        self.device = device

    def update_robot(self, twophi_robot, twophi_robot_std=None):
        self.cosphi_robot = twophi_robot #math.cos(twophi_robot / 2.0 / 180.0 * math.pi)
        self.twophi_robot_std = twophi_robot_std
        self.cosphi_dict = {}


    def sample_robot_cosphi(self, N):
        """Sample N cosphi values for the robot from Normal(mean, std)."""
        if self.twophi_robot_std is None:
            return math.cos(self.twophi_robot / 2.0 / 180.0 * math.pi)
        twophi_samples = torch.normal(
            mean=self.twophi_robot_mean,
            std=self.twophi_robot_std,
            size=(N,),
            device=self.device
        )
        return torch.cos(twophi_samples / 2.0 / 180.0 * math.pi)  # [N]


    def __call__(self, e, f):
        """
        Compute field-of-view weighting factor.

        Args:
            e: [N, 2] tensor of desired directions (normalized)
            f: [N, N, 2] tensor of pairwise vectors (e.g., r_ab)
        Returns:
            [N, N] tensor of FoV weights
        """
        N = e.shape[0]

        # --- Cached cosine thresholds per agent ---
        if N in self.cosphi_dict:
            cosphi_self = self.cosphi_dict[N]
        else:
            cosphi_self = torch.full((N, 1), self.cosphi_human, device=self.device)
            # Robot row: robot perceiving others
            cosphi_self[self.robot_index] = math.cos(self.twophi_robot / 2.0 / 180.0 * math.pi)#self.cosphi_robot
            # Robot column: humans perceiving the robot
            cosphi_self = cosphi_self.expand(-1, N).clone()
            cosphi_self[:, self.robot_index] = self.sample_robot_cosphi(N)  #self.cosphi_robot
            # Cache for reuse
            self.cosphi_dict[N] = cosphi_self

        # --- Field-of-view computation ---
        cosphi_l = torch.einsum('aj,abj->ab', (e, f))
        norm_f = torch.linalg.norm(f, ord=2, dim=-1)
        in_sight = cosphi_l > norm_f * cosphi_self

        out = torch.full_like(cosphi_l, self.out_of_view_factor)
        out[in_sight] = 1.0
        torch.diagonal(out)[:] = 0.0
        return out
