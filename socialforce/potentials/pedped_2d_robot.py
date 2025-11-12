"""Interaction potentials."""

import math
import torch

from .pedped_1d import PedPedPotentialMLP #, PedPedPotential
from .pedped_1d_robot import PedPedPotential

class PedPedPotential2D(torch.nn.Module):
    """Ped-ped interaction potential based on distance b and relative angle.

    v0 is in m^2 / s^2.
    sigma is in m.
    """
    delta_t_step = PedPedPotential.delta_t_step
    robot_delta_t_step = PedPedPotential.robot_delta_t_step

    b = PedPedPotential.b
    value_b = PedPedPotential.value_b
    r_ab = staticmethod(PedPedPotential.r_ab)
    norm_r_ab = staticmethod(PedPedPotential.norm_r_ab)
    grad_r_ab = PedPedPotential.grad_r_ab
    grad_r_ab_ = PedPedPotential.grad_r_ab_
    forward = PedPedPotential.forward

    def __init__(self, v0=2.1, sigma=0.3, asymmetry=0.0,
                 v0_robot=2.1, sigma_robot=0.3, asymmetry_robot=0.0):
        super().__init__()

        self.v0 = v0
        self.sigma = sigma
        self.asymmetry = asymmetry
        self.register_buffer('rot90', torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))

        self.robot_index = 0
        self.v0_robot = v0_robot  # used in PedPedPotential.value_b
        self.sigma_robot = sigma_robot  # used in PedPedPotential.value_b
        self.v0_robot_std = None
        self.sigma_robot_std = None
        self.v_sigma_dict = {}  # buffer for v0,sigma given diferent human sizes
        self.asymmetry_robot = asymmetry_robot
        self.asymmetry_robot_std = None

    def update_robot_params(self, v0, sigma, asymmetry, v0_std=None, sigma_std=None, asymmetry_std=None):
        self.v0_robot = v0
        self.sigma_robot = sigma
        self.asymmetry_robot = asymmetry
        self.v0_robot_std = v0_std
        self.sigma_robot_std = sigma_std
        self.asymmetry_robot_std = asymmetry_std
        self.v_sigma_dict = {}


    def sample_robot_asymmetry(self, N):
        """Sample N asymmetry values for the robot from Normal(mean, std)."""""
        if self.asymmetry_robot_std is None:
            return self.asymmetry_robot

        asymmetry_robot_samples = torch.normal(
            mean=self.asymmetry_robot,
            std=self.asymmetry_robot_std,
            size=(N,),
            device=self.device
        )

        return asymmetry_robot_samples


    @staticmethod
    def parallel_d(r_ab, desired_directions):
        parallel_d = torch.einsum('abj,bj->ab', r_ab, desired_directions)
        torch.diagonal(parallel_d)[:] = 0.0
        return parallel_d

    @staticmethod
    def asymmetry_factor(asymmetry, perpendicular_d):
        return 1.0 / math.log(2.0) * torch.nn.functional.softplus(asymmetry * perpendicular_d)

    def perpendicular_d(self, r_ab, desired_directions):
        desired_directions_p = torch.matmul(desired_directions, self.rot90)
        perpendicular_d = torch.einsum('abj,bj->ab', r_ab, desired_directions_p)
        torch.diagonal(perpendicular_d)[:] = 0.0
        return perpendicular_d

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        b = self.b(r_ab, speeds, desired_directions)
        value = self.value_b(b)
        if self.asymmetry != 0.0:
            perpendicular_d = self.perpendicular_d(r_ab, desired_directions)
            asym_factor = self.asymmetry_factor(self.asymmetry, perpendicular_d)
            ## Added by Rashid Alyassi
            if self.asymmetry_robot != 0.0 or self.asymmetry_robot_std is not None:
                asymmetry_robot_samples = self.sample_robot_asymmetry(r_ab.shape[0])
                asym_factor[self.robot_index, :] = self.asymmetry_factor(self.asymmetry_robot, perpendicular_d[self.robot_index, :])
                asym_factor[:, self.robot_index] = self.asymmetry_factor(asymmetry_robot_samples, perpendicular_d[:, self.robot_index])

            value = asym_factor * value
        return value


class PedPedPotentialDiamond(PedPedPotential2D):
    """Ped-ped interaction potential."""
    def __init__(self, v0=2.1, sigma=0.3, *,
                 asymmetry=0.0,
                 asymmetry_offset=0.0,
                 asymmetry_angle=0.0,
                 speed_dependent=True):
        super().__init__()

        self.v0 = v0
        self.sigma = sigma
        self.asymmetry = asymmetry
        self.asymmetry_offset = asymmetry_offset
        self.speed_dependent = speed_dependent

        if asymmetry_angle:
            asymmetry_angle *= math.pi / 180.0
            self.register_buffer('asymmetry_rotation', torch.tensor([
                [math.cos(asymmetry_angle), math.sin(asymmetry_angle)],
                [-math.sin(asymmetry_angle), math.cos(asymmetry_angle)],
            ]))
        else:
            self.asymmetry_rotation = None

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        assert speeds.requires_grad is False
        assert desired_directions.requires_grad is False

        if self.asymmetry_rotation is not None:
            desired_directions = torch.einsum(
                'ai,ij->aj', desired_directions, self.asymmetry_rotation)

        parallel_d = self.parallel_d(r_ab, desired_directions)
        perpendicular_d = self.perpendicular_d(r_ab, desired_directions)
        if self.asymmetry_offset:
            perpendicular_d += self.asymmetry_offset

        sigma_perp = self.sigma
        sigma_parallel = self.sigma
        if self.asymmetry or self.speed_dependent:
            sigma_perp = torch.full_like(perpendicular_d, self.sigma, requires_grad=False)
            sigma_parallel = torch.full_like(parallel_d, self.sigma, requires_grad=False)
            if self.asymmetry != 0.0:
                sigma_perp *= 1.0 + self.asymmetry * torch.sign(perpendicular_d)
            if self.speed_dependent:
                front = parallel_d > 0.0
                # Given speeds are for pedestrian b. Need to make it of shape ab
                # and repeat the values along the dimension for pedestrian a.
                speeds_ab = torch.repeat_interleave(
                    torch.unsqueeze(speeds, 0), sigma_parallel.shape[0], dim=0)
                sigma_parallel[front] += self.delta_t_step * speeds_ab[front]

        l1 = torch.abs(perpendicular_d) / sigma_perp + torch.abs(parallel_d) / sigma_parallel
        return self.v0 * torch.clamp_min(1.0 - 0.5 * l1, 0.0)


class PedPedPotentialMLP1p1D(PedPedPotential2D):
    """Ped-ped interaction potential."""
    def __init__(self, *, hidden_units=5):
        super().__init__()

        self.pedped_b = PedPedPotentialMLP(hidden_units=hidden_units)

        perpendicular_lin1 = torch.nn.Linear(1, hidden_units)
        perpendicular_lin2 = torch.nn.Linear(hidden_units, 1)
        self.mlp_perpendicular = torch.nn.Sequential(
            perpendicular_lin1, torch.nn.Softplus(),
            perpendicular_lin2, torch.nn.Softplus(),
        )

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        assert speeds.requires_grad is False
        assert desired_directions.requires_grad is False
        out_b = self.pedped_b.value_r_ab(r_ab, speeds, desired_directions)

        perpendicular_d = self.perpendicular_d(r_ab, desired_directions)
        out_perpendicular = self.mlp_perpendicular(
            perpendicular_d.reshape(-1, 1)
        ).view(r_ab[:, :, 0].shape)

        out = torch.mul(out_b, out_perpendicular)
        return out


class PedPedPotentialMLP2D(PedPedPotential2D):
    """Ped-ped interaction potential."""
    def __init__(self, *,
                 hidden_units=16,
                 n_hidden_layers=1,
                 n_fourier_features=None,
                 n_spherical_features=None,
                 fourier_scale=1.0,
                 tanh_range=2.0):
        super().__init__()

        #################
        input_features = 3

        assert not (n_fourier_features and n_spherical_features)
        if n_fourier_features:
            fourier_featurizer = torch.randn((n_fourier_features // 2,))
            self.register_buffer('featurizer', fourier_featurizer)
            input_features *= n_fourier_features
        elif n_spherical_features:
            spherical_featurizer = torch.randn((input_features, n_spherical_features // 2))
            self.register_buffer('featurizer', spherical_featurizer)
            input_features = n_spherical_features
        else:
            self.featurizer = None
        self.fourier_scale = fourier_scale
        self.tanh_range = tanh_range

        lin_in = torch.nn.Linear(input_features, hidden_units)
        lin_hidden = [torch.nn.Linear(hidden_units, hidden_units)
                      for _ in range(n_hidden_layers)]
        lin_out = torch.nn.Linear(hidden_units, 1)

        # activation_function = torch.nn.Softplus
        activation_function = lambda: torch.nn.Softplus(beta=5)  # pylint: disable=unnecessary-lambda-assignment
        self.mlp = torch.nn.Sequential(
            lin_in, activation_function(),
            *[layer for lin in lin_hidden for layer in (lin, activation_function())],
            lin_out, torch.nn.Softplus(),
        )

    def input_features(self, r_ab, speeds, desired_directions):
        input_vector = torch.stack((
            self.b(r_ab, speeds, desired_directions),
            self.perpendicular_d(r_ab, desired_directions),
            self.parallel_d(r_ab, desired_directions),
        ), dim=-1)

        if self.tanh_range is not None:
            input_vector = self.tanh_range * torch.tanh(input_vector / self.tanh_range)
        if self.featurizer is not None:
            input_vector = self.fourier_features(input_vector)

        return input_vector

    def fourier_features(self, input_vector):
        input_vector = 2.0 * math.pi / self.fourier_scale * input_vector
        if len(self.featurizer.shape) > 1:  # spherical
            ff = torch.matmul(input_vector, self.featurizer)
        else:
            ff = torch.matmul(torch.unsqueeze(input_vector, -1),
                              torch.unsqueeze(self.featurizer, 0))
            ff = torch.reshape(ff, list(ff.shape)[:-2] + [-1])
        ff = torch.cat((torch.sin(ff), torch.cos(ff)), dim=-1)
        return ff

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        input_vector = self.input_features(r_ab, speeds, desired_directions)
        flattened = input_vector.view(-1, input_vector.shape[-1])
        out = self.mlp(flattened).view(r_ab[:, :, 0].shape)
        out = out.clone()

        # Added By Rashid Alyassi
        # Robot uses standard 2D potential
        b = self.b(r_ab, speeds, desired_directions)  # ellipse scalar
        std_val = self.value_b(b)  # V0 * exp(-b/sigma)
        if getattr(self, "asymmetry_robot", 0.0) != 0.0 or getattr(self, "asymmetry_robot_std", None) is not None:
            sample_asymmetry_robot = self.sample_robot_asymmetry(r_ab.shape[0])
            perp = self.perpendicular_d(r_ab, desired_directions)  # signed lateral offset
            std_val = self.asymmetry_factor(sample_asymmetry_robot, perp) * std_val

        out[self.robot_index, :] = std_val[self.robot_index, :]  # robot affects others
        out[:, self.robot_index] = std_val[:, self.robot_index]  # others affect robot

        torch.diagonal(out)[:] = 0.0  # removes self force
        return out
