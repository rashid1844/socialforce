"""Interaction potentials."""

import torch

from .. import stateutils


class PedPedPotential(torch.nn.Module):
    """Ped-ped interaction potential based on distance b.

    v0 is in m^2 / s^2.
    sigma is in m.

    Changes by Rashid Alyassi
    - Unique value for V0 and sigma for the robot
        - std allows sampling of v0 & sigma for human-robot interactions
        - (note: robot-human interactions always use the mean, since planner shouldn't predict robot behavior)
    - Add GPU Device
    - use_autograd: originally autograd used to support MLP, we added use_autograd to disable it if classical SFM is used
    - grad_r_ab_analytic: analytic gradient, assuming a circle
    - grad_r_ab_analytic_ellipse: analytic gradient, assuming an ellipse
    """
    delta_t_step = 0.4
    robot_delta_t_step = 0.4

    def __init__(self, v0=2.1, sigma=0.3, v0_robot=2.1, sigma_robot=0.3, use_autograd=True):
        super().__init__()
        self.v0 = v0
        self.sigma = sigma
        self.robot_index = 0
        self.v0_robot = v0_robot
        self.sigma_robot = sigma_robot
        self.v0_robot_std = None
        self.sigma_robot_std = None
        self.v_sigma_dict = {}  # buffer for v0,sigma given diferent human sizes

        self.use_autograd = use_autograd


    def grad_r_ab_analytic(self, state):
        """Analytical gradient of V(r_ab) for exponential potential."""
        r_ab = self.r_ab(state)
        speeds = stateutils.speeds(state)
        desired_directions = stateutils.desired_directions(state)

        # Compute scalar b and V(b)
        b = self.b(r_ab, speeds, desired_directions)
        v_mat = self.value_b(b)

        # Compute analytic gradient: dV/dr = -(V/σb) * r_ab
        norm_r = self.norm_r_ab(r_ab).unsqueeze(-1).clamp_min(1e-6)
        grad = -v_mat.unsqueeze(-1) / (self.sigma * norm_r) * r_ab

        # Zero diagonal (no self-interaction)
        torch.diagonal(grad[..., 0])[:] = 0.0
        torch.diagonal(grad[..., 1])[:] = 0.0
        return grad

    def grad_r_ab_analytic_ellipse(self, state):
        """
        Analytic ∂V/∂r_ab for the elliptical b used in self.b(...).
        Returns [N, N, 2].
        """
        eps = 1e-8
        r_ab = self.r_ab(state)  # [N,N,2]
        speeds = stateutils.speeds(state)  # [N]
        e_b = stateutils.desired_directions(state)  # [N,2]

        # Recreate the same Δ and s used in b(...)
        speeds_b = speeds.unsqueeze(0)  # [1,N]
        speeds_b_abc = speeds_b.unsqueeze(2)  # [1,N,1]
        delta_vec = self.delta_t_step * speeds_b_abc * e_b.unsqueeze(0)  # [1,N,2]
        s = (self.delta_t_step * speeds_b)  # [1,N] (scalar per beta)

        # A = ||r||, B = ||r - Δ||
        A = torch.linalg.norm(r_ab, dim=-1).clamp_min(eps)  # [N,N]
        B = torch.linalg.norm(r_ab - delta_vec, dim=-1).clamp_min(eps)  # [N,N]

        # b = 0.5*sqrt((A+B)^2 - s^2)
        S = (A + B) ** 2 - s ** 2  # [N,N]
        S = torch.clamp(S, min=eps)
        b = 0.5 * torch.sqrt(S)  # [N,N]

        # V(b) and sigma matrix (use your cached v_sigma_dict for robot row/col)
        V = self.value_b(b)  # [N,N]

        # sigma_mat: replicate your logic from value_b to support robot params
        N = b.shape[0]
        if N in self.v_sigma_dict:
            v0_mat, sigma_mat = self.v_sigma_dict[N]
        else:
            v0_mat = torch.full((N, N), self.v0, device=b.device)
            sigma_mat = torch.full((N, N), self.sigma, device=b.device)
            v0_robot_samples, sigma_robot_samples = self.sample_robot_v0_sigma(N)
            v0_mat[self.robot_index, :] = self.v0_robot
            v0_mat[:, self.robot_index] = v0_robot_samples
            sigma_mat[self.robot_index, :] = self.sigma_robot
            sigma_mat[:, self.robot_index] = sigma_robot_samples
            self.v_sigma_dict[N] = (v0_mat, sigma_mat)
        # dV/db = -V/sigma
        dVdb = -V / sigma_mat  # [N,N]

        # Unit directions r/A and (r-Δ)/B
        r_unit = r_ab / A.unsqueeze(-1)  # [N,N,2]
        r_minus_d = r_ab - delta_vec  # [N,N,2]
        rmd_unit = r_minus_d / B.unsqueeze(-1)  # [N,N,2]

        # db/dr = (A+B)/(4b) * (r/A + (r-Δ)/B)
        pref = ((A + B) / (4.0 * b)).unsqueeze(-1)  # [N,N,1]
        db_dr = pref * (r_unit + rmd_unit)  # [N,N,2]

        # ∂V/∂r = dV/db * db/dr
        grad = dVdb.unsqueeze(-1) * db_dr  # [N,N,2]

        # Zero self-interactions on diagonal
        torch.diagonal(grad[..., 0]).zero_()
        torch.diagonal(grad[..., 1]).zero_()
        return grad

    def update_robot_params(self, v0, sigma, v0_std=None, sigma_std=None):
        self.v0_robot = v0
        self.sigma_robot = sigma
        self.v0_robot_std = v0_std
        self.sigma_robot_std = sigma_std
        self.v_sigma_dict = {}


    def sample_robot_v0_sigma(self, N):
        """Sample N v0 and sigma values for the robot from Normal(mean, std)."""
        if self.v0_robot_std is None or self.sigma_robot_std is None:
            return self.v0_robot, self.sigma_robot

        v0_robot_samples = torch.normal(
            mean=self.v0_robot,
            std=self.v0_robot_std,
            size=(N,),
            device=self.device
        )
        sigma_robot_samples = torch.normal(
            mean=self.sigma_robot,
            std=self.sigma_robot_std,
            size=(N,),
            device=self.device
        )
        return v0_robot_samples, sigma_robot_samples





    def value_b(self, b):
        """Value of potential parametrized with b."""
        N = b.shape[0]
        if N in self.v_sigma_dict:
            v0_mat, sigma_mat = self.v_sigma_dict[N]
        else:
            v0_mat = torch.full((N, N), self.v0, device=b.device)
            sigma_mat = torch.full((N, N), self.sigma, device=b.device)
            # Robot row: robot reacting to others
            v0_mat[self.robot_index, :] = self.v0_robot
            sigma_mat[self.robot_index, :] = self.sigma_robot

            # Robot column: others reacting to robot
            v0_robot_samples, sigma_robot_samples = self.sample_robot_v0_sigma(N)
            v0_mat[:, self.robot_index] = v0_robot_samples
            sigma_mat[:, self.robot_index] = sigma_robot_samples
            self.v_sigma_dict[N] = (v0_mat, sigma_mat)

        return v0_mat * torch.exp(-b / sigma_mat)


    def b(self, r_ab, speeds, desired_directions):
        """Calculate b."""
        speeds_b = speeds.unsqueeze(0)
        speeds_b_abc = speeds_b.unsqueeze(2)  # abc = alpha, beta, coordinates
        e_b = desired_directions.unsqueeze(0)

        speeds_b_abc_delta = self.delta_t_step * speeds_b_abc
        speeds_b_abc_delta[0, self.robot_index] *= self.robot_delta_t_step / self.delta_t_step
        speeds_b_delta = self.delta_t_step * speeds_b
        speeds_b_delta[0, self.robot_index] *= self.robot_delta_t_step / self.delta_t_step


        in_sqrt = (
            self.norm_r_ab(r_ab)
            + self.norm_r_ab(r_ab - speeds_b_abc_delta * e_b)
        )**2 - (speeds_b_delta)**2

        # torch.diagonal(in_sqrt)[:] = 0.0  # protect forward pass
        in_sqrt = torch.clamp(in_sqrt, min=1e-8)
        out = 0.5 * torch.sqrt(in_sqrt)
        # torch.diagonal(out)[:] = 0.0  # protect backward pass

        return out


    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        b = self.b(r_ab, speeds, desired_directions)
        return self.value_b(b)

    @staticmethod
    def r_ab(state):
        """Construct r_ab using broadcasting."""
        r = state[:, 0:2]
        r_a0 = r.unsqueeze(1)
        r_0b = r.unsqueeze(0).detach()  # detach others
        r_ab = r_a0 - r_0b
        torch.diagonal(r_ab)[:] = 0.0  # detach diagonal gradients
        return r_ab

    def forward(self, state):
        speeds = stateutils.speeds(state).detach()
        desired_directions = stateutils.desired_directions(state).detach()
        return self.value_r_ab(self.r_ab(state), speeds, desired_directions)

    def grad_r_ab_finite_difference(self, state, delta=1e-3):
        """Compute gradient wrt r_ab using finite difference differentiation."""
        r_ab = self.r_ab(state[:, 0:2])
        speeds = stateutils.speeds(state)
        desired_directions = stateutils.desired_directions(state)

        dx = torch.tensor([[[delta, 0.0]]], device=state.device)
        dy = torch.tensor([[[0.0, delta]]], device=state.device)

        v = self.value_r_ab(r_ab, speeds, desired_directions)
        dvdx = (self.value_r_ab(r_ab + dx, speeds, desired_directions) - v) / delta
        dvdy = (self.value_r_ab(r_ab + dy, speeds, desired_directions) - v) / delta

        # remove gradients from self-intereactions
        torch.diagonal(dvdx)[:] = 0.0
        torch.diagonal(dvdy)[:] = 0.0

        return torch.stack((dvdx, dvdy), dim=-1, device=state.device)

    def grad_r_ab(self, state):
        """Compute gradient wrt r_ab using autograd."""
        if not hasattr(self, 'autograd') or self.use_autograd:
            speeds = stateutils.speeds(state).detach()
            desired_directions = stateutils.desired_directions(state).detach()
            r_ab = self.r_ab(state)
            return self.grad_r_ab_(r_ab, speeds, desired_directions)
        else:
            #return self.grad_r_ab_analytic(state)
            return self.grad_r_ab_analytic_ellipse(state)

    def grad_r_ab_(self, r_ab, speeds, desired_directions):
        """Compute gradient wrt r_ab using autograd."""
        def compute(r_ab):
            return self.value_r_ab(r_ab, speeds, desired_directions)

        with torch.enable_grad():
            vector = torch.ones(r_ab.shape[0:2], requires_grad=False, device=r_ab.device)
            _, r_ab_grad = torch.autograd.functional.vjp(
                compute, r_ab, vector,
                create_graph=True, strict=True)

        return r_ab_grad

    @staticmethod
    def norm_r_ab(r_ab):
        """Norm of r_ab.

        Special treatment of diagonal terms for backpropagation.

        Without this treatment, backpropagating through a norm of a
        zero vector gives nan gradients.
        """
        out = torch.linalg.norm(r_ab, ord=2, dim=2, keepdim=False)

        # only take the upper and lower triangles and leaving the
        # diagonal at zero and do it in a differentiable way
        # without inplace ops
        out = torch.triu(out, diagonal=1) + torch.tril(out, diagonal=-1)

        return out


class PedPedPotentialWall(PedPedPotential):
    """Ped-ped interaction potential based on distance b.

    v0 is in m^2 / s^2.
    sigma is in m.
    """
    delta_t_step = 0.4

    def __init__(self, sigma=0.3, w=0.1):
        super().__init__()
        self.sigma = sigma
        self.w = w

    def value_b(self, b):
        """Value of potential parametrized with b."""
        return torch.exp(-(b - self.sigma) / self.w)


class PedPedPotentialMLP(PedPedPotential):
    """Ped-ped interaction potential."""

    def __init__(self, *, hidden_units=5, small_init=False, dropout_p=0.0):
        super().__init__()

        lin1 = torch.nn.Linear(1, hidden_units)
        lin2 = torch.nn.Linear(hidden_units, 1)

        # initialize
        if small_init:
            torch.nn.init.normal_(lin1.weight, std=0.03)
            torch.nn.init.normal_(lin1.bias, std=0.03)
            torch.nn.init.normal_(lin2.weight, std=0.03)
            torch.nn.init.normal_(lin2.bias, std=0.03)

        if dropout_p == 0.0:
            self.mlp = torch.nn.Sequential(
                lin1, torch.nn.Softplus(),
                lin2, torch.nn.Softplus(),
            )
        else:
            self.mlp = torch.nn.Sequential(
                lin1, torch.nn.Softplus(), torch.nn.Dropout(dropout_p),
                lin2, torch.nn.Softplus(),
            )

    def __value_b(self, b):
        """Calculate value given b."""
        b = torch.clamp(b, max=100.0)
        return self.mlp(b.view(-1, 1)).view(b.shape)


    def value_b(self, b):
        """Calculate value given b.
        Uses MLP for human-human interactions,
        and analytic SFM potential for robot-related interactions.
        """
        b = torch.clamp(b, max=100.0)
        v_b = self.mlp(b.view(-1, 1)).view(b.shape)

        # Added by Rashid Alyassi
        # --- Robot-specific override (analytic exponential) ---
        N = b.shape[0]
        if N in self.v_sigma_dict:
            v0_mat, sigma_mat = self.v_sigma_dict[N]
        else:
            v0_mat = torch.full((N, N), self.v0, device=b.device)
            sigma_mat = torch.full((N, N), self.sigma, device=b.device)
            # Robot parameters (row & column)
            v0_robot_samples, sigma_robot_samples = self.sample_robot_v0_sigma(N)
            v0_mat[self.robot_index, :] = self.v0_robot
            v0_mat[:, self.robot_index] = v0_robot_samples
            sigma_mat[self.robot_index, :] = self.sigma_robot
            sigma_mat[:, self.robot_index] = sigma_robot_samples
            self.v_sigma_dict[N] = (v0_mat, sigma_mat)

        v_robot = v0_mat * torch.exp(-b / sigma_mat)

        # ---- Replace robot-related rows & columns ----
        v_b[self.robot_index, :] = v_robot[self.robot_index, :]
        v_b[:, self.robot_index] = v_robot[:, self.robot_index]

        return v_b
