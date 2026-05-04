"""TorchCDE neural CDE core used by OAT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

try:
    import torchcde
except ImportError as exc:  # pragma: no cover
    torchcde = None
    _TORCHCDE_IMPORT_ERROR = exc
else:
    _TORCHCDE_IMPORT_ERROR = None

try:
    from torchdiffeq import odeint
except ImportError as exc:  # pragma: no cover
    odeint = None
    _TORCHDIFFEQ_IMPORT_ERROR = exc
else:
    _TORCHDIFFEQ_IMPORT_ERROR = None


def _build_mlp(in_dim: int, out_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    if depth <= 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers: list[nn.Module] = []
    cur_dim = in_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(cur_dim, hidden_dim))
        layers.append(nn.Softplus())
        cur_dim = hidden_dim
    layers.append(nn.Linear(cur_dim, out_dim))
    return nn.Sequential(*layers)


class _CDEVectorField(nn.Module):
    def __init__(self, hidden_channels: int, input_channels: int, hidden_dim: int, depth: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.net = _build_mlp(hidden_channels, hidden_channels * input_channels, hidden_dim, depth)

    def forward(self, _t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(z.shape[0], self.hidden_channels, self.input_channels)


class _ControlGate(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int, depth: int, eps: float):
        super().__init__()
        self.eps = float(max(eps, 1e-8))
        self.net = _build_mlp(input_channels, input_channels, hidden_dim, depth)

    def forward(self, d_x_dt: torch.Tensor) -> torch.Tensor:
        steer = torch.tanh(self.net(d_x_dt))
        return F.normalize(steer, p=2, dim=-1, eps=self.eps)


class _GatedControl(nn.Module):
    def __init__(self, base_control: nn.Module, gate: nn.Module):
        super().__init__()
        self.base_control = base_control
        self.gate = gate

    @property
    def interval(self):
        return self.base_control.interval

    @property
    def grid_points(self):
        return self.base_control.grid_points

    def evaluate(self, t: torch.Tensor) -> torch.Tensor:
        return self.base_control.evaluate(t)

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        derivative = self.base_control.derivative(t)
        return self.gate(derivative) * derivative


class _HiddenBridgeODEFunc(nn.Module):
    def __init__(self, hidden_channels: int, hidden_dim: int, depth: int):
        super().__init__()
        self.net = _build_mlp(hidden_channels, hidden_channels, hidden_dim, depth)

    def forward(self, _t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class OATModel(nn.Module):
    """Question-conditioned neural CDE with gated controls."""

    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        hidden_dim: int = config.OAT_HIDDEN,
        depth: int = config.OAT_DEPTH,
        interpolation: str = config.OAT_INTERPOLATION,
        solver: str = config.OAT_SOLVER,
        adjoint: bool = config.OAT_ADJOINT,
        control_gate_hidden: int = config.OAT_CONTROL_GATE_HIDDEN,
        control_gate_depth: int = config.OAT_CONTROL_GATE_DEPTH,
        control_gate_eps: float = config.OAT_CONTROL_GATE_EPS,
        h0_norm_reg: float = config.OAT_H0_NORM_REG,
        h1_bridge_hidden: int = config.OAT_H1_BRIDGE_HIDDEN,
        h1_bridge_depth: int = config.OAT_H1_BRIDGE_DEPTH,
        h1_bridge_solver: str = config.OAT_H1_BRIDGE_SOLVER,
    ):
        super().__init__()
        if torchcde is None:
            raise ImportError("torchcde is required. Install with `pip install torchcde`.") from _TORCHCDE_IMPORT_ERROR
        if odeint is None:
            raise ImportError("torchdiffeq is required. Install with `pip install torchdiffeq`.") from _TORCHDIFFEQ_IMPORT_ERROR
        if interpolation not in {"cubic", "linear", "rectilinear"}:
            raise ValueError("interpolation must be one of: cubic, linear, rectilinear.")

        self.latent_dim = latent_dim
        self.input_channels = latent_dim + 1
        self.hidden_channels = hidden_dim
        self.interpolation = interpolation
        self.solver = solver
        self.adjoint = bool(adjoint)
        self.h0_norm_reg = float(max(h0_norm_reg, 0.0))
        self.h1_bridge_solver = h1_bridge_solver

        self.initial = _build_mlp(self.input_channels, hidden_dim, hidden_dim, depth)
        self.func = _CDEVectorField(hidden_dim, self.input_channels, hidden_dim, depth)
        self.readout = nn.Linear(hidden_dim, latent_dim)
        self.question_h0_init = _build_mlp(latent_dim, hidden_dim, hidden_dim, depth)
        self.h1_bridge_func = _HiddenBridgeODEFunc(hidden_dim, h1_bridge_hidden, h1_bridge_depth)
        self.control_gate = _ControlGate(self.input_channels, control_gate_hidden, control_gate_depth, control_gate_eps)

    @staticmethod
    def _default_time_grid(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.arange(length, device=device, dtype=dtype)
        if length > 1:
            t = t / (length - 1)
        return t

    def _prepare_inputs(self, z_obs: torch.Tensor, time_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z_obs.ndim != 2:
            raise ValueError(f"Expected z_obs shape (T, d), got {tuple(z_obs.shape)}.")
        if z_obs.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {z_obs.shape[1]}.")
        t_len = int(z_obs.shape[0])
        if t_len < 2:
            raise ValueError("OAT forward requires at least 2 time points.")
        t = time_points.reshape(-1).to(device=z_obs.device, dtype=z_obs.dtype)
        if t.numel() != t_len or torch.any(t[1:] <= t[:-1]):
            t = self._default_time_grid(t_len, z_obs.device, z_obs.dtype)
        x = torch.cat([t.unsqueeze(-1), z_obs], dim=-1).unsqueeze(0)
        return x, t

    def _build_control(self, x: torch.Tensor):
        if self.interpolation == "cubic":
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            return _GatedControl(torchcde.CubicSpline(coeffs), self.control_gate)
        if self.interpolation == "rectilinear":
            coeffs = torchcde.linear_interpolation_coeffs(x, rectilinear=0)
            return _GatedControl(torchcde.LinearInterpolation(coeffs), self.control_gate)
        coeffs = torchcde.linear_interpolation_coeffs(x)
        return _GatedControl(torchcde.LinearInterpolation(coeffs), self.control_gate)

    def _integrate_hidden_path(
        self,
        control: nn.Module,
        t: torch.Tensor,
        hidden_init_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x0 = control.evaluate(t[0])
        z0 = self.initial(x0) if hidden_init_override is None else hidden_init_override
        kwargs = {}
        if self.solver == "dopri5" and self.interpolation in {"linear", "rectilinear"}:
            kwargs["options"] = {"jump_t": control.grid_points}
        z_t = torchcde.cdeint(X=control, func=self.func, z0=z0, t=t, method=self.solver, adjoint=self.adjoint, **kwargs)
        if z_t.shape[0] == t.shape[0]:
            return z_t[:, 0, :]
        return z_t[0, :, :]

    def _integrate_h0_to_h1(self, h0: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        ts = torch.stack([torch.zeros((), device=h0.device, dtype=h0.dtype), dt.to(device=h0.device, dtype=h0.dtype)])
        return odeint(self.h1_bridge_func, h0, ts, method=self.h1_bridge_solver)[-1]

    @staticmethod
    def _step_errors(z_true: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
        return (z_true - z_pred).pow(2).sum(dim=-1)

    def _question_h0_step_errors(self, z_obs: torch.Tensor, time_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z_obs.shape[0] < 2:
            return torch.zeros(0, device=z_obs.device), torch.zeros(self.hidden_channels, device=z_obs.device)
        h0 = self.question_h0_init(z_obs[0]).unsqueeze(0)
        t_full = time_points.reshape(-1).to(device=z_obs.device, dtype=z_obs.dtype)
        if t_full.numel() != z_obs.shape[0] or torch.any(t_full[1:] <= t_full[:-1]):
            t_full = self._default_time_grid(z_obs.shape[0], z_obs.device, z_obs.dtype)
        h1 = self._integrate_h0_to_h1(h0, (t_full[1] - t_full[0]).clamp(min=1e-6))
        first_err = self._step_errors(z_obs[1], self.readout(h1).squeeze(0))
        if z_obs.shape[0] == 2:
            return first_err.unsqueeze(0), h0.squeeze(0)
        x_resp, t_resp = self._prepare_inputs(z_obs[1:], time_points[1:])
        h_resp = self._integrate_hidden_path(self._build_control(x_resp), t_resp, hidden_init_override=h1)
        later_pred = self.readout(h_resp[:-1])
        later_err = self._step_errors(z_obs[2:], later_pred)
        return torch.cat([first_err.unsqueeze(0), later_err], dim=0), h0.squeeze(0)

    def forward(self, z_obs: torch.Tensor, time_points: torch.Tensor, **_kwargs) -> dict:
        if z_obs.shape[0] < 2:
            zero = torch.tensor(0.0, device=z_obs.device)
            return {"loss": zero, "recon_per_step": torch.zeros(z_obs.shape[0], device=z_obs.device)}
        step_err, h0 = self._question_h0_step_errors(z_obs, time_points)
        loss = step_err.mean()
        if self.h0_norm_reg > 0.0:
            loss = loss + self.h0_norm_reg * h0.pow(2).mean()
        recon_per_step = torch.cat([torch.zeros(1, device=z_obs.device), step_err], dim=0)
        return {"loss": loss, "recon_per_step": recon_per_step.detach()}

    def anomaly_scores(self, z_obs: torch.Tensor, time_points: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(z_obs, time_points)["recon_per_step"]
