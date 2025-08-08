import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor


class SumScalarizer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.dynamic = False

    def forward(self, losses: Tensor) -> Tensor:
        loss = losses.sum()
        return loss


class MeanScalarizer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.dynamic = False

    def forward(self, losses: Tensor) -> Tensor:
        loss = losses.mean()
        return loss


class ScaleInvariantScalarizer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.dynamic = False

    def forward(self, losses: Tensor) -> Tensor:
        return (losses[losses > 0]).log().sum()


class DWAScalarizer(nn.Module):
    def __init__(
        self, n_tasks: int, device: torch.device, interval_steps: int = 500, alpha: float = 0.1, T: float = 1.0, log=False, **kwargs
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.interval_steps = interval_steps
        self.alpha = alpha
        self.T = T
        self.eps = 1e-8
        self.log = log

        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("weights", torch.full((n_tasks,), 1.0, device=device))
        self.register_buffer("ema_losses", torch.zeros(n_tasks, device=device))
        self.register_buffer("prev_ema_losses", torch.zeros(n_tasks, device=device))

        self.dynamic = True

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        if self.log:
            active = losses != 0
            return (self.weights[active] * losses[active].log1p()).sum()
        else:
            return (self.weights * losses).sum()

    def update(self, losses: torch.Tensor) -> None:
        if dist.is_initialized():
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= dist.get_world_size()

        active = losses != 0

        if self.log:
            losses[active] = losses[active].log1p()

        self.step_count += 1
        if self.step_count.item() == 1:
            self.ema_losses.copy_(losses)
            self.prev_ema_losses.copy_(losses)
            return

        self.ema_losses.mul_(1 - self.alpha).add_(self.alpha, losses)

        if (self.step_count.item() % self.interval_steps) == 0:
            n_active = int(active.sum().item())

            speeds = self.ema_losses[active] / (self.prev_ema_losses[active] + self.eps)
            exp_s = torch.exp(speeds / self.T)
            weights = torch.zeros_like(self.weights)
            weights[active] = n_active * (exp_s / exp_s.sum())

            if not dist.is_initialized() or dist.get_rank() == 0:
                self.weights.copy_(weights)

            if dist.is_initialized():
                dist.broadcast(self.weights, src=0)

            self.prev_ema_losses.copy_(self.ema_losses)


class SmoothTchebycheffScalarizer(nn.Module):
    def __init__(self, n_tasks: int, device: torch.device, mu: float, warmup_steps: int, eps: float = 1e-20, **kwargs):
        super().__init__()
        self.n_tasks = n_tasks
        self.mu = mu
        self.warmup_steps = warmup_steps
        self.eps = eps
        self.dynamic = True

        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("warmup_sum", torch.zeros(n_tasks, device=device))
        self.register_buffer("warmup_count", torch.zeros(1, device=device))
        self.register_buffer("baseline_losses", torch.zeros(n_tasks, device=device))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        active = losses > 0

        if self.step_count.item() <= self.warmup_steps:
            return torch.log(losses[active]).sum()

        n_active = active.sum().float()

        ratio = losses[active] / (self.baseline_losses[active] + self.eps) + self.eps
        log_ratio = torch.log(ratio)
        mmax = log_ratio.max().detach()

        return self.mu * ((log_ratio - mmax) / self.mu).exp().sum().log() * n_active

    def update(self, losses: torch.Tensor) -> None:
        if dist.is_initialized():
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= dist.get_world_size()

        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            self.warmup_sum += losses.detach()
            self.warmup_count += 1

            if self.step_count == self.warmup_steps:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    self.baseline_losses.copy_(self.warmup_sum / self.warmup_count)

                if dist.is_initialized():
                    dist.broadcast(self.baseline_losses, src=0)

                print(f"Baseline losses: {self.baseline_losses}")


class LogEMAScalarizer(nn.Module):
    def __init__(self, n_tasks: int, device: torch.device, interval_steps: int = 500, ema_alpha: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.n_tasks = n_tasks
        self.interval_steps = interval_steps
        self.ema_alpha = ema_alpha
        self.eps = eps
        self.dynamic = True

        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("ema_baseline", torch.ones(n_tasks, device=device))
        self.register_buffer("weights", torch.ones(n_tasks, device=device))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        return torch.log1p(losses * self.weights).sum()

    def update(self, losses: torch.Tensor) -> None:
        if dist.is_initialized():
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= dist.get_world_size()

        self.step_count += 1

        if self.step_count.item() == 1:
            self.ema_baseline.copy_(losses)
        else:
            self.ema_baseline.mul_(1 - self.ema_alpha).add_(self.ema_alpha, losses)

        if (self.step_count.item() % self.interval_steps) == 0:
            active = losses != 0
            weights = torch.zeros_like(self.weights)
            weights[active] = 1 / (self.ema_baseline[active] + self.eps)
            self.weights.copy_(weights)

            if dist.is_initialized():
                dist.broadcast(self.weights, src=0)


class EMAScalarizer(nn.Module):
    def __init__(self, n_tasks: int, device: torch.device, interval_steps: int = 500, ema_alpha: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.n_tasks = n_tasks
        self.interval_steps = interval_steps
        self.ema_alpha = ema_alpha
        self.eps = eps
        self.dynamic = True

        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("ema_baseline", torch.ones(n_tasks, device=device))
        self.register_buffer("weights", torch.ones(n_tasks, device=device))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        return (losses * self.weights).sum()

    def update(self, losses: torch.Tensor) -> None:
        if dist.is_initialized():
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= dist.get_world_size()

        self.step_count += 1

        if self.step_count.item() == 1:
            self.ema_baseline.copy_(losses)
        else:
            self.ema_baseline.mul_(1 - self.ema_alpha).add_(self.ema_alpha, losses)

        if (self.step_count.item() % self.interval_steps) == 0:
            active = losses != 0
            weights = torch.zeros_like(self.weights)
            weights[active] = torch.clamp(1.0 / (self.ema_baseline[active] + self.eps), min=0.0, max=1e6)
            self.weights.copy_(weights)

            if dist.is_initialized():
                dist.broadcast(self.weights, src=0)


SCALARIZER_CLASS_MAP = {
    "Sum": SumScalarizer,
    "Mean": MeanScalarizer,
    "ScaleInvariant": ScaleInvariantScalarizer,
    "DWA": DWAScalarizer,
    "SmoothTchebycheff": SmoothTchebycheffScalarizer,
    "LogEMA": LogEMAScalarizer,
    "EMA": EMAScalarizer,
}
