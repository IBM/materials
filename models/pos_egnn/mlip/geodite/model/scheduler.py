from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, StepLR, _LRScheduler


class WarmupLinearDecayLR(_LRScheduler):
    """
    Warmup + single-run linear decay, then hold at eta_min.
    """

    def __init__(self, optimizer, warmup_iters: int, T_max: int, eta_max=None, eta_min=0.0, last_epoch: int = -1):
        if not isinstance(warmup_iters, int) or warmup_iters < 0:
            raise ValueError(f"warmup_iters must be integer ≥ 0, got {warmup_iters}")
        if not isinstance(T_max, int) or T_max <= 0:
            raise ValueError(f"T_max must be positive integer, got {T_max}")

        self.warmup_iters = warmup_iters
        self.T_max = T_max
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

        if eta_max is None:
            self.eta_max = list(self.base_lrs)
        elif isinstance(eta_max, (list, tuple)):
            self.eta_max = list(eta_max)
        else:
            self.eta_max = [eta_max] * len(optimizer.param_groups)

        if isinstance(eta_min, (list, tuple)):
            self.eta_min = list(eta_min)
        else:
            self.eta_min = [eta_min] * len(optimizer.param_groups)

        self.lr_delta = [(max_lr - min_lr) / T_max for max_lr, min_lr in zip(self.eta_max, self.eta_min)]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [
                base + (max_lr - base) * (self.last_epoch / float(self.warmup_iters)) for base, max_lr in zip(self.base_lrs, self.eta_max)
            ]

        t = self.last_epoch - self.warmup_iters
        if t <= self.T_max:
            return [max(max_lr - delta * t, min_lr) for max_lr, delta, min_lr in zip(self.eta_max, self.lr_delta, self.eta_min)]

        return list(self.eta_min)


SCHEDULER_CLASS_MAP = {
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "CosineAnnealingLR": CosineAnnealingLR,
    "LinearLR": LinearLR,
    "StepLR": StepLR,
    "WarmupLinearDecayLR": WarmupLinearDecayLR,
}
