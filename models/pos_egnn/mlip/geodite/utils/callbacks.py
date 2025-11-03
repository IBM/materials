# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py

import contextlib
import copy
import os
import threading
from typing import Any, Dict, Iterable

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_info

try:
    from pytorch_lightning import Callback

    LIGHTNING_AVAILABLE = True
except ImportError:
    LightningModule = None
    LIGHTNING_AVAILABLE = False


class ZClip:
    def __init__(
        self,
        alpha=0.97,
        z_thresh=2.5,
        max_grad_norm=1.0,
        eps=1e-6,
        warmup_steps=25,
        mode="zscore",
        clip_option="adaptive_scaling",
        clip_factor=1.0,
        skip_update_on_spike=False,
    ):
        """
        ZClip: An adaptive gradient clipping mechanism using EMA and anomaly detection.

        Args:
            alpha (float): Smoothing factor for mean and variance.
            z_thresh (float): Threshold value.
                              In percentile mode, the clipping threshold is computed as:
                                  EMA mean + (z_thresh × std)
                              In zscore mode, z_thresh is used to determine whether to clip to the baseline
                              or to compute an adaptive threshold.
            max_grad_norm (float or None): Optional maximum gradient norm.
                                           If None, max norm clipping is not applied.
            eps (float): Small constant to avoid division by zero.
            warmup_steps (int): Number of steps to collect gradient norms before EMA initialization.
            mode (str): Clipping mode. Options:
                        - "percentile": Always clip to a fixed threshold defined as :- mean + (z_thresh × std).
                        - "zscore":     Use z-score based clipping.
            clip_option (str): Only used when mode is "zscore". Options:
                        - "adaptive_scaling": If the gradient norm is a strong outlier (z-score > z_thresh),
                                               compute an adaptive threshold as:
                                                   EMA mean + (z_thresh × std) / (z/z_thresh)
                        - "mean": Simply clip to the EMA mean when the z-score exceeds z_thresh.
            clip_factor (float): Multiplier for the (z_thresh * std) in the adaptive scaling threshold.
                                 Default is 1.0. (This can be adjusted to control the aggressiveness of clipping (0.5–0.9 for aggressive settings).)
            skip_update_on_spike (bool): If True, skip updating EMA statistics when a spike is detected.
                                         Default is False.
        """
        self.alpha = alpha
        self.z_thresh = z_thresh
        self.max_grad_norm = max_grad_norm
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.mode = mode.lower()
        self.clip_factor = clip_factor
        self.skip_update_on_spike = skip_update_on_spike

        if self.mode == "zscore":
            assert clip_option in ["mean", "adaptive_scaling"], "For zscore mode, clip_option must be either 'mean' or 'adaptive_scaling'."
            self.clip_option = clip_option.lower()
        elif self.mode == "percentile":
            self.clip_option = None  # clip_option is ignored in percentile mode.
        else:
            raise ValueError("mode must be either 'zscore' or 'percentile'.")

        self.buffer = []
        self.initialized = False
        self.mean = None
        self.var = None

    def _initialize_ema(self):
        self.mean = sum(self.buffer) / len(self.buffer)
        self.var = sum((x - self.mean) ** 2 for x in self.buffer) / len(self.buffer)
        self.initialized = True
        self.buffer = []

    def _update_ema(self, grad_norm):
        # Update EMA for mean and variance using the new effective gradient norm.
        self.mean = self.alpha * self.mean + (1 - self.alpha) * grad_norm
        self.var = self.alpha * self.var + (1 - self.alpha) * (grad_norm - self.mean) ** 2

    def _compute_positive_zscore(self, grad_norm):
        std = self.var**0.5
        z = (grad_norm - self.mean) / (std + self.eps)
        return z, std

    def _compute_grad_norm(self, model):
        """
        Compute the total gradient norm.
          - For FSDP: Sum the squared norms across sharded parameters and perform an all-reduce.
          - For DDP or non-distributed: Use all local parameters.
        """
        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype

        grad_norms = [p.grad.to(dtype).norm(2) for p in model.parameters() if p.grad is not None]
        if not grad_norms:
            return 0.0
        grad_norms_tensor = torch.stack(grad_norms).to(device)
        total_norm = torch.sqrt(torch.sum(torch.pow(grad_norms_tensor, 2)))
        return total_norm.item()

    def _compute_clip_val(self, grad_norm):
        std = self.var**0.5

        # Fixed behavior: In percentile mode, always clip to a threshold computed as:
        #   EMA mean + (z_thresh × std)
        if self.mode == "percentile":
            threshold = self.mean + self.z_thresh * std
            if grad_norm > threshold:
                return threshold
        elif self.mode == "zscore":
            # Compute the z-score for the current gradient norm.
            z, std = self._compute_positive_zscore(grad_norm)
            if z > self.z_thresh:
                if self.clip_option == "adaptive_scaling":
                    eta = z / self.z_thresh  # This rescaling ratio imposes a greater penalty on large outliers.
                    threshold = self.mean + (self.z_thresh * std) / eta
                    threshold = threshold * self.clip_factor
                elif self.clip_option == "mean":
                    threshold = self.mean
                return threshold
        return None  # No clipping needed.

    def apply_in_place_clipping(self, pl_module, global_norm: float, max_global_norm: float):
        """
        Computes the clipping coefficient and applies gradient clipping in-place.

        Args:
            pl_module (LightningModule): The module whose gradients will be clipped.
            global_norm (float): The precomputed global norm of all gradients.
            max_global_norm (float): The maximum allowed global norm.
        """
        # Calculate the clipping coefficient.
        clip_coef = (max_global_norm / (global_norm + 1e-6)) if global_norm > max_global_norm else 1.0

        # If clipping is needed, scale each gradient in-place.
        if clip_coef < 1.0:
            for param in pl_module.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

    def _apply_clipping(self, model, clip_val, total_norm):
        """
        Applies clipping to the gradients by merging the computed clip value with the optional max_grad_norm.
        """
        # Use the computed clip_val if available; otherwise, use the total norm.
        adaptive_clip = clip_val if clip_val is not None else total_norm
        if self.max_grad_norm is not None:
            effective_clip = min(adaptive_clip, self.max_grad_norm)
        else:
            effective_clip = adaptive_clip
        self.apply_in_place_clipping(model, total_norm, effective_clip)
        return effective_clip

    def step(self, model):
        """
        Call this after loss.backward() but before optimizer.step().

        Args:
            model (torch.nn.Module): The model with computed gradients.

        Returns:
            float: The total gradient norm (before clipping) for monitoring.
        """
        total_norm = self._compute_grad_norm(model)

        # During warmup, collect gradient norms without applying clipping.
        if not self.initialized:
            self.buffer.append(total_norm)
            if len(self.buffer) >= self.warmup_steps:
                self._initialize_ema()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            return total_norm

        # Compute the clip value based on the selected mode and clip_option.
        clip_val = self._compute_clip_val(total_norm)
        self._apply_clipping(model, clip_val, total_norm)
        if clip_val is not None and self.skip_update_on_spike:
            return total_norm

        # Update EMA with the effective norm (either the computed clip or the original norm).
        self._update_ema(clip_val if clip_val is not None else total_norm)
        return total_norm


class ZClipLightningCallback(Callback):
    def __init__(self, **zclip_kwargs):
        super().__init__()
        self.zclip = ZClip(**zclip_kwargs)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.zclip.step(pl_module)


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).

    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.

    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        validate_original_weights: Validate the original weights, as apposed to the EMA weights.
        every_n_steps: Apply EMA every N steps.
        cpu_offload: Offload weights to CPU.
    """

    def __init__(
        self,
        decay: float,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
    ):
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        device = pl_module.device if not self.cpu_offload else torch.device("cpu")
        trainer.optimizers = [
            EMAOptimizer(
                optim,
                device=device,
                decay=self.decay,
                every_n_steps=self.every_n_steps,
                current_step=trainer.global_step,
            )
            for optim in trainer.optimizers
            if not isinstance(optim, EMAOptimizer)
        ]

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def _should_validate_ema_weights(self, trainer: "pl.Trainer") -> bool:
        return not self.validate_original_weights and self._ema_initialized(trainer)

    def _ema_initialized(self, trainer: "pl.Trainer") -> bool:
        return any(isinstance(optimizer, EMAOptimizer) for optimizer in trainer.optimizers)

    def swap_model_weights(self, trainer: "pl.Trainer", saving_ema_model: bool = False):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.switch_main_parameter_weights(saving_ema_model)

    @contextlib.contextmanager
    def save_ema_model(self, trainer: "pl.Trainer"):
        """
        Saves an EMA copy of the model + EMA optimizer states for resume.
        """
        self.swap_model_weights(trainer, saving_ema_model=True)
        try:
            yield
        finally:
            self.swap_model_weights(trainer, saving_ema_model=False)

    @contextlib.contextmanager
    def save_original_optimizer_state(self, trainer: "pl.Trainer"):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.save_original_optimizer_state = True
        try:
            yield
        finally:
            for optimizer in trainer.optimizers:
                optimizer.save_original_optimizer_state = False

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint_callback = trainer.checkpoint_callback

        # use the connector as NeMo calls the connector directly in the exp_manager when restoring.
        connector = trainer._checkpoint_connector  # noqa: F841
        # Replace connector._ckpt_path with below to avoid calling into lightning's protected API
        ckpt_path = trainer.ckpt_path

        if ckpt_path and checkpoint_callback is not None and "NeMo" in type(checkpoint_callback).__name__:
            ext = checkpoint_callback.FILE_EXTENSION
            if ckpt_path.endswith(f"-EMA{ext}"):
                rank_zero_info(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = ckpt_path.replace(ext, f"-EMA{ext}")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"))

                checkpoint["optimizer_states"] = ema_state_dict["optimizer_states"]
                del ema_state_dict
                rank_zero_info("EMA state has been restored.")
            else:
                raise MisconfigurationException(
                    "Unable to find the associated EMA weights when re-loading, "
                    f"training will start with new EMA weights. Expected them to be at: {ema_path}",
                )


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple,
        current_model_tuple,
        alpha=(1.0 - decay),
    )


def run_ema_update_cpu(ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None):
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()

    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""
    EMAOptimizer is a wrapper for torch.optim.Optimizer that computes
    Exponential Moving Average of parameters registered in the optimizer.

    EMA parameters are automatically updated after every step of the optimizer
    with the following formula:

        ema_weight = decay * ema_weight + (1 - decay) * training_weight

    To access EMA parameters, use ``swap_ema_weights()`` context manager to
    perform a temporary in-place swap of regular parameters with EMA
    parameters.

    Notes:
        - EMAOptimizer is not compatible with APEX AMP O2.

    Args:
        optimizer (torch.optim.Optimizer): optimizer to wrap
        device (torch.device): device for EMA parameters
        decay (float): decay factor

    Returns:
        returns an instance of torch.optim.Optimizer that computes EMA of
        parameters

    Example:
        model = Model().to(device)
        opt = torch.optim.Adam(model.parameters())

        opt = EMAOptimizer(opt, device, 0.9999)

        for epoch in range(epochs):
            training_loop(model, opt)

            regular_eval_accuracy = evaluate(model)

            with opt.swap_ema_weights():
                ema_eval_accuracy = evaluate(model)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.9999,
        every_n_steps: int = 1,
        current_step: int = 0,
    ):
        self.optimizer = optimizer
        self.decay = decay
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps
        self.save_original_optimizer_state = False

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None

        self.ema_params = ()
        self.in_saving_ema_model_context = False

    def all_parameters(self) -> Iterable[torch.Tensor]:
        return (param for group in self.param_groups for param in group["params"])

    def step(self, closure=None, grad_scaler=None, **kwargs):
        self.join()

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()

            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())

            self.ema_params += tuple(copy.deepcopy(param.data.detach()).to(self.device) for param in opt_params[len(self.ema_params) :])
            self.rebuild_ema_params = False

        if getattr(self.optimizer, "_step_supports_amp_scaling", False) and grad_scaler is not None:
            loss = self.optimizer.step(closure=closure, grad_scaler=grad_scaler)
        else:
            loss = self.optimizer.step(closure)

        if self._should_update_at_step():
            self.update()
        self.current_step += 1
        return loss

    def _should_update_at_step(self) -> bool:
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self):
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            current_model_state = tuple(param.data.to(self.device, non_blocking=True) for param in self.all_parameters())

            if self.device.type == "cuda":
                ema_update(self.ema_params, current_model_state, self.decay)

        if self.device.type == "cpu":
            self.thread = threading.Thread(
                target=run_ema_update_cpu,
                args=(
                    self.ema_params,
                    current_model_state,
                    self.decay,
                    self.stream,
                ),
            )
            self.thread.start()

    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False):
        self.join()
        self.in_saving_ema_model_context = saving_ema_model
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        r"""
        A context manager to in-place swap regular parameters with EMA
        parameters.
        It swaps back to the original regular parameters on context manager
        exit.

        Args:
            enabled (bool): whether the swap should be performed
        """

        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def join(self):
        if self.stream is not None:
            self.stream.synchronize()

        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        self.join()

        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()

        # if we are in the context of saving an EMA model, the EMA weights are in the modules' actual weights
        ema_params = self.ema_params if not self.in_saving_ema_model_context else list(self.all_parameters())
        state_dict = {
            "opt": self.optimizer.state_dict(),
            "ema": ema_params,
            "current_step": self.current_step,
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.join()

        self.optimizer.load_state_dict(state_dict["opt"])
        self.ema_params = tuple(param.to(self.device) for param in copy.deepcopy(state_dict["ema"]))
        self.current_step = state_dict["current_step"]
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]
        self.rebuild_ema_params = False

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True


CALLBACKS_MAPPING = {
    "summary": ModelSummary,
    "checkpoint": ModelCheckpoint,
    "ema": EMA,
    "early_stopping": EarlyStopping,
    "lr_monitor": LearningRateMonitor,
    "zclip": ZClipLightningCallback,
}
