from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: Optional[float] = None
    dropout: float = 0.0
    merge_on_save: bool = True
    freeze_base: bool = True
    include_names: Optional[Sequence[str]] = None
    exclude_names: Optional[Sequence[str]] = None
    preset: Optional[str] = "posegnn"
    log_skipped: bool = False 