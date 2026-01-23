from .snapshot import SnapshotDecoder

TASK_TO_DECODER = {
    "Snapshot": SnapshotDecoder,
}
DECODER_TO_TASK = {value.__name__: key for key, value in TASK_TO_DECODER.items()}

__all__ = [
    "TASK_TO_DECODER",
    "DECODER_TO_TASK",
]
