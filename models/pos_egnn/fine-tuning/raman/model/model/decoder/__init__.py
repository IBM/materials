from .common.classification import BinaryClassificationDecoder
from .common.regression import AtomicContributionDecoder, ScalarRegressionDecoder
from .common.snapshot import SnapshotDecoder
from .crystal.spectra import RamanDecoder

TASK_TO_DECODER = {
    "Binary Classification": BinaryClassificationDecoder,
    "Scalar Regression": ScalarRegressionDecoder,
    "Atomic Contribution": AtomicContributionDecoder,
    "Snapshot": SnapshotDecoder,
    "Spectra": RamanDecoder,
}
DECODER_TO_TASK = {value.__name__: key for key, value in TASK_TO_DECODER.items()}

__all__ = [
    "TASK_TO_DECODER",
    "DECODER_TO_TASK",
]
