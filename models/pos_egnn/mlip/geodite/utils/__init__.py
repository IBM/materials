import torch


@torch.jit.script
class DataInput:
    def __init__(
        self,
        z: torch.Tensor = torch.tensor(0),
        cutoff_edge_index: torch.Tensor = torch.tensor(0),
        cutoff_edge_distance: torch.Tensor = torch.tensor(0),
        cutoff_edge_vec: torch.Tensor = torch.tensor(0),
        batch: torch.Tensor = torch.tensor(0),
        embedding_0: torch.Tensor = torch.tensor(0),
        pos: torch.Tensor = torch.tensor(0),
        displacements: torch.Tensor = torch.tensor(0),
        box: torch.Tensor = torch.tensor(0),
        num_graphs: int = 0,
        cutoff_shifts_idx: torch.Tensor = torch.tensor(0),
        embedding_1: torch.Tensor = torch.tensor(0),
    ):
        self.z = z
        self.cutoff_edge_index = cutoff_edge_index
        self.cutoff_edge_distance = cutoff_edge_distance
        self.cutoff_edge_vec = cutoff_edge_vec
        self.batch = batch
        self.embedding_0 = embedding_0
        self.pos = pos
        self.displacements = displacements
        self.box = box
        self.num_graphs = num_graphs
        self.cutoff_shifts_idx = cutoff_shifts_idx
        self.embedding_1 = embedding_1
