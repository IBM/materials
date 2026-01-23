from ...utils.graph import BatchedPeriodicDistance


def compute_nn_list(data, cutoff):
    """
    Utility method to compute the neighbour list

    Parameters
        out_data: a Torch Geometric Data with a pos field
        cutoff: the nearest neighbour cutoff
    """

    distance_fn = BatchedPeriodicDistance(cutoff)
    edge_index, _, _, shifts_idx = distance_fn(data.pos, data.box)

    data.cutoff_edge_index, data.cutoff_shifts_idx = edge_index, shifts_idx

    return data
