from ..transforms.pre_transforms import compute_nn_list


class Parser:
    def __init__(self, dataset, cutoff: float = 5.0, pre_transforms=None):
        self.dataset_name = dataset.name
        self.cutoff = cutoff
        self.pre_transforms = pre_transforms

    def parse(self, data):
        data = compute_nn_list(data, self.cutoff)
        data["dataset_name"] = self.dataset_name
        if self.pre_transforms:
            data = self.pre_transforms(data)
        return data
