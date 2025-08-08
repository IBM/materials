from torch import nn

ACT_CLASS_MAPPING = {"silu": nn.SiLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "gelu": nn.GELU}
