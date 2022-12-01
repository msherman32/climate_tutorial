import torch
import math


class MinMaxRescaleTransformation(torch.nn.Module):
    def __init__(self, r_min, r_max, t_min, t_max):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.t_min = t_min
        self.t_max = t_max

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # TODO: need to floor or round to nearest whole number when forwarding..
        # return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        return ((x - self.r_min) / (self.r_max - self.r_min))(self.t_max - self.t_min) + self.t_min

    # def string(self):
    #     """
    #     Just like any class in Python, you can also define custom method on PyTorch modules
    #     """
    #     return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'


    #  def __init__(self, mean, std, inplace=False):
    #     super().__init__()
    #     _log_api_usage_once(self)
    #     self.mean = mean
    #     self.std = std
    #     self.inplace = inplace

    # def forward(self, tensor: Tensor) -> Tensor:
    #     """
    #     Args:
    #         tensor (Tensor): Tensor image to be normalized.

    #     Returns:
    #         Tensor: Normalized Tensor image.
    #     """
    #     return F.normalize(tensor, self.mean, self.std, self.inplace)

    # def __repr__(self) -> str:
    #     return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"