import torch
import torch.nn.functional as F



class Conv2dFilter(torch.nn.Module):
    """
    Convolve a filter on data.

    Args:
        filter (str): Filter to convolve data with.
        stride (int or tuple): Stride of the convolution. the default is 1.
        padding (int or tuple): Padding added to all four sides of the input. the default is 0.
    """

    def __init__(self, filters, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.filters = filters

    def __call__(self, input):
        print(self.filters.shape)
        return F.conv2d(input,self.filters, stride=self.stride, padding=self.padding)