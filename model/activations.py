import torch
from torch import nn


class Hyper(nn.Module):
    __constants__ = ['offset', 'value', 'inplace']

    def __init__(self, offset=.5, value=None, inplace=False):
        super(Hyper, self).__init__()
        self.offset = offset
        self.value = value if value is not None else offset * 2.
        self.inplace = inplace

    def forward(self, input):
        return self._step(input, self.offset, self.value)

    def _step(self, input, offset, value):
        return (input >= offset) * value

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'offset={}, value={}{}'.format(self.offset, self.value, inplace_str)


class Hypo(nn.Module):
    __constants__ = ['min', 'max', 'inplace']

    def __init__(self, min=0., max=1., inplace=False):
        super(Hypo, self).__init__()
        if min > max:
            print(f'Error: min({min}) is not supposed to be larger than max({max})!')
            exit(1)
        self.min = min
        self.max = max
        self.inplace = inplace

    def forward(self, input):
        return torch.clamp(input, min=self.min, max=self.max)

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'min={}, max={}{}'.format(self.min, self.max, inplace_str)
