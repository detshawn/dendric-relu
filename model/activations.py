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


class DendricLinear(nn.Module):
    def __init__(self, in_features, out_features,
                 hypers=[], hypos=[], T=1.):
        super(DendricLinear, self).__init__()
        self.relus = {'hyper': hypers,
                      'hypo': hypos,
                      'standard': [in_features-sum(hypers+hypos)]}

        self.hyper_relu = Hyper(offset=T/2)
        self.hypo_relu = Hypo(max=T/2)
        self.final_relu = nn.ReLU()

        hyper_linears = []
        for i in self.relus['hyper']:
            hyper_linears.append(nn.Linear(in_features=i, out_features=out_features))
        self.hyper_linears = nn.ModuleList(hyper_linears)
        hypo_linears = []
        for i in self.relus['hypo']:
            hypo_linears.append(nn.Linear(in_features=i, out_features=out_features))
        self.hypo_linears = nn.ModuleList(hypo_linears)
        self.standard_linear = nn.Linear(in_features=self.relus['standard'][0], out_features=out_features)

    def forward(self, input):
        out = []
        idx = 0
        for i, linear in zip(self.relus['hyper'], self.hyper_linears):
            out.append(self.hyper_relu(linear(input[:, idx:idx+i])))
            idx = idx+i
        for i, linear in zip(self.relus['hypo'], self.hypo_linears):
            out.append(self.hypo_relu(linear(input[:, idx:idx+i])))
            idx = idx+i
        out.append(self.standard_linear(input[:, idx:]))

        out = torch.stack(out, dim=2)
        out = torch.sum(out, dim=2)
        out = self.final_relu(out)

        return out
