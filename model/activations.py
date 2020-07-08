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
                 hypers=[], hypos=[], T=1., final_relu=True, multi_position=1):
        super(DendricLinear, self).__init__()
        self.relus = {'hyper': hypers,
                      'hypo': hypos,
                      'standard': [in_features-sum(hypers+hypos)]}

        self.hyper_relu = Hyper(offset=T/2)
        self.hypo_relu = Hypo(max=T/2)
        self.final_relu = nn.ReLU() if final_relu else None
        self.multi_position = multi_position
        self.circular_step = min(sum(self.relus['hyper']), 1)
        self.circular_offset = 0 # min(int(min(self.relus['hyper'] + self.relus['hypo'])/2), 0)

        sublinear_sizes = [int(out_features/self.multi_position)]*(self.multi_position-1) + \
                          [out_features-int(out_features/self.multi_position)*(self.multi_position-1)]
        hyper_linears = []
        hypo_linears = []
        standard_linears = []
        for sublinear_size in sublinear_sizes:
            hyper_sublinears = []
            for i in self.relus['hyper']:
                hyper_sublinears.append(nn.Linear(in_features=i, out_features=sublinear_size))
            hyper_linears.append(nn.ModuleList(hyper_sublinears))

            hypo_sublinears = []
            for i in self.relus['hypo']:
                hypo_sublinears.append(nn.Linear(in_features=i, out_features=sublinear_size))
            hypo_linears.append(nn.ModuleList(hypo_sublinears))

            standard_linears.append(nn.Linear(in_features=self.relus['standard'][0], out_features=sublinear_size))

        self.hyper_linears = nn.ModuleList(hyper_linears)
        self.hypo_linears = nn.ModuleList(hypo_linears)
        self.standard_linears = nn.ModuleList(standard_linears)

    @staticmethod
    def _circular_shift(x, c):
        c = c % x.size()[1]
        return torch.cat([x[:, c:], x[:, :c]], dim=1)

    def forward(self, input):
        out = []
        for m, (hyper_linears, hypo_linears, standard_linear) in \
                enumerate(zip(self.hyper_linears, self.hypo_linears, self.standard_linears)):
            out_idx = 0
            idx = 0
            c = m * self.circular_step - (m > 0) * self.circular_offset
            input_circ = self._circular_shift(input, c)
            for i, linear in zip(self.relus['hyper'], hyper_linears):
                linear_out = linear(input_circ[:, idx:idx + i])
                if m == 0:
                    out.append([self.hyper_relu(linear_out)])
                else:
                    out[out_idx].append(self.hyper_relu(linear_out))
                    out_idx = out_idx + 1
                idx = idx+i
            for i, linear in zip(self.relus['hypo'], hypo_linears):
                linear_out = linear(input_circ[:, idx:idx + i])
                if m == 0:
                    out.append([self.hypo_relu(linear_out)])
                else:
                    out[out_idx].append(self.hypo_relu(linear_out))
                    out_idx = out_idx + 1
                idx = idx+i
            linear_out = input_circ[:, idx:]
            if m == 0:
                out.append([standard_linear(linear_out)])
            else:
                out[out_idx].append(standard_linear(linear_out))

        out = [torch.cat(item, dim=1) for item in out]
        out = torch.stack(out, dim=2)
        out = torch.sum(out, dim=2)
        if self.final_relu is not None:
            out = self.final_relu(out)

        return out
