import torch
from torch import nn


class CondBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 conditional=True, conditional_kwargs=None,
                 track_running_stats=True):
        super(CondBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.conditional = conditional
        self.conditional_kwargs = conditional_kwargs or {}
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.conditional:
            layers = self.conditional_kwargs['layers']
            self.conditional_net = nn.Sequential()
            for i, l in enumerate(layers[:-1]):
                self.conditional_net.add_module(f'l{i}', nn.Linear(l, layers[i + 1]))
                self.conditional_net.add_module(f'relu{i}', nn.ReLU())
        else:
            self.register_parameter('conditional_net', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input, conditional_input=None):
        self._check_input_dim(input)

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.conditional and conditional_input is not None:
            conditions = self.conditional_net(conditional_input).mean(dim=0, keepdim=False)
            return torch.nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                conditions[0] * self.weight + conditions[1],
                conditions[2] * self.bias + conditions[3],
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)

        else:
            return torch.nn.functional.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'conditional={conditional}, conditional_kwargs={conditional_kwargs}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


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
