from torch import nn
import torch
from .activations import DendricLinear


class Encoder(nn.Module):
    def __init__(self, config, dropout=True, dendric=False, multi_position=1):
        super(Encoder, self).__init__()
        layer_config = config['layers']
        guess_net_config = config['guess_net']

        layers = []
        self.dendric = dendric
        for i in range(0, len(layer_config)-2):
            layer = nn.Sequential()
            if self.dendric and i != 0:
                hypers, hypos = [pow(2, 3-i)] * pow(2, 4-i), [pow(2, 3-i)] * pow(2, 4-i)
                layer.add_module(f'dl{i}', DendricLinear(in_features=layer_config[i], out_features=layer_config[i+1],
                                                        hypers=hypers, hypos=hypos, multi_position=multi_position))
            else:
                layer.add_module(f'l{i}', nn.Linear(in_features=layer_config[i], out_features=layer_config[i+1]))
            layer.add_module(f'bn{i}', nn.BatchNorm1d(num_features=layer_config[i+1]))
            layers.append(layer)
        two_layers = [nn.Linear(in_features=layer_config[-2], out_features=layer_config[-1]),
                      nn.Linear(in_features=layer_config[-2], out_features=layer_config[-1])]
        layers.append(nn.ModuleList(two_layers))
        self.layers = nn.ModuleList(layers)

        guess_nets = []
        guess_concat_length = 0
        self.guess_nets_flag = {}
        for i in range(len(guess_net_config)):
            if len(guess_net_config[i]) > 0:
                guess_net = nn.Sequential()
                in_features = layer_config[i]
                for j, guess_l in enumerate(guess_net_config[i]):
                    guess_net.add_module(f'guess_l{j}', nn.Linear(in_features=in_features, out_features=guess_l))
                    guess_net.add_module(f'bn{j}', nn.BatchNorm1d(num_features=guess_l))
                    in_features = guess_l
                guess_nets.append(guess_net)
                self.guess_nets_flag[f'{i}'] = guess_net
                guess_concat_length += in_features
        self.guess_nets = nn.ModuleList(guess_nets)
        self.guesser = nn.Linear(in_features=guess_concat_length, out_features=1)

        ext_layer_config = config['ext_layers']
        ext_layers = []
        self.dendric = dendric
        for i in range(0, len(layer_config)-2):
            in_features = ext_layer_config[i] + (layer_config[i] if i == 1 else 0)
            layer = nn.Sequential()
            if self.dendric and i != 0:
                hypers, hypos = [pow(2, 3-i)] * pow(2, 4-i), [pow(2, 3-i)] * pow(2, 4-i)
                layer.add_module(f'dl{i}', DendricLinear(in_features=in_features, out_features=ext_layer_config[i+1],
                                                        hypers=hypers, hypos=hypos, multi_position=multi_position))
            else:
                layer.add_module(f'l{i}', nn.Linear(in_features=in_features, out_features=ext_layer_config[i+1]))
            layer.add_module(f'bn{i}', nn.BatchNorm1d(num_features=ext_layer_config[i+1]))
            ext_layers.append(layer)
        two_layers = [nn.Linear(in_features=layer_config[-2]+ext_layer_config[-2], out_features=layer_config[-1]),
                      nn.Linear(in_features=layer_config[-2]+ext_layer_config[-2], out_features=layer_config[-1])]
        ext_layers.append(nn.ModuleList(two_layers))
        self.ext_layers = nn.ModuleList(ext_layers)

        self.do = nn.Dropout(p=0.15) if dropout else None
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _sampling(z_mean, z_log_var):
        '''
        Reparameterization trick by sampling from an isotropic unit Gaussian
         instead of sampling from Q(z|X), sample epsilon = N(0, I)
          z = z_mean + sqrt(var) * epsilon

        :param args: mean and log variance of Q(z|X)
        :return z: sampled latent vector
        '''

        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5*z_log_var) * epsilon

    def forward(self, x, dropout=True, guess=False, ext_training=False):
        out = x
        outs = []
        guess_in = []
        for i, l in enumerate(self.layers[:-1]):
            out = l(out)
            if self.do is not None and dropout:
                out = self.do(out)
            out = self.relu(out)
            outs.append(out.clone().detach())
            if f'{i+1}' in self.guess_nets_flag:
                guess_in.append(self.guess_nets_flag[f'{i+1}'](out.clone().detach()))

        guess_out = torch.cat(guess_in, dim=1)
        guess_out = self.guesser(guess_out)
        guess_out = self.sigmoid(guess_out)

        if guess:
            if ext_training:
                out = x
                for i, l in enumerate(self.ext_layers[:-1]):
                    out = torch.cat((outs[i-1], out), dim=1) if i == 1 else out
                    out = l(out)
                    if self.do is not None and dropout:
                        out = self.do(out)
                    out = self.relu(out)

                out = torch.cat((outs[-1], out), dim=1)
                z_mean = self.ext_layers[-1][0](out)
                z_log_var = self.ext_layers[-1][1](out)
            else:
                guessed_mask = guess_out < .5
                out = out * (~guessed_mask).view(out.size()[0], -1).repeat((1, out.size()[1]))
                z_mean = self.layers[-1][0](out)
                z_log_var = self.layers[-1][1](out)

                out = x * guessed_mask.view(x.size()[0], -1).repeat((1, x.size()[1]))
                for i, l in enumerate(self.ext_layers[:-1]):
                    out = torch.cat((outs[i - 1], out), dim=1) if i == 1 else out
                    out = l(out)
                    if self.do is not None and dropout:
                        out = self.do(out)
                    out = self.relu(out)

                out = torch.cat((outs[-1], out), dim=1)
                z_mean = z_mean + self.ext_layers[-1][0](out)
                z_log_var = z_log_var + self.ext_layers[-1][1](out)

        else:
            z_mean = self.layers[-1][0](out)
            z_log_var = self.layers[-1][1](out)

        z = self._sampling(z_mean, z_log_var)

        return z, (z_mean, z_log_var, guess_out)


class Decoder(nn.Module):
    def __init__(self, config, dropout=True, dendric=False, multi_position=1):
        super(Decoder, self).__init__()
        layer_config = config['layers']

        layers = []
        self.dendric = dendric
        for i in range(0, len(layer_config)-2):
            layer = nn.Sequential()
            if self.dendric and i != 0:
                hypers, hypos = [pow(2, i)] * pow(2, i+1), [pow(2, i)] * pow(2, i+1)
                layer.add_module(f'dl{i}', DendricLinear(in_features=layer_config[i], out_features=layer_config[i+1],
                                                        hypers=hypers, hypos=hypos, multi_position=multi_position))
            else:
                layer.add_module(f'l{i}', nn.Linear(in_features=layer_config[i], out_features=layer_config[i+1]))
            layer.add_module(f'bn{i}', nn.BatchNorm1d(num_features=layer_config[i+1]))
            layers.append(layer)
        layers.append(nn.Linear(in_features=layer_config[-2], out_features=layer_config[-1]))
        self.layers = nn.ModuleList(layers)
        self.do = nn.Dropout(p=0.15) if dropout else None
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, dropout=True):
        out = x
        for l in self.layers[:-1]:
            out = l(out)
            if self.do is not None and dropout:
                out = self.do(out)
            out = self.relu(out)

        out = self.layers[-1](out)
        out = self.sigmoid(out)

        return out


class Classifier(nn.Module):
    def __init__(self, config, dropout=True, dendric=False, multi_position=1):
        super(Classifier, self).__init__()
        layer_config = config['layers']

        layers = []
        self.dendric = dendric
        for i in range(0, len(layer_config)-2):
            layer = nn.Sequential()
            layer.add_module(f'l{i}', nn.Linear(in_features=layer_config[i], out_features=layer_config[i + 1]))
            layers.append(layer)
        layer = nn.Sequential()
        if self.dendric:
            layer.add_module(f'dl{len(layer_config)-1}', DendricLinear(in_features=layer_config[-2], out_features=layer_config[-1],
                                                     hypers=[2], hypos=[2],
                                                     multi_position=multi_position, final_relu=False))
        else:
            layer.add_module(f'l{len(layer_config)-1}', nn.Linear(in_features=layer_config[-2], out_features=layer_config[-1]))
        layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.do = nn.Dropout(p=0.15) if dropout else None
        self.relu = nn.ReLU()

    def forward(self, x, dropout=True):
        out = x
        for l in self.layers[:-1]:
            out = l(out)
            if self.do is not None and dropout:
                out = self.do(out)
            out = self.relu(out)
        out = self.layers[-1](out)

        return out


class ShallowNet(nn.Module):
    def __init__(self, in_features, out_features, n_hiddens=2, config=None, dendric=False, multi_position=1):
        super(ShallowNet, self).__init__()
        self.config = config
        if config is None:
            self.config = {'Encoder':{}, 'Decoder':{}, 'Classifier':{}}
            encoder_config = [in_features]
            step = int((in_features-out_features)/(n_hiddens+1))
            for i in range(n_hiddens):
                encoder_config.append(out_features+step*i)
            encoder_config.append(out_features)

            self.layer_config['Encoder']['layers'] = encoder_config
            self.layer_config['Classifier']['layers'] = [out_features, 8, 10]
            self.layer_config['Decoder']['layers'] = encoder_config[-1::-1]

        self.encoder = Encoder(self.config['Encoder'], dendric=dendric, multi_position=multi_position)
        self.classifier = Classifier(self.config['Classifier'], dendric=dendric, multi_position=multi_position)
        self.decoder = Decoder(self.config['Decoder'], dendric=dendric, multi_position=multi_position)

    def forward(self, x, dropout=True, guess=False, ext_training=True):
        out = x
        z_sample, (z_mean, z_log_var, guess_out) = self.encoder(out, dropout=dropout, guess=guess, ext_training=ext_training)
        cl = self.classifier(z_sample, dropout=dropout)
        out = self.decoder(z_sample, dropout=dropout)
        return out, (cl, z_sample, z_mean, z_log_var, guess_out)
