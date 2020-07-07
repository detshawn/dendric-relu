from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, layer_config, dropout=True):
        super(Encoder, self).__init__()
        layers = []
        for i in range(0, len(layer_config)-2):
            layer = nn.Sequential()
            layer.add_module(f'l{i}', nn.Linear(in_features=layer_config[i], out_features=layer_config[i+1]))
            layer.add_module(f'bn{i}', nn.BatchNorm1d(num_features=layer_config[i+1]))
            layers.append(layer)
        two_layers = [nn.Linear(in_features=layer_config[-2], out_features=layer_config[-1]),
                      nn.Linear(in_features=layer_config[-2], out_features=layer_config[-1])]
        layers.append(nn.ModuleList(two_layers))
        self.layers = nn.ModuleList(layers)
        self.do = nn.Dropout(p=0.15) if dropout else None
        self.relu = nn.ReLU()

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

    def forward(self, x):
        out = x
        for l in self.layers[:-1]:
            out = l(out)
            if self.do is not None:
                out = self.do(out)
            out = self.relu(out)

        z_mean= self.layers[-1][0](out)
        z_log_var = self.layers[-1][1](out)

        z = self._sampling(z_mean, z_log_var)

        return z, (z_mean, z_log_var)


class Decoder(nn.Module):
    def __init__(self, layer_config, dropout=True):
        super(Decoder, self).__init__()
        layers = []
        for i in range(0, len(layer_config)-2):
            layer = nn.Sequential()
            layer.add_module(f'l{i}', nn.Linear(in_features=layer_config[i], out_features=layer_config[i+1]))
            layer.add_module(f'bn{i}', nn.BatchNorm1d(num_features=layer_config[i+1]))
            layers.append(layer)
        layers.append(nn.Linear(in_features=layer_config[-2], out_features=layer_config[-1]))
        self.layers = nn.ModuleList(layers)
        self.do = nn.Dropout(p=0.15) if dropout else None
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x
        for l in self.layers[:-1]:
            out = l(out)
            if self.do is not None:
                out = self.do(out)
            out = self.relu(out)

        out = self.layers[-1](out)
        out = self.sigmoid(out)

        return out


class Classifier(nn.Module):
    def __init__(self, layer_config, dropout=True):
        super(Classifier, self).__init__()
        layers = []
        for i in range(0, len(layer_config)-1):
            layer = nn.Sequential()
            layer.add_module(f'l{i}', nn.Linear(in_features=layer_config[i], out_features=layer_config[i+1]))
            if i+2 != len(layer_config):
                layer.add_module(f'bn{i}', nn.BatchNorm1d(num_features=layer_config[i+1]))
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.do = nn.Dropout(p=0.15) if dropout else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        for l in self.layers[:-1]:
            out = l(out)
            if self.do is not None:
                out = self.do(out)
            out = self.relu(out)
        out = self.layers[-1](out)

        return out


class ShallowNet(nn.Module):
    def __init__(self, in_features, out_features, n_hiddens=2, layer_config=None):
        super(ShallowNet, self).__init__()
        self.layer_config = layer_config
        if layer_config is None:
            self.layer_config = {}
            encoder_config = [in_features]
            step = int((in_features-out_features)/(n_hiddens+1))
            for i in range(n_hiddens):
                encoder_config.append(out_features+step*i)
            encoder_config.append(out_features)

            self.layer_config['encoder'] = encoder_config
            self.layer_config['classifier'] = [out_features, 8, 10]
            self.layer_config['decoder'] = encoder_config[-1::-1]

        self.encoder = Encoder(self.layer_config['encoder'])
        self.classifier = Classifier(self.layer_config['classifier'])
        self.decoder = Decoder(self.layer_config['decoder'])

    def forward(self, x):
        out = x
        z_sample, (z_mean, z_log_var) = self.encoder(out)
        cl = self.classifier(z_sample)
        out = self.decoder(z_sample)
        return out, (cl, z_sample, z_mean, z_log_var)
