from torch import nn


class LinearNN(nn.Module):
    def __init__(self, layer_config):
        super(LinearNN, self).__init__()
        layers = []
        for i in range(0, len(layer_config)-1):
            layer = nn.Sequential()
            layer.add_module(f'l{i}', nn.Linear(in_features=layer_config[i], out_features=layer_config[i+1]))
            layer.add_module(f'bn{i}', nn.BatchNorm1d(num_features=layer_config[i+1]))
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.do = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        for l in self.layers:
            out = l(out)
            out = self.do(out)
            out = self.relu(out)
        return out


class Classifier(nn.Module):
    def __init__(self, layer_config):
        super(Classifier, self).__init__()
        layers = []
        for i in range(0, len(layer_config)-1):
            layer = nn.Sequential()
            layer.add_module(f'l{i}', nn.Linear(in_features=layer_config[i], out_features=layer_config[i+1]))
            layer.add_module(f'bn{i}', nn.BatchNorm1d(num_features=layer_config[i+1]))
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.do = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        for l in self.layers[:-1]:
            out = l(out)
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

        self.encoder = LinearNN(self.layer_config['encoder'])
        self.classifier = Classifier(self.layer_config['classifier'])
        self.decoder = LinearNN(self.layer_config['decoder'])

    def forward(self, x):
        out = x
        enc = self.encoder(out)
        cl = self.classifier(enc)
        out = self.decoder(enc)
        return out, (cl, enc)
