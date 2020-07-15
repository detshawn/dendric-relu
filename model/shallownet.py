from torch import nn
import torch
from .activations import DendricLinear

import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


class Encoder(nn.Module):
    def __init__(self, config, dropout=True, dendric=False, multi_position=1):
        super(Encoder, self).__init__()
        layer_config = config['layers']
        self.is_guess_net = ('guess_net' in config)

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

        if self.is_guess_net:
            guess_net_config = config['guess_net']
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

            if self.is_guess_net and guess:
                outs.append(out.clone().detach())
                if f'{i+1}' in self.guess_nets_flag:
                    guess_in.append(self.guess_nets_flag[f'{i+1}'](out.clone().detach()))

        intermediates = {}
        if self.is_guess_net and guess:
            guess_out = torch.cat(guess_in, dim=1)
            guess_out = self.guesser(guess_out)
            guess_out = self.sigmoid(guess_out)
            intermediates['guess_out'] = guess_out

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

        intermediates['z_mean'] = z_mean
        intermediates['z_log_var'] = z_log_var

        z = self._sampling(z_mean, z_log_var)

        return z, intermediates


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
    def __init__(self, in_features, out_features, device, n_hiddens=2, config=None, sub_kwargs=None):
        super(ShallowNet, self).__init__()
        self.config = config
        self.device = device
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

        sub_kwargs = sub_kwargs or {}
        self.encoder = Encoder(self.config['Encoder'], **sub_kwargs)
        self.classifier = Classifier(self.config['Classifier'], **sub_kwargs)
        self.decoder = Decoder(self.config['Decoder'], **sub_kwargs)

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.], device=device))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.], device=device))
        # Loss
        self.ge2e_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, dropout=True, enc_kwargs=None):
        enc_kwargs = enc_kwargs or {}
        out = x
        z_sample, enc_intermediates = self.encoder(out, dropout=dropout, **enc_kwargs)
        cl = self.classifier(z_sample, dropout=dropout)
        out = self.decoder(z_sample, dropout=dropout)
        return out, (cl, z_sample, enc_intermediates)

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch,
        #                             utterances_per_speaker).to(self.device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)

        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix

    def ge2e_loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker,
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.device)
        loss = self.ge2e_loss_fn(sim_matrix, target)

        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().to('cpu').numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            try:
                fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
                eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            except:
                eer = 0.0

        return loss, eer
