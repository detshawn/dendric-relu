import numpy as np
import torch
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import math


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def val_to_dict(self):
        return {self.name: self.val}

    def avg_to_dict(self):
        return {self.name: self.avg}

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterList(object):
    def __init__(self, meters):
        self.meters = meters

    def list_to_dict(self):
        l = len(self.meters)
        d = None
        if l == 1:
            d = dict(self.meters[0].avg_to_dict())
        elif l > 1:
            d = dict(self.meters[0].avg_to_dict(), **self.meters[1].avg_to_dict())
            for meter in self.meters[2:]:
                d.update(meter.avg_to_dict())
        return d


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def sigmoid(x):
    return 1/(1+math.exp(-x))


def build_focal_loss(_gamma):
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    def focal_loss_fn(x, target, gamma=_gamma):
        log_p = logsoftmax(x)
        weights = (1-torch.exp(log_p.clone().detach())).pow(gamma)
        loss = []
        mean_factor = []
        for log_p_i, target_i, w_i in zip(log_p, target, weights):
            weight = w_i[target_i]
            loss.append(-weight * log_p_i[target_i])
            mean_factor.append(weight)
        loss = torch.stack(loss, dim=0).sum()
        mean_factor = torch.stack(mean_factor, dim=0).sum()
        if mean_factor > 0:
            loss = loss / mean_factor
        return loss
    return focal_loss_fn


def KLLoss(z_mean, z_log_var):
    # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss = kl_loss * -0.5
    return -5e-4 * torch.mean(torch.mean(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var), dim=-1))


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.

    # Example

    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def scalars_summary(self, tag, dictionary, step):
        self.writer.add_scalars(tag, dictionary, step)
        self.writer.flush()

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)
        self.writer.flush()

    def graph_summary(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model=input_to_model, profile_with_cuda=torch.cuda.is_available())
        self.writer.flush()

    def audio_summary(self, tag, value, step, sr):
        self.writer.add_audio(tag, value, step, sample_rate=sr)
        self.writer.flush()

    def image_summary(self, tag, value, step, dataformats='CHW'):
        self.writer.add_image(tag, value, step, dataformats=dataformats)
        self.writer.flush()


def plot_results(data,
                 model_name="vae",
                 tag=""):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    z_means, y_labels, i_successes = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, tag+"_"+"vae_mean.png")
    # display a 2D plot of the classes in the latent space

    p_success = np.sum(i_successes == 1) / i_successes.shape[0]
    classes = ['total', 'success', 'failure']
    z_mean_tsne = {}
    z_mean_tsne['total'] = TSNE(learning_rate=300, init='pca').fit_transform(z_means)
    z_mean_tsne['success'] = TSNE(learning_rate=300, init='pca').fit_transform(z_means[i_successes == 1])
    z_mean_tsne['failure'] = TSNE(learning_rate=300, init='pca').fit_transform(z_means[i_successes == 0])

    y_class = {}
    y_class['total'] = y_labels
    y_class['success'] = y_labels[i_successes == 1]
    y_class['failure'] = y_labels[i_successes == 0]

    f, axes = plt.subplots(nrows=int((len(classes)+1)/2), ncols=2, figsize=(10*int((len(classes)+1)/2), 8*2))

    for k, c in enumerate(classes):
        i, j = int(k/2), k%2
        x, y = z_mean_tsne[c][:, 0], z_mean_tsne[c][:, 1]
        im = axes[i, j].scatter(x, y, c=y_class[c])
        plt.colorbar(im, ax=axes[i, j])
        axes[i, j].set_xlabel("z[0]")
        axes[i, j].set_ylabel("z[1]")
        axes[i, j].set_title('TSNE latent_vector for ' + c)

    f.savefig(filename)
    # plt.show()
    # plt.close(f)

    return f


def fig2rgb_array(fig, expand=True):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)
