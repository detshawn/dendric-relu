import numpy as np
import torch
from tensorboardX import SummaryWriter


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
