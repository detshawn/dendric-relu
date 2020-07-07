from model.activations import Hyper, Hypo
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from mnist import MNIST
from utils import to_categorical

from model.shallownet import ShallowNet


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        super(MNISTDataset, self).__init__()
        if len(images) != len(labels):
            min_len = min([len(images), len(labels)])
            print(f'Warning: MNISTDataset is cut to the minimum size of {min_len}')
            images, labels = images[:min_len], labels[:min_len]

        self.images = (images/255.).astype(np.float32)
        self.labels = labels.astype(np.long)

    def _to_oh(self, y, num_classes=None):
        return np.array(list(map(lambda x: to_categorical(x, num_classes=num_classes), y)))

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)


def test_activation_functions():
    print('>>> activation functions test starts!')
    l = [i*.1 for i in range(-6, 6)]
    t = torch.Tensor(l)
    print(f'data: {t}')

    hyper = Hyper(offset=.3)
    print(f'hyper(data): {hyper(t)}')

    hypo = Hypo(min=0, max=.3)
    print(f'hypo(data): {hypo(t)}')


def train(model, opt, device,
          train_dataloader, val_dataloader, val_set_ratio,
          epochs):
    mse_fn = torch.nn.MSELoss()
    ce_fn = torch.nn.CrossEntropyLoss()

    display_step = 100
    eval_step = int(1/val_set_ratio)

    losses = {'iter': [], 'loss': [], 'mse_loss': [], 'ce_loss': []}
    val_losses = {'iter': [], 'loss': [], 'mse_loss': [], 'ce_loss': []}

    for epoch in range(epochs):
        print(f':: {epoch}-th epoch >>>')

        val_iter = iter(val_dataloader)
        model.train()
        for i, data in enumerate(iter(train_dataloader)):
            opt.zero_grad()

            x, target = data
            x = x.to(device)
            target = target.to(device)

            y, (cl, _) = model(x)
            mse_loss = mse_fn(y, x)
            ce_loss = ce_fn(cl, target)

            loss = mse_loss + ce_loss

            loss.backward()
            opt.step()

            if (i+1) % display_step == 0 or (i+1) == len(train_dataloader):
                print(f'>\t {i+1}-th iter:\tloss={loss:.2f},\tmse_loss={mse_loss:.3f},\tce_loss={ce_loss:.2f}')
                losses['iter'].append(epoch*len(train_dataloader)+i)
                losses['loss'].append(loss)
                losses['mse_loss'].append(mse_loss)
                losses['ce_loss'].append(ce_loss)

            del x, y, target
            del loss, mse_loss, ce_loss

            if (i + 1) % eval_step == 0:
                model.eval()
                try:
                    val_data = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataloader)
                    val_data = next(val_iter)
                val_x, val_target = val_data
                val_x = val_x.to(device)
                val_target = val_target.to(device)
                with torch.no_grad():
                    val_y, (val_cl, _) = model(val_x)
                    val_mse_loss = mse_fn(val_y, val_x)
                    val_ce_loss = ce_fn(val_cl, val_target)
                    val_loss = val_mse_loss + val_ce_loss

                    if (i + 1) % display_step == 0 or (i + 1) == len(train_dataloader):
                        print(f'>\t {i+1}-th iter:\t\t\t\t\t\tval_loss={val_loss:.2f},\tval_mse_loss={val_mse_loss:.3f},\tval_ce_loss={val_ce_loss:.2f}')
                        val_losses['iter'].append(epoch * len(train_dataloader) + i)
                        val_losses['loss'].append(val_loss)
                        val_losses['mse_loss'].append(val_mse_loss)
                        val_losses['ce_loss'].append(val_ce_loss)
                    del val_loss, val_mse_loss, val_ce_loss
                del val_x, val_y, val_target
                model.train()

    return {'losses': losses,
            'val_losses': val_losses}


def test_MNIST():
    print('>>> MNIST test starts!')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8 if device is "cuda" else 4

    print('importing data ...')
    mndata = MNIST('../mnist')
    train_images, train_labels = mndata.load_testing()
    # print(mndata.display(train_images[10]))
    train_dataset = MNISTDataset(np.array(train_images), np.array(train_labels))
    print(f'len(train_dataset): {len(train_dataset)}')
    l = len(train_dataset)
    val_set_ratio = 0.1
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                               [l - int(l * val_set_ratio),
                                                                int(l * val_set_ratio)])
    print(f'=> len(train_dataset), len(val_dataset): {len(train_dataset)}, {len(val_dataset)}')

    train_dl = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
    print(f'len(train_dl): {len(train_dl)}')
    val_dl = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=True)
    print(f'len(val_dl): {len(val_dl)}')

    test_images, test_labels = mndata.load_testing()
    # print(mndata.display(test_images[10]))
    test_dataset = MNISTDataset(np.array(test_images), np.array(test_labels))
    print(f'len(test_dataset): {len(test_dataset)}')
    test_dl = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)
    print(f'len(test_dl): {len(test_dl)}')

    print('constructing a model ...')
    in_features = train_dataset[0][0].shape[0]
    model = ShallowNet(in_features=in_features, out_features=8,
                       layer_config={'encoder':[in_features, 256, 64, 8],
                                     'classifier':[8, 8, 10],
                                     'decoder':[8, 64, 256, in_features]})
    model = model.to(device)
    print(model)
    opt = torch.optim.Adam(model.parameters(), 1e-3)

    print('training the model ...')

    result = train(model=model, opt=opt, device=device,
                   train_dataloader=train_dl,
                   val_dataloader=val_dl, val_set_ratio=val_set_ratio,
                   epochs=1)

    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax2 = ax1.twinx()
    x = result['losses']['iter']
    ax1.semilogy(x, result['losses']['loss'], 'ro-')
    ax1.semilogy(x, result['losses']['mse_loss'], 'bo-')
    ax1.semilogy(x, result['losses']['ce_loss'], 'go-')

    x = result['val_losses']['iter']
    ax1.semilogy(x, result['val_losses']['loss'], 'rx-.')
    ax1.semilogy(x, result['val_losses']['mse_loss'], 'bx-.')
    ax1.semilogy(x, result['val_losses']['ce_loss'], 'gx-.')

    ax1.legend(('loss', 'mse_loss', 'ce_loss', 'val_loss', 'val_mse_loss', 'val_ce_loss'))
    ax1.set_xlabel('iter')
    plt.grid()

    tag = 'bn'
    plt.savefig(f'./losses_{tag}.png')


def main():
    test_activation_functions()
    test_MNIST()


if __name__ == "__main__":
    main()
