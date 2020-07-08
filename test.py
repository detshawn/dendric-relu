from model.activations import Hyper, Hypo
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from mnist import MNIST
from utils import *

from model.shallownet import ShallowNet
from argparse import ArgumentParser


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
          train_dataloader, val_dataloader, eval_step,
          epochs, display_step):
    logger = Logger()
    mse_fn = torch.nn.MSELoss()
    ce_fn = torch.nn.CrossEntropyLoss()
    kl_fn = KLLoss

    losses = {'iter': [], 'loss': [], 'mse_loss': [], 'ce_loss': [], 'kl_loss': [], 'accuracy': []}
    val_losses = {'iter': [], 'loss': [], 'mse_loss': [], 'ce_loss': [], 'kl_loss': [], 'accuracy': []}

    for epoch in range(epochs):
        print(f':: {epoch}-th epoch >>>')

        pred_cnt = {'total': 0, 'true': 0}
        val_pred_cnt = {'total': 0, 'true': 0}
        val_iter = iter(val_dataloader)
        model.train()
        for i, data in enumerate(iter(train_dataloader)):
            opt.zero_grad()

            x, target = data
            x = x.to(device)
            target = target.to(device)

            # forward
            y, (cl, z_sample, z_mean, z_log_var) = model(x)
            # predict classification
            pred = (torch.argmax(cl, dim=1) == target).detach().numpy().sum()

            pred_cnt['total'] = pred_cnt['total'] + x.size()[0]
            pred_cnt['true'] = pred_cnt['true'] + pred

            # losses
            mse_loss = mse_fn(y, x)
            ce_loss = ce_fn(cl, target)
            kl_loss = kl_fn(z_mean, z_log_var)

            loss = mse_loss + ce_loss + kl_loss
            meta = {
                'loss': {'total': loss.clone().detach().cpu(), 'mse': mse_loss.clone().detach().cpu(),
                          'ce': ce_loss.clone().detach().cpu(), 'kl': kl_loss.clone().detach().cpu()},
                'acc': {'acc': pred_cnt["true"]/pred_cnt["total"]}
            }

            # backward
            loss.backward()
            # update
            opt.step()

            if (i+1) % display_step == 0 or (i+1) == len(train_dataloader):
                accuracy = meta['acc']['acc']
                print(f'>\t {i+1}-th iter:\tloss={loss:.2f},\tmse_loss={mse_loss:.3f},\tce_loss={ce_loss:.2f}\tkl_loss={kl_loss:.5f}\taccuracy={accuracy:.2f}')
                logger.scalars_summary(f'{args.tag}/train', meta['loss'], epoch * len(train_dataloader) + i + 1)
                logger.scalars_summary(f'{args.tag}/train_acc', meta['acc'], epoch * len(train_dataloader) + i + 1)
                losses['iter'].append(epoch*len(train_dataloader)+i)
                losses['loss'].append(loss)
                losses['mse_loss'].append(mse_loss)
                losses['ce_loss'].append(ce_loss)
                losses['kl_loss'].append(kl_loss)
                losses['accuracy'].append(accuracy)
                pred_cnt = {'total': 0, 'true': 0}

            del x, y, target
            del loss, mse_loss, ce_loss, kl_loss

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
                    val_y, (val_cl, val_z_sample, val_z_mean, val_z_log_var) = model(val_x)
                    val_pred = (torch.argmax(val_cl, dim=1) == val_target).detach().numpy().sum()
                    val_pred_cnt['total'] = val_pred_cnt['total'] + val_x.size()[0]
                    val_pred_cnt['true'] = val_pred_cnt['true'] + val_pred
                    val_mse_loss = mse_fn(val_y, val_x)
                    val_ce_loss = ce_fn(val_cl, val_target)
                    val_kl_loss = kl_fn(val_z_mean, val_z_log_var)
                    val_loss = val_mse_loss + val_ce_loss + val_kl_loss
                    val_meta = {
                        'loss': {'val_total': val_loss.clone().detach().cpu().numpy(), 'val_mse': val_mse_loss.clone().detach().cpu().numpy(),
                                 'val_ce': val_ce_loss.clone().detach().cpu().numpy(), 'val_kl': val_kl_loss.clone().detach().cpu().numpy()},
                        'acc': {'val_acc': val_pred_cnt["true"] / val_pred_cnt["total"]}
                    }

                    if (i + 1) % display_step == 0 or (i + 1) == len(train_dataloader):
                        val_accuracy = val_meta['acc']['val_acc']
                        print(f'>\t {i+1}-th iter:\t\t\t\t\t\tval_loss={val_loss:.2f},\tval_mse_loss={val_mse_loss:.3f},\tval_ce_loss={val_ce_loss:.2f}\tval_kl_loss={val_kl_loss:.5f}\tval_accuracy={val_accuracy:.2f}')
                        logger.scalars_summary(f'{args.tag}/train', val_meta['loss'], epoch * len(train_dataloader) + i + 1)
                        logger.scalars_summary(f'{args.tag}/train_acc', val_meta['acc'], epoch * len(train_dataloader) + i + 1)
                        val_losses['iter'].append(epoch * len(train_dataloader) + i)
                        val_losses['loss'].append(val_loss)
                        val_losses['mse_loss'].append(val_mse_loss)
                        val_losses['ce_loss'].append(val_ce_loss)
                        val_losses['kl_loss'].append(val_kl_loss)
                        val_losses['accuracy'].append(val_accuracy)
                        val_pred_cnt = {'total': 0, 'true': 0}
                    del val_loss, val_mse_loss, val_ce_loss, val_kl_loss
                del val_x, val_y, val_target
                model.train()

    return {'losses': losses,
            'val_losses': val_losses}


def plot_result(result, tag=None):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))
    ax1, ax2 = axes[0], axes[1]

    x = result['losses']['iter']
    ax1.semilogy(x, result['losses']['loss'], 'ro-')
    ax1.semilogy(x, result['losses']['mse_loss'], 'bo-')
    ax1.semilogy(x, result['losses']['ce_loss'], 'go-')
    ax1.semilogy(x, result['losses']['kl_loss'], 'mo-')
    ax2.plot(x, [i*100 for i in result['losses']['accuracy']], 'ko-')

    x = result['val_losses']['iter']
    ax1.semilogy(x, result['val_losses']['loss'], 'rx-.')
    ax1.semilogy(x, result['val_losses']['mse_loss'], 'bx-.')
    ax1.semilogy(x, result['val_losses']['ce_loss'], 'gx-.')
    ax1.semilogy(x, result['val_losses']['kl_loss'], 'mx-.')
    ax2.plot(x, [i*100 for i in result['val_losses']['accuracy']], 'k-.')

    ax1.legend(('loss', 'mse_loss', 'ce_loss', 'kl_loss',
                'val_loss', 'val_mse_loss', 'val_ce_loss', 'val_kl_loss'))
    ax1.set_xlabel('iter.')
    ax1.set_ylabel('loss (in log scale)')
    ax1.set_title('Loss vs iterations')
    ax1.grid()

    ax2.legend(('accuracy', 'val_accuracy'))
    ax2.set_xlabel('iter.')
    ax2.set_ylabel('acc. (%)')
    ax2.set_title('Acc. vs iterations')
    ax2.grid()

    filename = './losses' + (f'_{tag}' if tag is not None else '') + '.png'
    plt.savefig(filename)


def test_MNIST():
    print('>>> MNIST test starts!')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size if device is "cuda" else 4

    print('importing data ...')
    mndata = MNIST('../mnist')
    train_images, train_labels = mndata.load_testing()
    # print(mndata.display(train_images[10]))
    train_dataset = MNISTDataset(np.array(train_images), np.array(train_labels))
    # print(f'len(train_dataset): {len(train_dataset)}')
    l = len(train_dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                               [l - int(l * args.val_set_ratio),
                                                                int(l * args.val_set_ratio)])
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
    # print(f'len(test_dataset): {len(test_dataset)}')
    test_dl = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)
    print(f'len(test_dl): {len(test_dl)}')

    print('constructing a model ...')
    in_features = train_dataset[0][0].shape[0]
    model = ShallowNet(in_features=in_features, out_features=8,
                       layer_config={'encoder':[in_features, 256, 64, 8],
                                     'classifier':[8, 8, 10],
                                     'decoder':[8, 64, 256, in_features]},
                       multi_position=args.multi_position)
    model = model.to(device)
    print(model)
    print(f'# parameters: {sum(p.numel() for p in model.parameters())}')
    opt = torch.optim.Adam(model.parameters(), 1e-3)

    print('training the model ...')

    result = train(model=model, opt=opt, device=device,
                   train_dataloader=train_dl,
                   val_dataloader=val_dl, eval_step=int(1/args.val_set_ratio),
                   epochs=args.epochs, display_step=args.display_step)

    plot_result(result, tag=args.tag)


def main():
    # test_activation_functions()
    test_MNIST()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-batch-size', default=8, type=int)
    parser.add_argument('-epochs', default=64, type=int)
    parser.add_argument('-tag', default='nonetag')
    parser.add_argument('-val-set-ratio', default=0.1, type=float)
    parser.add_argument('-display-step', default=300, type=int)
    parser.add_argument('-multi-position', default=1, type=int)
    args = parser.parse_args()
    main()
