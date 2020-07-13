from model.activations import Hyper, Hypo
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt

from mnist import MNIST
from utils import *

from model.shallownet import ShallowNet
from argparse import ArgumentParser
import yaml


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
        return idx, self.images[idx], self.labels[idx]

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
          train_dataloader, train_dataset,
          val_dataloader, eval_step,
          epochs, display_step):
    logger = Logger()
    mse_fn = torch.nn.MSELoss()
    ce_fn = torch.nn.CrossEntropyLoss()
    kl_fn = KLLoss
    guess_bce_fn = torch.nn.BCELoss()

    losses = {'iter': [],
              'loss': [], 'mse_loss': [], 'ce_loss': [], 'kl_loss': [],
              'guess_bce_loss': [],
              'accuracy': []}
    val_losses = {'iter': [],
                  'loss': [], 'mse_loss': [], 'ce_loss': [], 'kl_loss': [],
                  'guess_bce_loss': [],
                  'accuracy': []}

    log_step = 4
    extended_indices = []
    extended_clock = 1
    prev_iter = 0

    orig_train_dataloader = train_dataloader
    for epoch in range(epochs):
        print(f':: {epoch}-th epoch >>>')

        pred_cnt = {'total': 0, 'true': 0,
                    'guess_true_pos': 0, 'guess_true_neg': 0,
                    'guess_false_pos': 0, 'guess_false_neg': 0}
        val_pred_cnt = {'total': 0, 'true': 0, 'guess_true_pos': 0, 'guess_true_neg': 0}
        val_iter = iter(val_dataloader)
        distr = {'z_means': [], 'y_labels': [], 'i_successes': []}

        guess = (epoch+1) > args.guess_trigger_epoch
        if guess:
            if extended_clock == 0:
                print(f'extended_clock: {extended_clock}, extended_indices: {len(extended_indices)}')
                if len(extended_indices) > 0:
                    print(f'extended_indices[{len(extended_indices)}]: {extended_indices}')
                    train_dataloader = DataLoader(dataset=train_dataset,
                                                  batch_size=args.batch_size,
                                                  sampler=SubsetRandomSampler(extended_indices))
                    print(f'train_dataloader (for sampling): {len(train_dataloader)}')
                    extended_clock = args.extended_clock_timer
                else:
                    print(f'train_dataloader <- orig_train_dataloader {len(train_dataloader)}')
                    train_dataloader = orig_train_dataloader
                    extended_clock = 5

            else:
                extended_clock = max(extended_clock - 1, 0)

                if extended_clock == 0:
                    train_dataloader = orig_train_dataloader
                    print(f'train_dataloader (for measurement): {len(train_dataloader)}')
                    extended_indices = []

        model.train()
        for i, data in enumerate(iter(train_dataloader)):
            opt.zero_grad()
            meta = {'loss': {}, 'acc': {}}

            x_idx, x, target = data
            x = x.to(device)
            target = target.to(device)

            # forward
            y, (cl, z_sample, z_mean, z_log_var, guess_out) = model(x, guess=guess, ext_training=(extended_clock > 0))
            # predict classification
            _, (cl_eval, _, _, _, guess_out_eval) = model(x, dropout=False, guess=guess, ext_training=False)
            pred = (torch.argmax(cl_eval, dim=1) == target).detach()
            guess_pred = ((guess_out_eval>=.5) == pred).detach()

            # losses
            mse_loss = mse_fn(y, x)
            ce_loss = ce_fn(cl, target)
            kl_loss = kl_fn(z_mean, z_log_var)
            loss = mse_loss + ce_loss + kl_loss

            meta['loss']['mse'] = mse_loss.clone().detach().cpu()
            meta['loss']['ce'] = ce_loss.clone().detach().cpu()
            meta['loss']['kl'] = kl_loss.clone().detach().cpu()
            meta['loss']['total'] = loss.clone().detach().cpu()

            if guess:
                guess_bce_loss = guess_bce_fn(guess_out.double(), pred.double())
                guess_pred = ((guess_out >= .5) == pred.view(-1, 1)).detach()
                meta['loss']['guess_bce'] = guess_bce_loss.clone().detach().cpu()

                loss = loss + guess_bce_loss
                meta['loss']['total'] = meta['loss']['total'] + meta['loss']['guess_bce']

            # predict classification
            pred = pred.cpu().numpy()
            pred_sum = pred.sum()

            pred_cnt['total'] = pred_cnt['total'] + x.size()[0]
            pred_cnt['true'] = pred_cnt['true'] + pred_sum
            if guess:
                guess_pred = guess_pred.cpu().numpy()
                pred_cnt['guess_true_pos'] = pred_cnt['guess_true_pos'] + guess_pred[pred == 1].sum()
                pred_cnt['guess_true_neg'] = pred_cnt['guess_true_neg'] + (guess_pred[pred == 0] == 0).sum()
                pred_cnt['guess_false_pos'] = pred_cnt['guess_false_pos'] + (guess_pred[pred == 0] == 1).sum()
                pred_cnt['guess_false_neg'] = pred_cnt['guess_false_neg'] + (guess_pred[pred == 1] == 0).sum()
                if extended_clock == 0:
                    passed = (guess_pred == 0) * (pred.reshape(-1, 1) == 0)
                    extended_indices.extend([idx.item() for i, idx in enumerate(x_idx) if passed[i]])

            distr['z_means'].append(z_mean.clone().detach().cpu().numpy())
            distr['y_labels'].append(target.clone().detach().cpu().numpy())
            distr['i_successes'].append(pred)

            # backward
            loss.backward()
            # update
            opt.step()

            if (i+1) % display_step == 0 or (i+1) == len(train_dataloader):
                meta['acc']['acc'] = pred_cnt["true"] / pred_cnt["total"]
                accuracy = meta['acc']['acc']
                loss_print = f'>\t {i+1}-th iter:\tloss={loss:.2f},\tmse_loss={mse_loss:.3f},\tce_loss={ce_loss:.2f}\tkl_loss={kl_loss:.5f}'
                acc_print = f' \t               \taccuracy={accuracy:.2f}'
                if guess:
                    loss_print += f'\tguess_bce_loss={guess_bce_loss:.5f}'
                    meta['acc']['guess_acc'] = (pred_cnt["guess_true_pos"]+pred_cnt["guess_true_neg"]) / pred_cnt["total"]
                    guess_acc = meta['acc']['guess_acc']
                    acc_print += f' \t               \tguess_acc={guess_acc:.2f}'
                print(loss_print)
                print(acc_print)

                logger.scalars_summary(f'{args.tag}/train', meta['loss'], prev_iter + i + 1)
                logger.scalars_summary(f'{args.tag}/train_acc', meta['acc'], prev_iter + i + 1)
                losses['iter'].append(prev_iter+i)
                losses['loss'].append(loss)
                losses['mse_loss'].append(mse_loss)
                losses['ce_loss'].append(ce_loss)
                losses['kl_loss'].append(kl_loss)
                losses['accuracy'].append(accuracy)
                pred_cnt = {'total': 0, 'true': 0,
                            'guess_true_pos': 0, 'guess_true_neg': 0,
                            'guess_false_pos': 0, 'guess_false_neg': 0}

                if (epoch+1) % log_step == 0 and (i+1) == display_step:
                    distr['z_means'] = np.concatenate(distr['z_means'], axis=0)
                    distr['y_labels'] = np.concatenate(distr['y_labels'], axis=0)
                    distr['i_successes'] = np.concatenate(distr['i_successes'], axis=0)

                    fig = plot_results((distr['z_means'], distr['y_labels'], distr['i_successes']), tag=args.tag)
                    img = fig2rgb_array(fig, expand=False)
                    logger.image_summary(f'{args.tag}/train', img, prev_iter + i + 1, dataformats='HWC')

                    log_step = log_step * 2
                distr = {'z_means': [], 'y_labels': [], 'i_successes': []}

            del x, y, target
            del loss, mse_loss, ce_loss, kl_loss

            if (i + 1) % eval_step == 0 or (i+1) == len(train_dataloader):
                model.eval()
                val_meta = {'loss': {}, 'acc': {}}

                try:
                    val_data = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataloader)
                    val_data = next(val_iter)
                val_x_idx, val_x, val_target = val_data
                val_x = val_x.to(device)
                val_target = val_target.to(device)
                with torch.no_grad():
                    val_y, (val_cl, val_z_sample, val_z_mean, val_z_log_var, val_guess_out) = model(val_x, guess=guess, ext_training=False)
                    val_pred = (torch.argmax(val_cl, dim=1) == val_target).detach()
                    val_guess_pred = ((val_guess_out >= .5) == val_pred.view(-1, 1)).detach()

                    val_mse_loss = mse_fn(val_y, val_x)
                    val_ce_loss = ce_fn(val_cl, val_target)
                    val_kl_loss = kl_fn(val_z_mean, val_z_log_var)
                    val_loss = val_mse_loss + val_ce_loss + val_kl_loss

                    val_meta['loss']['val_mse'] = val_mse_loss.clone().detach().cpu()
                    val_meta['loss']['val_ce'] = val_ce_loss.clone().detach().cpu()
                    val_meta['loss']['val_kl'] = val_kl_loss.clone().detach().cpu()
                    val_meta['loss']['val_total'] = val_loss.clone().detach().cpu()

                    if guess:
                        val_guess_bce_loss = guess_bce_fn(val_guess_out.double(), val_pred.double())
                        val_meta['loss']['val_guess_bce'] = val_guess_bce_loss.clone().detach().cpu()

                        val_loss = val_loss + val_guess_bce_loss
                        val_meta['loss']['val_total'] = val_meta['loss']['val_total'] + val_meta['loss']['val_guess_bce']

                    val_pred = val_pred.cpu().numpy()
                    val_pred_sum = val_pred.sum()
                    val_pred_cnt['total'] = val_pred_cnt['total'] + val_x.size()[0]
                    val_pred_cnt['true'] = val_pred_cnt['true'] + val_pred_sum
                    if guess:
                        val_guess_pred = val_guess_pred.cpu().numpy()
                        val_pred_cnt['guess_true_pos'] = val_pred_cnt['guess_true_pos'] + val_guess_pred[val_pred == 1].sum()
                        val_pred_cnt['guess_true_neg'] = val_pred_cnt['guess_true_neg'] + (val_guess_pred[val_pred == 0] == 0).sum()

                    if (i + 1) % display_step == 0 or (i + 1) == len(train_dataloader):
                        val_meta['acc']['val_acc'] = val_pred_cnt["true"] / val_pred_cnt["total"]
                        val_accuracy = val_meta['acc']['val_acc']
                        val_loss_print = f'>\t {i+1}-th iter:\t\t\t\t\t\tval_loss={val_loss:.2f},\tval_mse_loss={val_mse_loss:.3f},\tval_ce_loss={val_ce_loss:.2f}\tval_kl_loss={val_kl_loss:.5f}'
                        val_acc_print = f' \t               \t\t\t\t\t\tval_accuracy={val_accuracy:.2f}'
                        if guess:
                            val_loss_print += f'\tval_guess_bce_loss={val_guess_bce_loss:.5f}'
                            val_meta['acc']['val_guess_acc'] = (val_pred_cnt["guess_true_pos"] + val_pred_cnt["guess_true_neg"]) / val_pred_cnt["total"]
                            val_guess_acc = val_meta['acc']['val_guess_acc']
                            val_acc_print += f' \t               \t\t\t\t\t\tval_guess_acc={val_guess_acc:.2f}'
                        print(val_loss_print)
                        print(val_acc_print)

                        logger.scalars_summary(f'{args.tag}/train', val_meta['loss'], prev_iter + i + 1)
                        logger.scalars_summary(f'{args.tag}/train_acc', val_meta['acc'], prev_iter + i + 1)
                        val_losses['iter'].append(prev_iter + i)
                        val_losses['loss'].append(val_loss)
                        val_losses['mse_loss'].append(val_mse_loss)
                        val_losses['ce_loss'].append(val_ce_loss)
                        val_losses['kl_loss'].append(val_kl_loss)
                        val_losses['accuracy'].append(val_accuracy)
                        val_pred_cnt = {'total': 0, 'true': 0, 'guess_true_pos': 0, 'guess_true_neg': 0}
                    del val_loss, val_mse_loss, val_ce_loss, val_kl_loss
                del val_x, val_y, val_target
                model.train()
        prev_iter = prev_iter + len(train_dataloader)

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
    random_seed = 42
    l = len(train_dataset)
    indices = list(range(l))
    split = int(np.floor(args.val_set_ratio * l))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

    train_dl = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          sampler=train_sampler)
    print(f'len(train_dl): {len(train_dl)}')
    val_dl = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        sampler=val_sampler)
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
    # load config file
    # with open(args.config) as f:
    #     hp_str = ''.join(f.readlines())
    with open(args.config) as f:
        config = yaml.load(f)
    # print(f'config: {config}')

    in_features = train_dataset[0][1].shape[0]
    model = ShallowNet(in_features=in_features, out_features=8,
                       config=config['ShallowNet'],
                       dendric=args.dendric,
                       multi_position=args.multi_position)
    model = model.to(device)
    print(model)
    print(f'# parameters: {sum(p.numel() for p in model.parameters())}')
    opt = torch.optim.Adam(model.parameters(), 1e-3)

    print('training the model ...')

    result = train(model=model, opt=opt, device=device,
                   train_dataloader=train_dl, train_dataset=train_dataset,
                   val_dataloader=val_dl,
                   eval_step=int(1/args.val_set_ratio),
                   epochs=args.epochs, display_step=args.display_step)

    # plot_result(result, tag=args.tag)


def main():
    # test_activation_functions()
    test_MNIST()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-config', required=True)
    parser.add_argument('-batch-size', default=8, type=int)
    parser.add_argument('-epochs', default=64, type=int)
    parser.add_argument('-guess-trigger-epoch', default=10, type=int)
    parser.add_argument('-extended-clock-timer', default=10, type=int)
    parser.add_argument('-tag', default='nonetag')
    parser.add_argument('--dendric', action='store_true')
    parser.add_argument('-val-set-ratio', default=0.1, type=float)
    parser.add_argument('-display-step', default=300, type=int)
    parser.add_argument('-multi-position', default=1, type=int)
    args = parser.parse_args()
    main()
