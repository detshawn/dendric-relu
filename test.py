from model.activations import Hyper, Hypo
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from functools import reduce
import time

from mnist import MNIST
from utils import *

from parallel import DataParallelModel, DataParallelCriterion
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

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


def train(model, opt, device, logger, args,
          train_dataloader, train_dataset,
          epoch, prev_iter, display_step, is_logging,
          final_epoch=False,
          mse_fn=None, ce_fn=None, kl_fn=None, guess_bce_fn=None,
          guess=False, ge2e=False, conditional=False, ext_training=False,
          partial_training=None, partial_set_sampling=False,
          focal_kwargs=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress_list = [batch_time, data_time]
    loss_meter = AverageMeter('Loss', ':.4e')
    mse_loss_meter = AverageMeter('MSE Loss', ':.4e')
    ce_loss_meter = AverageMeter('CE Loss', ':.4e')
    kl_loss_meter = AverageMeter('KL Loss', ':.4e')
    losses = [loss_meter, mse_loss_meter, ce_loss_meter, kl_loss_meter]
    if ge2e:
        ge2e_loss_meter = AverageMeter('GE2E Loss', ':.4e')
        losses.append(ge2e_loss_meter)
    if guess:
        guess_bce_loss_meter = AverageMeter('Guess BCE Loss', ':.4e')
        losses.append(guess_bce_loss_meter)
    progress_list.extend(losses)
    losses = AverageMeterList(losses)
    if partial_training:
        partial_acc_meter = AverageMeter('Partial Acc', ':.4e')
        accs = [partial_acc_meter]
    else:
        acc_meter = AverageMeter('Acc', ':6.2f')
        accs = [acc_meter]
    if guess:
        guess_acc_meter = AverageMeter('Guess Acc', ':.4e')
        accs.append(guess_acc_meter)
    progress_list.extend(accs)
    accs = AverageMeterList(accs)

    progress = ProgressMeter(len(train_dataloader), progress_list, prefix="Epoch: [{}]".format(epoch))

    embeds_for_ge2e = [[] for _ in range(10)]

    pred_cnt = {'total': 0, 'true': 0,
                'guess_true_pos': 0, 'guess_true_neg': 0,
                'guess_false_pos': 0, 'guess_false_neg': 0}
    distr = {'z_means': [], 'y_labels': [], 'i_successes': []}

    hist_pairs = np.zeros((10, 10))

    model.train()

    end = time.time()
    for i, data in enumerate(iter(train_dataloader)):
        # measure the data loading time
        data_time.update(time.time() - end)

        opt.zero_grad()

        x_idx, x, target = data
        x = x.to(device)
        target = target.to(device)

        # forward
        enc_kwargs = dict(guess=guess, conditional=conditional, ext_training=ext_training)
        y, (cl, z_sample, enc_intermediates) = model(x, enc_kwargs=enc_kwargs)

        z_mean = enc_intermediates['z_mean']
        z_log_var = enc_intermediates['z_log_var']
        # predict classification
        pred = (torch.argmax(cl, dim=1) == target).detach()

        # losses
        mse_loss = mse_fn(y, x)
        ce_loss = ce_fn(cl, target, **focal_kwargs)
        kl_loss = kl_fn(z_mean, z_log_var)
        loss = mse_loss + ce_loss + kl_loss

        mse_loss_meter.update(mse_loss.item())
        ce_loss_meter.update(ce_loss.item())
        kl_loss_meter.update(kl_loss.item())

        if ge2e:
            for emb, t in zip(z_mean, target):
                embeds_for_ge2e[t].append(emb)
            if (i+1) % args.ge2e_step == 0:
                n_min_samples = 12
                embeds_for_ge2e_input = [torch.stack(e, dim=0).clone() for e in embeds_for_ge2e if len(e) >= n_min_samples]
                if len(embeds_for_ge2e_input) > 2:
                    rect_n = reduce(lambda x, y: min(x, y.size()[0]), embeds_for_ge2e_input, n_min_samples)
                    embeds_for_ge2e_input = [e[:rect_n, :] for e in embeds_for_ge2e_input]
                    embeds_for_ge2e_input = torch.stack(embeds_for_ge2e_input, dim=0)
                    ge2e_loss, eer = model.ge2e_loss(embeds_for_ge2e_input)

                    ge2e_weight = .1
                    loss = loss + ge2e_weight * ge2e_loss
                    ge2e_loss_meter.update(ge2e_weight * ge2e_loss.item())

                    embeds_for_ge2e = [[] for i in range(10)]

        if guess:
            guess_out = enc_intermediates['guess_out']
            guess_bce_loss = guess_bce_fn(guess_out.double(), pred.double())
            # guess_pred = ((guess_out >= .5) == pred.view(-1, 1)).detach()

            loss = loss + guess_bce_loss
            guess_bce_loss_meter.update(guess_bce_loss.item())

        loss_meter.update(loss.item())

        # backward
        loss.backward()
        # update
        opt.step()

        # eval
        enc_kwargs_eval = dict(guess=guess, conditional=conditional)
        model.eval()
        with torch.no_grad():
            _, (cl_eval, _, enc_intermediates_eval) = model(x, dropout=False, enc_kwargs=enc_kwargs_eval)
            z_mean_eval = enc_intermediates_eval['z_mean']

            if is_logging or final_epoch:
                for r, c in zip(torch.argmax(cl_eval, dim=1), target):
                    if r != c:
                        hist_pairs[r, c] = hist_pairs[r, c] + 1

            pred_eval = (torch.argmax(cl_eval, dim=1) == target).detach()
            if guess:
                guess_out_eval = enc_intermediates_eval['guess_out']
                guess_pred_eval = ((guess_out_eval >= .5) == pred_eval.view(-1, 1)).detach()

            pred_eval = pred_eval.cpu().numpy()
            pred_eval_sum = pred_eval.sum()

            # predict classification
            if guess:
                guess_pred_eval = guess_pred_eval.cpu().numpy()
                pred_cnt['guess_true_pos'] = pred_cnt['guess_true_pos'] + guess_pred_eval[pred_eval == 1].sum()
                pred_cnt['guess_true_neg'] = pred_cnt['guess_true_neg'] + (guess_pred_eval[pred_eval == 0] == 0).sum()
                pred_cnt['guess_false_pos'] = pred_cnt['guess_false_pos'] + (guess_pred_eval[pred_eval == 0] == 1).sum()
                pred_cnt['guess_false_neg'] = pred_cnt['guess_false_neg'] + (guess_pred_eval[pred_eval == 1] == 0).sum()
                if partial_set_sampling:
                    passed = (guess_pred_eval == 0)
                    extended_indices.extend([idx.item() for i, idx in enumerate(x_idx) if passed[i]])

            distr['z_means'].append(z_mean_eval.clone().detach().cpu().numpy())
            distr['y_labels'].append(target.clone().detach().cpu().numpy())
            distr['i_successes'].append(pred_eval)

            pred_cnt['total'] = pred_cnt['total'] + x.size()[0]
            pred_cnt['true'] = pred_cnt['true'] + pred_eval_sum
        model.train()

        if (i+1) % display_step == 0 or (i+1) == len(train_dataloader):
            # acc calc
            if partial_training:
                partial_acc_meter.update(pred_cnt["true"], pred_cnt["total"])
            else:
                acc_meter.update(pred_cnt["true"], pred_cnt["total"])
            if guess:
                guess_acc_meter.update((pred_cnt["guess_true_pos"]+pred_cnt["guess_true_neg"]), pred_cnt["total"])

            progress.display(i)

            logger.scalars_summary(f'{args.tag}/train', losses.list_to_dict(), prev_iter + i + 1)
            logger.scalars_summary(f'{args.tag}/train_acc', accs.list_to_dict(), prev_iter + i + 1)
            pred_cnt = {'total': 0, 'true': 0,
                        'guess_true_pos': 0, 'guess_true_neg': 0,
                        'guess_false_pos': 0, 'guess_false_neg': 0}

            if is_logging or final_epoch:
                if (i + 1) == display_step:
                    distr['z_means'] = np.concatenate(distr['z_means'], axis=0)
                    distr['y_labels'] = np.concatenate(distr['y_labels'], axis=0)
                    distr['i_successes'] = np.concatenate(distr['i_successes'], axis=0)

                    fig = plot_results((distr['z_means'], distr['y_labels'], distr['i_successes']), tag=args.tag)
                    img = fig2rgb_array(fig, expand=False)
                    logger.image_summary(f'{args.tag}/train', img, prev_iter + i + 1, dataformats='HWC')
                    plt.close(fig)

                if (i + 1) == len(train_dataloader):
                    fig, ax = plt.subplots(1)
                    ax.xaxis.tick_top()
                    plt.gca().invert_yaxis()
                    p = ax.pcolormesh(hist_pairs, cmap=plt.cm.Reds,
                                      norm=colors.PowerNorm(gamma=0.5, vmax=int(len(train_dataset)*.1*.1)))
                    fig.colorbar(p, label='freq (log scale)')
                    ax.set_xlabel('target')
                    ax.set_ylabel('pred')
                    ax.set_title('Misclassification map')
                    fig.savefig('test_hist_pairs.png')
                    img = fig2rgb_array(fig, expand=False)
                    logger.image_summary(f'{args.tag}/train_hist_pairs', img, epoch, dataformats='HWC')
                    plt.close(fig)

            distr = {'z_means': [], 'y_labels': [], 'i_successes': []}

        del x, y, target
        del loss, mse_loss, ce_loss, kl_loss
        if guess:
            del guess_bce_loss

        # measure the batch time
        batch_time.update(time.time() - end)
        end = time.time()


def validate(model, device, logger, args,
             val_dataloader,
             epoch, prev_iter,
             mse_fn=None, ce_fn=None, kl_fn=None, guess_bce_fn=None,
             guess=False, ge2e=False, conditional=False, ext_training=False,
             partial_training=None, partial_set_sampling=False,
             focal_kwargs=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress_list = [batch_time, data_time]
    loss_meter = AverageMeter('Val Loss', ':.4e')
    mse_loss_meter = AverageMeter('Val MSE Loss', ':.4e')
    ce_loss_meter = AverageMeter('Val CE Loss', ':.4e')
    kl_loss_meter = AverageMeter('Val KL Loss', ':.4e')
    val_losses = [loss_meter, mse_loss_meter, ce_loss_meter, kl_loss_meter]
    if ge2e:
        ge2e_loss_meter = AverageMeter('Val GE2E Loss', ':.4e')
        val_losses.append(ge2e_loss_meter)
    if guess:
        guess_bce_loss_meter = AverageMeter('Val Guess BCE Loss', ':.4e')
        val_losses.append(guess_bce_loss_meter)
    progress_list.extend(val_losses)
    val_losses = AverageMeterList(val_losses)
    if partial_training:
        partial_acc_meter = AverageMeter('Val Partial Acc', ':.4e')
        val_accs = [partial_acc_meter]
    else:
        acc_meter = AverageMeter('Val Acc', ':6.2f')
        val_accs = [acc_meter]
    if guess:
        guess_acc_meter = AverageMeter('Val Guess Acc', ':.4e')
        val_accs.append(guess_acc_meter)
    progress_list.extend(val_accs)
    val_accs = AverageMeterList(val_accs)

    progress = ProgressMeter(len(val_dataloader), progress_list, prefix="Epoch: [{}]".format(epoch))

    val_pred_cnt = {'total': 0, 'true': 0, 'guess_true_pos': 0, 'guess_true_neg': 0}

    model.eval()

    end = time.time()
    for i, val_data in enumerate(iter(val_dataloader)):
        # measure the data loading time
        data_time.update(time.time() - end)

        val_x_idx, val_x, val_target = val_data
        val_x = val_x.to(device)
        val_target = val_target.to(device)

        with torch.no_grad():
            val_enc_kwargs = dict(guess=guess, conditional=conditional)
            val_y, (val_cl, val_z_sample, val_enc_intermediates) = model(val_x, enc_kwargs=val_enc_kwargs)
            val_z_mean = val_enc_intermediates['z_mean']
            val_z_log_var = val_enc_intermediates['z_log_var']
            val_pred = (torch.argmax(val_cl, dim=1) == val_target).detach()

            val_mse_loss = mse_fn(val_y, val_x)
            val_ce_loss = ce_fn(val_cl, val_target, **focal_kwargs)
            val_kl_loss = kl_fn(val_z_mean, val_z_log_var)
            val_loss = val_mse_loss + val_ce_loss + val_kl_loss

            mse_loss_meter.update(val_mse_loss.clone().detach().cpu())
            ce_loss_meter.update(val_ce_loss.clone().detach().cpu())
            kl_loss_meter.update(val_kl_loss.clone().detach().cpu())

            if guess:
                val_guess_out = val_enc_intermediates['guess_out']
                val_guess_pred = ((val_guess_out >= .5) == val_pred.view(-1, 1)).detach()

                val_guess_bce_loss = guess_bce_fn(val_guess_out.double(), val_pred.double())
                guess_bce_loss_meter.update(val_guess_bce_loss.clone().detach().cpu())

                val_loss = val_loss + val_guess_bce_loss

            loss_meter.update(val_loss.clone().detach().cpu())

            val_pred = val_pred.cpu().numpy()
            val_pred_sum = val_pred.sum()
            val_pred_cnt['total'] = val_pred_cnt['total'] + val_x.size()[0]
            val_pred_cnt['true'] = val_pred_cnt['true'] + val_pred_sum
            if guess:
                val_guess_pred = val_guess_pred.cpu().numpy()
                val_pred_cnt['guess_true_pos'] = val_pred_cnt['guess_true_pos'] + val_guess_pred[val_pred == 1].sum()
                val_pred_cnt['guess_true_neg'] = val_pred_cnt['guess_true_neg'] + (val_guess_pred[val_pred == 0] == 0).sum()

            del val_loss, val_mse_loss, val_ce_loss, val_kl_loss
        del val_x, val_y, val_target

        if (i + 1) == len(val_dataloader):
            # acc calc
            if partial_training:
                partial_acc_meter.update(val_pred_cnt["true"], val_pred_cnt["total"])
            else:
                acc_meter.update(val_pred_cnt["true"], val_pred_cnt["total"])
            if guess:
                acc_meter.update((val_pred_cnt["guess_true_pos"] + val_pred_cnt["guess_true_neg"]), val_pred_cnt["total"])

            progress.display(i)

            logger.scalars_summary(f'{args.tag}/train', val_losses.list_to_dict(), prev_iter + 1)
            logger.scalars_summary(f'{args.tag}/train_acc', val_accs.list_to_dict(), prev_iter + 1)
            val_pred_cnt = {'total': 0, 'true': 0, 'guess_true_pos': 0, 'guess_true_neg': 0}

        # measure the batch time
        batch_time.update(time.time() - end)
        end = time.time()


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    print('>>> MNIST test starts!')
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
        if args.gpu is not None:
            device = device + f":{args.gpu}"
        torch.cuda.set_device(args.gpu)

        print("Use GPU: {} for training".format(args.gpu))
        if args.rank == -1:
            args.rank = int(os.environ["RANK"])
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:6006',
                                world_size=args.world_size,
                                rank=args.rank)

    batch_size = args.batch_size if device is not "cpu" else 4

    print('importing data ...')
    mndata = MNIST(args.mnist_data_path)
    train_images, train_labels = mndata.load_testing()
    # print(mndata.display(train_images[10]))
    full_dataset = MNISTDataset(np.array(train_images), np.array(train_labels))
    l = len(full_dataset)
    split = int(np.floor((1-args.val_set_ratio) * l))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [split, l-split])
    train_sampler = DistributedSampler(train_dataset) if args.multiprocessing_distributed else None
    val_sampler = DistributedSampler(val_dataset) if args.multiprocessing_distributed else None

    train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler)
    print(f'len(train_dl): {len(train_dl)}')
    val_dl = DataLoader(dataset=val_dataset, batch_size=batch_size,
                        shuffle=(val_sampler is None), pin_memory=True, sampler=val_sampler)
    print(f'len(val_dl): {len(val_dl)}')

    test_images, test_labels = mndata.load_testing()
    # print(mndata.display(test_images[10]))
    test_dataset = MNISTDataset(np.array(test_images), np.array(test_labels))
    # print(f'len(test_dataset): {len(test_dataset)}')
    test_dl = DataLoader(dataset=test_dataset, batch_size=batch_size,
                         shuffle=True, pin_memory=True)
    print(f'len(test_dl): {len(test_dl)}')

    print('constructing a model ...')
    # load config file
    # with open(args.config) as f:
    #     hp_str = ''.join(f.readlines())
    with open(args.config) as f:
        config = yaml.load(f)
    # print(f'config: {config}')

    sub_kwargs = dict(dendric=args.dendric,
                      multi_position=args.multi_position)
    enc_kwargs = dict(is_extended_layers=args.extended_layers,
                      conditional_batch_norm=args.conditional_batch_norm)
    in_features = train_dataset[0][1].shape[0]
    model = ShallowNet(in_features=in_features, out_features=8, device=device,
                       config=config['ShallowNet'],
                       sub_kwargs=sub_kwargs, enc_kwargs=enc_kwargs)
    if args.load_model:
        if os.path.exists(args.load_model_path):
            ckpts = torch.load(args.load_model_path)
            init_epoch = ckpts['epoch']
            init_iter = ckpts['iter']
            model.load_state_dict(ckpts['model_state_dict'])
        else:
            print(f'error: the model path ({args.load_model_path}) is invalid!')
            exit(1)
    else:
        init_epoch, init_iter = 0, 0

    if args.multiprocessing_distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
    elif args.data_parallel:
        if not args.data_parallel_loss_parallel:
            model = torch.nn.DataParallel(model)
        else:
            model = DataParallelModel(model)

    model = model.to(device)
    print(model)
    print(f'# parameters: {sum(p.numel() for p in model.parameters())}')
    opt = torch.optim.Adam(model.parameters(), 1e-3)

    print('training the model ...')
    logger = Logger()

    def build_decorate_data_parallel_criterion(is_distributed, is_data_parallel):
        def _decorate_data_parallel_criterion(fn):
            if is_distributed:
                return fn.to(device)
            elif is_data_parallel:
                return DataParallelCriterion(fn)
            else:
                return fn
        return _decorate_data_parallel_criterion
    dec = build_decorate_data_parallel_criterion(args.multiprocessing_distributed,
                                                 args.data_parallel and args.data_parallel_loss_parallel)

    mse_fn = dec(torch.nn.MSELoss())
    ce_fn = dec(build_focal_loss(args.gamma) if args.focal_loss else torch.nn.CrossEntropyLoss())
    kl_fn = dec(KLLoss)
    guess_bce_fn = dec(torch.nn.BCELoss())
    criteria_kwargs = dict(mse_fn=mse_fn, ce_fn=ce_fn, kl_fn=kl_fn, guess_bce_fn=guess_bce_fn)

    gamma_scaling_fn = None
    if args.focal_loss:
        def build_gamma_scaling_fn(function):
            def scaling_sigmoid(x, offset, width_factor=20):
                return sigmoid((x-offset) * 4 / width_factor)

            def scaling_cos(x, offset, pre_offset=False, T=20):
                x_tilde = x - offset
                if not pre_offset:
                    x_tilde = max(x_tilde, 0)
                return (-math.cos(x_tilde / T*(2*math.pi)) + 1) / 2

            if function == 'sigmoid':
                return scaling_sigmoid
            elif function == 'cos':
                return scaling_cos

        gamma_scaling_fn = build_gamma_scaling_fn(args.gamma_scaling_function)

    log_step = 4
    extended_indices = []
    extended_clock = 1
    prev_iter = init_iter

    times_per_epoch = []

    orig_train_dataloader = train_dl

    for epoch in range(init_epoch, args.epochs):
        print(f':: {epoch}-th epoch >>>')
        start = time.time()

        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        guess = (epoch + 1) > args.guess_trigger_epoch and args.guess
        ge2e = (epoch + 1) > args.ge2e_trigger_epoch and args.ge2e_loss
        conditional = (epoch + 1) > args.conditional_trigger_epoch and args.conditional_batch_norm
        partial_training = None
        partial_set_sampling = False

        if guess:
            if args.partial_training:
                if extended_clock == 0:
                    print(f'extended_clock: {extended_clock}, extended_indices: {len(extended_indices)}')
                    if len(extended_indices) > 0:
                        print(f'extended_indices[{len(extended_indices)}]')
                        train_dataloader = DataLoader(dataset=torch.utils.data.Subset(train_dataset, extended_indices),
                                                      batch_size=args.batch_size,
                                                      sampler=train_sampler,
                                                      shuffle=(train_sampler is None),
                                                      drop_last=True,
                                                      pin_memory=True)
                        print(f'train_dataloader (for sampling): {len(train_dataloader)}')
                        extended_clock = args.extended_clock_timer
                        for p in model.parameters():
                            p.requires_grad = False
                        for p in model.encoder.bns.parameters():
                            p.requires_grad = True

                        partial_training = True

                    else:
                        print(f'train_dataloader <- orig_train_dataloader {len(train_dataloader)}')
                        train_dataloader = orig_train_dataloader
                        extended_clock = 5
                        for p in model.parameters():
                            p.requires_grad = True
                        for p in model.encoder.bns.parameters():
                            p.requires_grad = False

                        partial_training = False

                else:
                    extended_clock = max(extended_clock - 1, 0)

                    if extended_clock == 0:
                        train_dataloader = orig_train_dataloader

                        partial_set_sampling = not (args.fixed_partial_set and len(extended_indices) > 0)
                        if partial_set_sampling:
                            print(f'train_dataloader (for measurement): {len(train_dataloader)}')
                            extended_indices = []
                        else:
                            print(f'train_dataloader (for entire set): {len(train_dataloader)}')

                        for p in model.parameters():
                            p.requires_grad = True
                        for p in model.encoder.bns.parameters():
                            p.requires_grad = False

                        partial_training = False

                    else:
                        partial_training = True

        focal_kwargs = {}
        if args.focal_loss:
            scaling_factor = gamma_scaling_fn(epoch, 3 / 5 * args.epochs - 1)
            focal_kwargs = dict(gamma=(args.gamma * scaling_factor))
            print(f'> scaling_factor: {scaling_factor}')
        print(f'> focal_kwargs: {focal_kwargs}')

        ext_training = (extended_clock > 0) if args.extended_layers else False

        config_kwargs = dict(guess=guess, ge2e=ge2e, conditional=conditional, ext_training=ext_training,
                             partial_training=partial_training, partial_set_sampling=partial_set_sampling,
                             focal_kwargs=focal_kwargs)

        train(model=model, opt=opt, device=device, logger=logger, args=args,
              train_dataloader=train_dl, train_dataset=train_dataset,
              epoch=epoch, final_epoch=((epoch + 1) == args.epochs),
              prev_iter=prev_iter, display_step=args.display_step, is_logging=((epoch + 1) % log_step == 0),
              **criteria_kwargs, **config_kwargs)

        if (epoch + 1) % log_step == 0 and log_step < 32:
            log_step = log_step * 2

        if (epoch + 1) % args.save_step == 0 or (epoch + 1) == args.epochs:
            if not os.path.exists(args.ckpt_model_dir):
                os.mkdir(args.ckpt_model_dir)
            ckpt_model_filename = f'shallownet_{args.tag}_{epoch}epoch.ckpt'
            torch.save({
                'epoch': epoch,
                'iter': prev_iter + len(train_dl),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()
            }, os.path.join(args.ckpt_model_dir, ckpt_model_filename))
            print("> ", str(epoch), "th checkpoint is saved!")

        prev_iter = prev_iter + len(train_dl)

        validate(model=model, device=device, logger=logger, args=args,
                 val_dataloader=val_dl,
                 epoch=epoch, prev_iter=prev_iter, **criteria_kwargs, **config_kwargs)

        times_per_epoch.append(time.time() - start)
        print(f'   {epoch}-th epoch <<< processing time: '
              f'{int((times_per_epoch[-1])/3600)}:{int((times_per_epoch[-1])%3600/60)}:{(times_per_epoch[-1])%60:.3f}')

def main():
    if args.world_size == -1:
        # Node is currently set to 1 since this script is written for a single-hardware case
        args.world_size = 1  # int(os.environ["WORLD_SIZE"])
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-config', required=True)
    parser.add_argument('-batch-size', default=8, type=int)
    parser.add_argument('-epochs', default=64, type=int)
    parser.add_argument('-extended-clock-timer', default=10, type=int)
    parser.add_argument('-tag', default='nonetag')
    parser.add_argument('--dendric', action='store_true')
    parser.add_argument('-val-set-ratio', default=0.1, type=float)
    parser.add_argument('-display-step', default=256, type=int)
    parser.add_argument('-save-step', default=50, type=int)
    parser.add_argument('-multi-position', default=1, type=int)

    parser.add_argument('--focal-loss', action='store_true')
    parser.add_argument('-gamma', default=2, type=float)
    parser.add_argument('-gamma-scaling-function', default='sigmoid')

    parser.add_argument('--ge2e-loss', action='store_true')
    parser.add_argument('-ge2e-step', default=16, type=float)
    parser.add_argument('-ge2e-trigger-epoch', type=int)

    parser.add_argument('--guess', action='store_true')
    parser.add_argument('--partial-training', action='store_true')
    parser.add_argument('--fixed-partial-set', action='store_true')
    parser.add_argument('--extended-layers', action='store_true')
    parser.add_argument('--conditional-batch-norm', action='store_true')
    parser.add_argument('-guess-trigger-epoch', type=int)
    parser.add_argument('-conditional-trigger-epoch', type=int)

    parser.add_argument('-ckpt-model-dir', default='./ckpts/')
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('-load-model-path')
    parser.add_argument('-mnist-data-path', default='../mnist')

    parser.add_argument('-gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('-rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--data-parallel', action='store_true')
    parser.add_argument('-data-parallel-loss-parallel', default=False, type=bool)
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')

    args = parser.parse_args()

    if args.guess:
        if args.guess_trigger_epoch is None:
            args.guess_trigger_epoch = int(args.epochs * 2/3)
    else:
        args.guess_trigger_epoch = args.epochs

    if args.conditional_batch_norm:
        if args.conditional_trigger_epoch is None:
            args.conditional_trigger_epoch = args.guess_trigger_epoch + 5
    else:
        args.conditional_trigger_epoch = args.epochs

    if args.ge2e_loss:
        if args.ge2e_trigger_epoch is None:
            args.ge2e_trigger_epoch = int(args.epochs * 2/3)
    else:
        args.ge2e_trigger_epoch = args.epochs

    args.eval_step = pow(2, math.floor(math.log2(int(1 / args.val_set_ratio))))

    print(f'args: {args}')
    main()
