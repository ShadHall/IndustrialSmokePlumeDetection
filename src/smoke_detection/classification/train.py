# Copyright (C) 2020 Michael Mommert
# This file is part of IndustrialSmokePlumeDetection
# <https://github.com/HSG-AIML/IndustrialSmokePlumeDetection>.
#
# This file contains a wrapper for training the 4-channel classifier.

import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import os
import platform
import numpy as np
import torch
from torch import nn, optim
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from smoke_detection import dataset_paths
from smoke_detection.classification.model import model, device
from smoke_detection.classification.data import create_dataset

_NUM_WORKERS = 0 if platform.system() == "Windows" else 4
_CLS_DIR = os.path.dirname(os.path.abspath(__file__))


def _save_checkpoint(path, state_dict):
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    try:
        torch.save(
            state_dict,
            tmp,
            _use_new_zipfile_serialization=False,
        )
        os.replace(tmp, path)
    except Exception:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def _save_plots_cls(history, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    epochs_range = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs_range, history['train_loss'], label='Train')
    axes[0].plot(epochs_range, history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(epochs_range, history['train_acc'], label='Train')
    axes[1].plot(epochs_range, history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    axes[2].plot(epochs_range, history['lr'])
    axes[2].set_title('Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log')

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'training_curves.png'), dpi=100)
    plt.close(fig)


def train_model(model, epochs, opt, loss, batch_size):
    data_train = create_dataset(
        datadir=dataset_paths.classification_split('train'),
        balance='upsample', mult=1)

    data_val = create_dataset(
        datadir=dataset_paths.classification_split('val'),
        balance='upsample', mult=1)

    train_sampler = RandomSampler(data_train, replacement=True,
                                  num_samples=int(2*len(data_train)/3))
    val_sampler = RandomSampler(data_val, replacement=True,
                                num_samples=int(2*len(data_val)/3))

    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        data_train,
        batch_size=batch_size,
        num_workers=_NUM_WORKERS,
        pin_memory=pin,
        sampler=train_sampler,
    )
    val_dl = DataLoader(
        data_val,
        batch_size=batch_size,
        num_workers=_NUM_WORKERS,
        pin_memory=pin,
        sampler=val_sampler,
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    plot_dir = os.path.join(_CLS_DIR, 'plots')

    for epoch in range(epochs):
        model.train()

        train_loss_total, train_acc_total = 0, 0
        progress = tqdm(enumerate(train_dl), desc="Train Loss: ",
                        total=len(train_dl))
        for i, batch in progress:
            x = batch['img'].float().to(device)
            y = batch['lbl'].float().to(device)

            output = model(x)
            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1

            acc = accuracy_score(y.cpu().detach().numpy(), output_binary)
            train_acc_total += acc

            loss_epoch = loss(output, y.reshape(-1, 1))
            train_loss_total += loss_epoch.item()
            progress.set_description("Train Loss: {:.4f}".format(
                train_loss_total/(i+1)))

            opt.zero_grad()
            loss_epoch.backward()
            opt.step()

        writer.add_scalar("training loss", train_loss_total/(i+1), epoch)
        writer.add_scalar("training acc", train_acc_total/(i+1), epoch)
        writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], epoch)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.eval()
        val_loss_total, val_acc_total = 0, 0
        progress = tqdm(enumerate(val_dl), desc="val Loss: ",
                        total=len(val_dl))
        for j, batch in progress:
            x = batch['img'].float().to(device)
            y = batch['lbl'].float().to(device)

            output = model(x)

            loss_epoch = loss(output, y.reshape(-1, 1))
            val_loss_total += loss_epoch.item()
            progress.set_description("val Loss: {:.4f}".format(
                val_loss_total/(j+1)))

            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1
            acc = accuracy_score(y.cpu().detach().numpy(), output_binary)
            val_acc_total += acc

        writer.add_scalar("val loss", val_loss_total/(j+1), epoch)
        writer.add_scalar("val accuracy", val_acc_total/(j+1), epoch)

        history['train_loss'].append(train_loss_total / (i + 1))
        history['train_acc'].append(train_acc_total / (i + 1))
        history['val_loss'].append(val_loss_total / (j + 1))
        history['val_acc'].append(val_acc_total / (j + 1))
        history['lr'].append(opt.param_groups[0]['lr'])
        _save_plots_cls(history, plot_dir)

        print(("Epoch {:d}: train loss={:.3f}, val loss={:.3f}, "
               "train acc={:.3f}, val acc={:.3f}").format(
                   epoch+1, train_loss_total/(i+1), val_loss_total/(j+1),
                   train_acc_total/(i+1), val_acc_total/(j+1)))

        fname = '4ch_ep{:0d}_lr{:.0e}_bs{:02d}_mo{:.1f}_{:03d}.model'.format(
            args.ep, args.lr, args.bs, args.mo, epoch)
        ckpt_path = os.path.join(_CLS_DIR, fname)
        _save_checkpoint(ckpt_path, model.state_dict())
        _save_checkpoint(os.path.join(_CLS_DIR, 'classification_4ch.model'),
                         model.state_dict())

        writer.flush()
        scheduler.step(val_loss_total / (j + 1))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


parser = argparse.ArgumentParser()
parser.add_argument('-ep', type=int, default=100, help='Number of epochs')
parser.add_argument('-bs', type=int, nargs='?', default=30, help='Batch size')
parser.add_argument('-lr', type=float, nargs='?', default=0.3,
                    help='Learning rate')
parser.add_argument('-mo', type=float, nargs='?', default=0.7,
                    help='Momentum')
args = parser.parse_args()

writer = SummaryWriter('runs/' + "4ch_ep{:0d}_lr{:.0e}_bs{:03d}_mo{:.1f}/".format(
    args.ep, args.lr, args.bs, args.mo))

loss = nn.BCEWithLogitsLoss()
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min',
                                                  factor=0.5, threshold=1e-4,
                                                  min_lr=1e-6)

train_model(model, args.ep, opt, loss, args.bs)
writer.close()
