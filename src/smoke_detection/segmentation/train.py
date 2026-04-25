# Copyright (C) 2020 Michael Mommert
# This file is part of IndustrialSmokePlumeDetection
# <https://github.com/HSG-AIML/IndustrialSmokePlumeDetection>.
#
# Wrapper for training the 4-channel segmentation model.

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
from sklearn.metrics import jaccard_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from smoke_detection.common import paths as dataset_paths
from smoke_detection.segmentation.model import model, device
from smoke_detection.segmentation.data import create_dataset

_NUM_WORKERS = 0 if platform.system() == "Windows" else 6
_SEG_DIR = os.path.dirname(os.path.abspath(__file__))

print('running on...', device)


def _save_checkpoint(path, state_dict):
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    try:
        torch.save(state_dict, tmp, _use_new_zipfile_serialization=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def _save_plots_seg(history, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    epochs_range = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    axes[0, 0].plot(epochs_range, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs_range, history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()

    axes[0, 1].plot(epochs_range, history['train_iou'], label='Train')
    axes[0, 1].plot(epochs_range, history['val_iou'], label='Val')
    axes[0, 1].set_title('IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()

    axes[0, 2].plot(epochs_range, history['train_acc'], label='Train')
    axes[0, 2].plot(epochs_range, history['val_acc'], label='Val')
    axes[0, 2].set_title('Accuracy')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()

    axes[1, 0].plot(epochs_range, history['train_arearatio_mean'], label='Train')
    axes[1, 0].plot(epochs_range, history['val_arearatio_mean'], label='Val')
    axes[1, 0].set_title('Area Ratio Mean')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()

    axes[1, 1].plot(epochs_range, history['train_arearatio_std'], label='Train')
    axes[1, 1].plot(epochs_range, history['val_arearatio_std'], label='Val')
    axes[1, 1].set_title('Area Ratio Std')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()

    axes[1, 2].plot(epochs_range, history['lr'])
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_yscale('log')

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'training_curves.png'), dpi=100)
    plt.close(fig)


def train_model(model, epochs, opt, loss, batch_size):
    tr_img, tr_lbl = dataset_paths.segmentation_split('train')
    va_img, va_lbl = dataset_paths.segmentation_split('val')
    data_train = create_dataset(datadir=tr_img, seglabeldir=tr_lbl, mult=1)
    data_val = create_dataset(datadir=va_img, seglabeldir=va_lbl, mult=1)

    train_sampler = RandomSampler(data_train, replacement=True,
                                  num_samples=int(2*len(data_train)/3))
    val_sampler = RandomSampler(data_val, replacement=True,
                                num_samples=int(2*len(data_val)/3))

    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        data_train, batch_size=batch_size, num_workers=_NUM_WORKERS,
        pin_memory=pin, sampler=train_sampler
    )
    val_dl = DataLoader(
        data_val, batch_size=batch_size, num_workers=_NUM_WORKERS,
        pin_memory=pin, sampler=val_sampler
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_acc': [], 'val_acc': [],
        'train_arearatio_mean': [], 'val_arearatio_mean': [],
        'train_arearatio_std': [], 'val_arearatio_std': [],
        'lr': [],
    }
    plot_dir = os.path.join(_SEG_DIR, 'plots')

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0
        train_ious = []
        train_acc_total = 0
        train_arearatios = []
        progress = tqdm(enumerate(train_dl), desc="Train Loss: ", total=len(train_dl))
        for i, batch in progress:
            x = batch['img'].float().to(device)
            y = batch['fpt'].float().to(device)

            output = model(x)
            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1

            for j in range(y.shape[0]):
                yt = y[j].flatten().cpu().detach().numpy()
                yp = output_binary[j][0].flatten()
                if np.sum(yp) != 0 and np.sum(yt) != 0:
                    train_ious.append(jaccard_score(yt, yp, zero_division=0))

            y_bin = np.array(np.sum(y.cpu().detach().numpy(), axis=(1, 2)) != 0).astype(int)
            pred_bin = np.array(np.sum(output_binary, axis=(1, 2, 3)) != 0).astype(int)
            train_acc_total += accuracy_score(y_bin, pred_bin)

            loss_epoch = loss(output, y.unsqueeze(dim=1))
            train_loss_total += loss_epoch.item()
            progress.set_description("Train Loss: {:.4f}".format(train_loss_total/(i+1)))

            area_pred = np.sum(output_binary, axis=(1, 2, 3))
            area_true = np.sum(y.cpu().detach().numpy(), axis=(1, 2))

            arearatios = []
            for k in range(len(area_pred)):
                if area_pred[k] == 0 and area_true[k] == 0:
                    arearatios.append(1)
                elif area_true[k] == 0:
                    arearatios.append(0)
                else:
                    arearatios.append(area_pred[k]/area_true[k])
            train_arearatios = np.ravel([*train_arearatios, *arearatios])

            opt.zero_grad()
            loss_epoch.backward()
            opt.step()

        writer.add_scalar("training loss", train_loss_total/(i+1), epoch)
        writer.add_scalar("training iou", float(np.average(train_ious)) if train_ious else 0.0, epoch)
        writer.add_scalar("training acc", train_acc_total/(i+1), epoch)
        writer.add_scalar('training arearatio mean', np.average(train_arearatios), epoch)
        writer.add_scalar('training arearatio std', np.std(train_arearatios), epoch)
        writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], epoch)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.eval()
        val_loss_total = 0
        val_ious = []
        val_acc_total = 0
        val_arearatios = []
        progress = tqdm(enumerate(val_dl), desc="val Loss: ", total=len(val_dl))
        for j, batch in progress:
            x = batch['img'].float().to(device)
            y = batch['fpt'].float().to(device)
            output = model(x)

            loss_epoch = loss(output, y.unsqueeze(dim=1))
            val_loss_total += loss_epoch.item()

            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1

            for k in range(y.shape[0]):
                yt = y[k].flatten().cpu().detach().numpy()
                yp = output_binary[k][0].flatten()
                if np.sum(yp) != 0 and np.sum(yt) != 0:
                    val_ious.append(jaccard_score(yt, yp, zero_division=0))

            y_bin = np.array(np.sum(y.cpu().detach().numpy(), axis=(1, 2)) != 0).astype(int)
            pred_bin = np.array(np.sum(output_binary, axis=(1, 2, 3)) != 0).astype(int)
            val_acc_total += accuracy_score(y_bin, pred_bin)

            area_pred = np.sum(output_binary, axis=(1, 2, 3))
            area_true = np.sum(y.cpu().detach().numpy(), axis=(1, 2))
            arearatios = []
            for k in range(len(area_pred)):
                if area_pred[k] == 0 and area_true[k] == 0:
                    arearatios.append(1)
                elif area_true[k] == 0:
                    arearatios.append(0)
                else:
                    arearatios.append(area_pred[k]/area_true[k])
            val_arearatios = np.ravel([*val_arearatios, *arearatios])

            progress.set_description("val Loss: {:.4f}".format(val_loss_total/(j+1)))

        writer.add_scalar("val loss", val_loss_total/(j+1), epoch)
        writer.add_scalar("val iou", float(np.average(val_ious)) if val_ious else 0.0, epoch)
        writer.add_scalar("val acc", val_acc_total/(j+1), epoch)
        writer.add_scalar('val arearatio mean', np.average(val_arearatios), epoch)
        writer.add_scalar('val arearatio std', np.std(val_arearatios), epoch)

        history['train_loss'].append(train_loss_total / (i + 1))
        history['val_loss'].append(val_loss_total / (j + 1))
        history['train_iou'].append(float(np.average(train_ious)) if train_ious else 0.0)
        history['val_iou'].append(float(np.average(val_ious)) if val_ious else 0.0)
        history['train_acc'].append(train_acc_total / (i + 1))
        history['val_acc'].append(val_acc_total / (j + 1))
        history['train_arearatio_mean'].append(np.average(train_arearatios))
        history['val_arearatio_mean'].append(np.average(val_arearatios))
        history['train_arearatio_std'].append(np.std(train_arearatios))
        history['val_arearatio_std'].append(np.std(val_arearatios))
        history['lr'].append(opt.param_groups[0]['lr'])
        _save_plots_seg(history, plot_dir)

        tr_iou = float(np.average(train_ious)) if train_ious else 0.0
        va_iou = float(np.average(val_ious)) if val_ious else 0.0
        print(("Epoch {:d}: train loss={:.3f}, val loss={:.3f}, "
               "train iou={:.3f}, val iou={:.3f}, "
               "train acc={:.3f}, val acc={:.3f}").format(
            epoch + 1, train_loss_total/(i+1), val_loss_total/(j+1),
            tr_iou, va_iou, train_acc_total/(i+1), val_acc_total/(j+1)))

        fname = '4ch_ep{:0d}_lr{:.0e}_bs{:02d}_mo{:.1f}_{:03d}.model'.format(
            args.ep, args.lr, args.bs, args.mo, epoch)
        ckpt_path = os.path.join(_SEG_DIR, fname)
        _save_checkpoint(ckpt_path, model.state_dict())
        _save_checkpoint(os.path.join(_SEG_DIR, 'segmentation_4ch.model'),
                         model.state_dict())

        writer.flush()
        scheduler.step(val_loss_total/(j+1))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


parser = argparse.ArgumentParser()
parser.add_argument('-ep', type=int, default=300, help='Number of epochs')
parser.add_argument('-bs', type=int, nargs='?', default=60, help='Batch size')
parser.add_argument('-lr', type=float, nargs='?', default=0.7, help='Learning rate')
parser.add_argument('-mo', type=float, nargs='?', default=0.7, help='Momentum')
args = parser.parse_args()

writer = SummaryWriter('runs/' + "4ch_ep{:0d}_lr{:.0e}_bs{:03d}_mo{:.1f}/".format(
    args.ep, args.lr, args.bs, args.mo))

loss = nn.BCEWithLogitsLoss()
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    opt, 'min', factor=0.5, threshold=1e-4, min_lr=1e-6
)

train_model(model, args.ep, opt, loss, args.bs)
writer.close()
