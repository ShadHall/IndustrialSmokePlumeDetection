# Copyright (C) 2020 Michael Mommert
# This file is part of IndustrialSmokePlumeDetection
# <https://github.com/HSG-AIML/IndustrialSmokePlumeDetection>.
#
# IndustrialSmokePlumeDetection is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# IndustrialSmokePlumeDetection is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with IndustrialSmokePlumeDetection.  If not,
# see <http://www.gnu.org/licenses/>.
#
# If you use this code for your own project, please cite the following
# conference contribution:
#   Mommert, M., Sigel, M., Neuhausler, M., Scheibenreif, L., Borth, D.,
#   "Characterization of Industrial Smoke Plumes from Remote Sensing Data",
#   Tackling Climate Change with Machine Learning workshop at NeurIPS 2020.
#
#
# This file contains routines for the evaluation of the trained model
# based on the test data set

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from sklearn.metrics import jaccard_score

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import dataset_paths

from model_unet import *
from data import create_dataset

np.random.seed(3)
torch.manual_seed(3)

_SEG_DIR = os.path.dirname(os.path.abspath(__file__))

# setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default=os.path.join(_SEG_DIR, "segmentation.model"),
    help="Path to model checkpoint (.model state_dict)",
)
args = parser.parse_args()

# load data
te_img, te_lbl = dataset_paths.segmentation_split('test')
valdata = create_dataset(datadir=te_img, seglabeldir=te_lbl)

batch_size = 1 # 1 to create diagnostic images, any value otherwise
all_dl = DataLoader(valdata, batch_size=batch_size, shuffle=True)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# load model
model_path = os.path.abspath(os.path.expanduser(args.model))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# define loss function
loss_fn = nn.BCEWithLogitsLoss()

# run through test data
all_ious = []
all_accs = []
all_arearatios = []
all_y_bin = []      # image-level true labels across all batches
all_pred_bin = []   # image-level predicted labels across all batches
for i, batch in progress:
    x, y = batch['img'].float().to(device), batch['fpt'].float().to(device)
    idx = batch['idx']

    output = model(x)

    # obtain binary prediction map
    pred = np.zeros(output.shape)
    pred[output >= 0] = 1

    # derive Iou score
    cropped_iou = []
    for j in range(y.shape[0]):
        yt = y[j].flatten().detach().numpy()
        yp = pred[j][0].flatten()
        if np.sum(yp) != 0 and np.sum(yt) != 0:
            cropped_iou.append(
                jaccard_score(yt, yp, zero_division=0))
    all_ious = [*all_ious, *cropped_iou]
    
    # derive scalar binary labels on a per-image basis
    y_bin = np.array(np.sum(y.detach().numpy(),
                            axis=(1,2)) != 0).astype(int)
    prediction = np.array(np.sum(pred,
                               axis=(1,2,3)) != 0).astype(int)

    # accumulate image-level labels for aggregate confusion matrix
    all_y_bin.extend(y_bin.tolist())
    all_pred_bin.extend(prediction.tolist())

    # derive image-wise accuracy for this batch
    all_accs.append(accuracy_score(y_bin, prediction))

    # derive binary segmentation map from prediction
    output_binary = np.zeros(output.shape)
    output_binary[output.cpu().detach().numpy() >= 0] = 1

    # derive smoke areas
    area_pred = np.sum(output_binary, axis=(1,2,3))
    area_true = np.sum(y.cpu().detach().numpy(), axis=(1,2))

    # derive smoke area ratios
    arearatios = []
    for k in range(len(area_pred)):
        if area_pred[k] == 0 and area_true[k] == 0:
            arearatios.append(1)
        elif area_true[k] == 0:
            arearatios.append(0)
        else:
            arearatios.append(area_pred[k]/area_true[k])
    all_arearatios = np.ravel([*all_arearatios, *arearatios])


    if batch_size == 1:

        if prediction == 1 and y_bin == 1:
            res = 'true_pos'
        elif prediction == 0 and y_bin == 0:
            res = 'true_neg'
        elif prediction == 0 and y_bin == 1:
            res = 'false_neg'
        elif prediction == 1 and y_bin == 0:
            res = 'false_pos'

        # create plot
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(1, 3))

        # RGB plot
        ax1.imshow(0.2+1.5*(np.dstack([x[0][3], x[0][2], x[0][1]])-
                    np.min([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()]))/
                   (np.max([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()])-
                    np.min([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()])),
                   origin='upper')
        ax1.set_title({'true_pos': 'True Positive',
                       'true_neg': 'True Negative',
                       'false_pos': 'False Positive',
                       'false_neg': 'False Negative'}[res],
                      fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # false color plot
        ax2.imshow(0.2+(np.dstack([x[0][0], x[0][9], x[0][10]])-
                    np.min([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()]))/
                   (np.max([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()])-
                    np.min([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()])),
                   origin='upper')

        ax2.set_xticks([])
        ax2.set_yticks([])

        # segmentation ground-truth and prediction
        ax3.imshow(y[0], cmap='Reds', alpha=0.3)
        ax3.imshow(pred[0][0], cmap='Greens', alpha=0.3)
        ax3.set_xticks([])
        ax3.set_yticks([])

        this_iou = jaccard_score(
            y[0].flatten().detach().numpy(),
            pred[0][0].flatten(),
            zero_division=0,
        )
        ax3.annotate("IoU={:.2f}".format(this_iou), xy=(5,15), fontsize=8)

        f.subplots_adjust(0.05, 0.02, 0.95, 0.9, 0.05, 0.05)

        plt.savefig(res+(os.path.split(batch['imgfile'][0])[1]).\
                    replace('.tif', '_eval.png').replace(':', '_'),
                    dpi=200, bbox_inches='tight')
        plt.close()

print('iou:', len(all_ious), np.average(all_ious))
print('accuracy:', len(all_accs), np.average(all_accs))
print('mean area ratio:', len(all_arearatios), np.average(all_arearatios),
      np.std(all_arearatios)/np.sqrt(len(all_arearatios)-1))

# --- IoU distribution histogram ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(all_ious, bins=20, edgecolor='black')
ax.axvline(np.average(all_ious), color='red', linestyle='--',
           label='Mean = {:.3f}'.format(np.average(all_ious)))
ax.set_xlabel('IoU Score')
ax.set_ylabel('Count')
ax.set_title('IoU Distribution (n={})'.format(len(all_ious)))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(_SEG_DIR, 'iou_distribution_eval.png'), dpi=150)
plt.close()

# --- Area ratio distribution histogram ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(all_arearatios, bins=30, edgecolor='black')
ax.axvline(1.0, color='red', linestyle='--', label='Reference = 1.0')
ax.axvline(np.average(all_arearatios), color='orange', linestyle='--',
           label='Mean = {:.3f}'.format(np.average(all_arearatios)))
ax.set_xlabel('Area Ratio (pred / true)')
ax.set_ylabel('Count')
ax.set_title('Area Ratio Distribution (n={})'.format(len(all_arearatios)))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(_SEG_DIR, 'area_ratio_distribution_eval.png'), dpi=150)
plt.close()

# --- Image-level confusion matrix ---
all_y_bin = np.array(all_y_bin)
all_pred_bin = np.array(all_pred_bin)
seg_tp = int(np.sum((all_pred_bin == 1) & (all_y_bin == 1)))
seg_tn = int(np.sum((all_pred_bin == 0) & (all_y_bin == 0)))
seg_fp = int(np.sum((all_pred_bin == 1) & (all_y_bin == 0)))
seg_fn = int(np.sum((all_pred_bin == 0) & (all_y_bin == 1)))
cm_seg = np.array([[seg_tn, seg_fp],
                   [seg_fn, seg_tp]])
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm_seg, interpolation='nearest', cmap='Blues')
fig.colorbar(im, ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pred Neg', 'Pred Pos'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['True Neg', 'True Pos'])
ax.set_title('Image-Level Confusion Matrix')
for r in range(2):
    for c in range(2):
        ax.text(c, r, str(cm_seg[r, c]),
                ha='center', va='center', fontsize=14,
                color='white' if cm_seg[r, c] > cm_seg.max() / 2 else 'black')
plt.tight_layout()
plt.savefig(os.path.join(_SEG_DIR, 'confusion_matrix_eval.png'), dpi=150)
plt.close()

