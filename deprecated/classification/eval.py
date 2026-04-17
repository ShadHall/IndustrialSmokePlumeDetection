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
# based on the test data set.

import os
import sys
import argparse
import numpy as np
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import dataset_paths

from model import *
from data import create_dataset

np.random.seed(100)
torch.manual_seed(100)

_CLS_DIR = os.path.dirname(os.path.abspath(__file__))

# setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default=os.path.join(_CLS_DIR, "classification.model"),
    help="Path to model checkpoint (.model state_dict)",
)
args = parser.parse_args()

# load data
batch_size = 10 # 1 to create diagnostic images, any value otherwise
testdata = create_dataset(datadir=dataset_paths.classification_split('test'))
all_dl = DataLoader(testdata, batch_size=batch_size, shuffle=True)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# load model
model_path = os.path.abspath(os.path.expanduser(args.model))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


# implant hooks for resnet layers
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.relu.register_forward_hook(get_activation('conv1'))
model.layer1.register_forward_hook(get_activation('layer1'))
model.layer2.register_forward_hook(get_activation('layer2'))
model.layer3.register_forward_hook(get_activation('layer3'))
model.layer4.register_forward_hook(get_activation('layer4'))

# run through test data set
tp = 0
tn = 0
fp = 0
fn = 0
all_scores = []   # raw output[0] per sample, for ROC curve
all_labels = []   # true label y[0] per sample, for ROC curve
for i, batch in progress:
    x, y = batch['img'].float().to(device), batch['lbl'].float().to(device)

    output = model(x)
    prediction = 1 if output[0] > 0 else 0

    all_scores.append(output[0].item())
    all_labels.append(int(y[0].item()))

    if prediction == 1 and y[0] == 1:
        res = 'true_pos'
        tp += 1
    elif prediction == 0 and y[0] == 0:
        res = 'true_neg'
        tn += 1
    elif prediction == 0 and y[0] == 1:
        res = 'false_neg'
        fn += 1
    elif prediction == 1 and y[0] == 0:
        res = 'false_pos'
        fp += 1

    if batch_size == 1:

        # create plot
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(1, 3))

        # rgb plot
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
        ax1.set_ylabel('RGB', fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # false color plot
        # RGB = (Aerosols; Water Vapor; SWIR1)
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
        ax2.set_ylabel('False Color', fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # layer2 activations plot
        map_layer2 = ax3.imshow(activation['layer2'].sum(axis=(0, 1)),
                                vmin=50, vmax=150)
        ax3.set_ylabel('Layer2', fontsize=8)
        ax3.set_xticks([])
        ax3.set_yticks([])

        f.subplots_adjust(0.05, 0.02, 0.95, 0.9, 0.05, 0.05)

        plt.savefig((res+os.path.split(batch['imgfile'][0])[1]).\
                    replace('.tif', '_eval.png').replace(':', '_'),
                    dpi=200, bbox_inches='tight')
        plt.close()

print('test set accuracy:', (tp + tn) / (tp + tn + fp + fn))

# --- Confusion matrix heatmap ---
cm = np.array([[tn, fp],
               [fn, tp]])
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
fig.colorbar(im, ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pred Neg', 'Pred Pos'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['True Neg', 'True Pos'])
ax.set_title('Confusion Matrix')
for r in range(2):
    for c in range(2):
        ax.text(c, r, str(cm[r, c]),
                ha='center', va='center', fontsize=14,
                color='white' if cm[r, c] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.savefig(os.path.join(_CLS_DIR, 'confusion_matrix_eval.png'), dpi=150)
plt.close()

# --- ROC curve ---
fpr, tpr, _ = roc_curve(all_labels, all_scores)
try:
    auc = roc_auc_score(all_labels, all_scores)
except ValueError:
    auc = float('nan')
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc))
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(_CLS_DIR, 'roc_curve_eval.png'), dpi=150)
plt.close()

