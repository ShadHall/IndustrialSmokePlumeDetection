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
# Data handling for 4-channel classifier input (B2, B3, B4, B8).

import os
import numpy as np
from matplotlib import pyplot as plt
import rasterio as rio
import torch
from torchvision import transforms

from smoke_detection import dataset_paths as _dataset_paths

# set random seeds
torch.manual_seed(3)
np.random.seed(3)


class SmokePlumeDataset():
    """SmokePlume Dataset class.

    The image data directory is expected to contain two directories, one labeled
    `positive` and one labeled `negative`, containing the corresponding image
    files.

    The function `create_dataset` can be used as a wrapper to create a
    data set.

    :param datadir: (str) image directory root, has to contain `negative` and
                    `positive` subdirectories, required
    :param mult: (int) factor by which to multiply data set size, default=1
    :param transform: (`torchvision.transform` object) transformations to be
                      applied, default: `None`
    :param balance: (str) method for balancing the data set; `'upsample'` the
                    smaller of the two classes or `'downsample'` the larger
                    of the two. Anything else omits balancing.
    """

    def __init__(self, datadir=None, mult=1, transform=None,
                 balance='upsample'):

        if datadir is None:
            datadir = _dataset_paths.classification_split('train')

        self.datadir = datadir
        self.transform = transform

        self.imgfiles = []
        self.labels = []
        self.positive_indices = []
        self.negative_indices = []

        idx = 0
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    continue
                self.imgfiles.append(os.path.join(root, filename))
                if 'positive' in root:
                    self.labels.append(True)
                    self.positive_indices.append(idx)
                    idx += 1
                elif 'negative' in root:
                    self.labels.append(False)
                    self.negative_indices.append(idx)
                    idx += 1

        self.imgfiles = np.array(self.imgfiles)
        self.labels = np.array(self.labels)
        self.positive_indices = np.array(self.positive_indices)
        self.negative_indices = np.array(self.negative_indices)

        if balance == 'downsample':
            self.balance_downsample()
        elif balance == 'upsample':
            self.balance_upsample()

        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.positive_indices = np.array([*self.positive_indices] * mult)
            self.negative_indices = np.array([*self.negative_indices] * mult)

    def __len__(self):
        return len(self.imgfiles)

    def balance_downsample(self):
        subsample_idc = np.ravel([
            self.positive_indices,
            self.negative_indices[
                np.random.randint(0, len(self.negative_indices),
                                  len(self.positive_indices))]]).astype(int)

        self.imgfiles = self.imgfiles[subsample_idc]
        self.labels = self.labels[subsample_idc]
        self.positive_indices = np.arange(0, len(self.labels), 1)[
            self.labels == True]
        self.negative_indices = np.arange(0, len(self.labels), 1)[
            self.labels == False]

    def balance_upsample(self):
        subsample_idc = np.ravel([
            self.positive_indices[
                np.random.randint(0, len(self.positive_indices),
                                  len(self.negative_indices) -
                                  len(self.positive_indices))]]).astype(int)

        self.imgfiles = np.concatenate((self.imgfiles,
                                        self.imgfiles[subsample_idc]), axis=0)
        self.labels = np.concatenate((self.labels,
                                      self.labels[subsample_idc]),
                                     axis=0)

        self.positive_indices = np.arange(0, len(self.labels), 1)[
            self.labels == True]
        self.negative_indices = np.arange(0, len(self.labels), 1)[
            self.labels == False]

    def __getitem__(self, idx):
        """Read and preprocess one sample."""
        imgfile = rio.open(self.imgfiles[idx])
        # Sentinel-2 bands: B2(490), B3(560), B4(665), B8(842)
        imgdata = np.array([imgfile.read(i) for i in [2, 3, 4, 8]])

        if imgdata.shape[1] != 120:
            newimgdata = np.empty((4, 120, imgdata.shape[2]))
            newimgdata[:, :imgdata.shape[1], :] = imgdata[:, :imgdata.shape[1], :]
            newimgdata[:, imgdata.shape[1]:, :] = imgdata[:, imgdata.shape[1]-1:, :]
            imgdata = newimgdata
        if imgdata.shape[2] != 120:
            newimgdata = np.empty((4, 120, 120))
            newimgdata[:, :, :imgdata.shape[2]] = imgdata[:, :, :imgdata.shape[2]]
            newimgdata[:, :, imgdata.shape[2]:] = imgdata[:, :, imgdata.shape[2]-1:]
            imgdata = newimgdata

        sample = {'idx': idx,
                  'img': imgdata,
                  'lbl': self.labels[idx],
                  'imgfile': self.imgfiles[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def display(self, idx, offset=0.2, scaling=1.5):
        """Display RGB channels from the 4-channel sample."""
        imgdata = self[idx]['img'].numpy()

        # RGB = (B4, B3, B2) from [B2, B3, B4, B8]
        imgdata = offset + scaling * (
            np.dstack([imgdata[2], imgdata[1], imgdata[0]]) -
            np.min([imgdata[2], imgdata[1], imgdata[0]])) / \
                (np.max([imgdata[2], imgdata[1], imgdata[0]]) -
                 np.min([imgdata[2], imgdata[1], imgdata[0]]))

        f, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow((imgdata - np.min(imgdata, axis=(0, 1))) /
                  (np.max(imgdata, axis=(0, 1)) -
                   np.min(imgdata, axis=(0, 1))))

        return f


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return {'idx': sample['idx'],
                'img': torch.from_numpy(sample['img'].copy()),
                'lbl': sample['lbl'],
                'imgfile': sample['imgfile']}


class Normalize(object):
    """Normalize channels [B2, B3, B4, B8] with dataset statistics."""
    def __init__(self):
        self.channel_means = np.array([900.5, 1061.4, 1091.7, 2186.3])
        self.channel_stds = np.array([624.7, 640.8, 718.1, 947.9])

    def __call__(self, sample):
        sample['img'] = (sample['img'] - self.channel_means.reshape(
            sample['img'].shape[0], 1, 1)) / self.channel_stds.reshape(
            sample['img'].shape[0], 1, 1)
        return sample


class Randomize(object):
    """Randomize image orientation: flips, mirrors, 90-degree rotations."""
    def __call__(self, sample):
        imgdata = sample['img']
        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
        rot = np.random.randint(0, 4)
        imgdata = np.rot90(imgdata, rot, axes=(1, 2))

        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'lbl': sample['lbl'],
                'imgfile': sample['imgfile']}


class RandomCrop(object):
    """Randomly crop 90x90 pixel image (from 120x120)."""
    def __call__(self, sample):
        imgdata = sample['img']
        x, y = np.random.randint(0, 30, 2)

        return {'idx': sample['idx'],
                'img': imgdata.copy()[:, y:y+90, x:x+90],
                'lbl': sample['lbl'],
                'imgfile': sample['imgfile']}


def create_dataset(*args, apply_transforms=True, **kwargs):
    """Create a dataset; uses same input parameters as SmokePlumeDataset.
    :param apply_transforms: if `True`, apply available transformations
    :return: data set"""
    if apply_transforms:
        data_transforms = transforms.Compose([
            Normalize(),
            RandomCrop(),
            Randomize(),
            ToTensor()
        ])
    else:
        data_transforms = None

    data = SmokePlumeDataset(*args, **kwargs, transform=data_transforms)

    return data
