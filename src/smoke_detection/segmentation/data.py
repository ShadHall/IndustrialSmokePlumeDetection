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
# Data handling for the 4-channel segmentation model (B2, B3, B4, B8).

import os
import numpy as np
import json
from matplotlib import pyplot as plt
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import Polygon
import torch
from torchvision import transforms

from smoke_detection import dataset_paths

torch.manual_seed(3)
np.random.seed(3)


def label_image_url_to_tif_key(image_url):
    """Map Label Studio image URL to on-disk GeoTIFF basename.

    Zenodo JSON uses ':' in timestamps; extracted GeoTIFFs use '_' instead.
    """
    key = "-".join(image_url.split('-')[1:]).replace('.png', '.tif')
    return key.replace(':', '_')


class SmokePlumeSegmentationDataset():
    """SmokePlumeSegmentation dataset class."""

    def __init__(self, datadir=None, seglabeldir=None, mult=1, transform=None):
        if datadir is None or seglabeldir is None:
            di, dl = dataset_paths.segmentation_split('train')
            datadir = datadir if datadir is not None else di
            seglabeldir = seglabeldir if seglabeldir is not None else dl

        self.datadir = datadir
        self.transform = transform
        self.imgfiles = []
        self.labels = []
        self.seglabels = []
        self.positive_indices = []
        self.negative_indices = []

        seglabels = []
        segfile_lookup = {}
        for i, seglabelfile in enumerate(os.listdir(seglabeldir)):
            segdata = json.load(open(os.path.join(seglabeldir, seglabelfile), 'r'))
            seglabels.append(segdata)
            segfile_lookup[label_image_url_to_tif_key(segdata['data']['image'])] = i

        idx = 0
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    continue
                if filename not in segfile_lookup.keys():
                    continue
                polygons = []
                for completions in seglabels[segfile_lookup[filename]]['completions']:
                    for result in completions['result']:
                        polygons.append(
                            np.array(result['value']['points'] +
                                     [result['value']['points'][0]]) * 1.2)
                if 'positive' in root and polygons != []:
                    self.labels.append(True)
                    self.positive_indices.append(idx)
                    self.imgfiles.append(os.path.join(root, filename))
                    self.seglabels.append(polygons)
                    idx += 1

        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    continue
                if idx >= len(self.positive_indices) * 2:
                    break
                if 'negative' in root:
                    self.labels.append(False)
                    self.negative_indices.append(idx)
                    self.imgfiles.append(os.path.join(root, filename))
                    self.seglabels.append([])
                    idx += 1

        self.imgfiles = np.array(self.imgfiles)
        self.labels = np.array(self.labels)
        self.positive_indices = np.array(self.positive_indices)
        self.negative_indices = np.array(self.negative_indices)

        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.positive_indices = np.array([*self.positive_indices] * mult)
            self.negative_indices = np.array([*self.negative_indices] * mult)
            self.seglabels = self.seglabels * mult

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, idx):
        imgfile = rio.open(self.imgfiles[idx])
        # Sentinel-2: B2(490), B3(560), B4(665), B8(842)
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

        fptdata = np.zeros(imgdata.shape[1:], dtype=np.uint8)
        polygons = self.seglabels[idx]
        shapes = []
        if len(polygons) > 0:
            for pol in polygons:
                try:
                    pol = Polygon(pol)
                    shapes.append(pol)
                except ValueError:
                    continue
            fptdata = rasterize(((g, 1) for g in shapes),
                                out_shape=fptdata.shape,
                                all_touched=True)

        sample = {'idx': idx,
                  'img': imgdata,
                  'fpt': fptdata,
                  'imgfile': self.imgfiles[idx]}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def display(self, idx, offset=0.2, scaling=1.5):
        sample = self[idx]
        imgdata = sample['img']
        fptdata = sample['fpt']

        # RGB = (B4, B3, B2)
        imgdata = offset + scaling * (
            np.dstack([imgdata[2], imgdata[1], imgdata[0]]) -
            np.min([imgdata[2], imgdata[1], imgdata[0]])) / \
                (np.max([imgdata[2], imgdata[1], imgdata[0]]) -
                 np.min([imgdata[2], imgdata[1], imgdata[0]]))

        f, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(imgdata)
        ax[1].imshow(fptdata)
        return f


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return {'idx': sample['idx'],
                'img': torch.from_numpy(sample['img'].copy()),
                'fpt': torch.from_numpy(sample['fpt'].copy()),
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
    """Randomize image orientation: flips, mirrors, 90-degree rotations.
    Applied consistently to both image and segmentation mask."""
    def __call__(self, sample):
        imgdata = sample['img']
        fptdata = sample['fpt']

        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
            fptdata = np.flip(fptdata, 1)
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
            fptdata = np.flip(fptdata, 0)
        rot = np.random.randint(0, 4)
        imgdata = np.rot90(imgdata, rot, axes=(1, 2))
        fptdata = np.rot90(fptdata, rot, axes=(0, 1))

        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'fpt': fptdata.copy(),
                'imgfile': sample['imgfile']}


class RandomCrop(object):
    """Randomly crop 90x90 pixel image and mask (from 120x120)."""
    def __call__(self, sample):
        imgdata = sample['img']
        x, y = np.random.randint(0, 30, 2)
        return {'idx': sample['idx'],
                'img': imgdata.copy()[:, y:y+90, x:x+90],
                'fpt': sample['fpt'].copy()[y:y+90, x:x+90],
                'imgfile': sample['imgfile']}


def create_dataset(*args, apply_transforms=True, **kwargs):
    """Create a dataset; uses same input parameters as SmokePlumeSegmentationDataset.
    :param apply_transforms: if `True`, apply available transformations
    :return: data set"""
    if apply_transforms:
        data_transforms = transforms.Compose([
            Normalize(),
            Randomize(),
            RandomCrop(),
            ToTensor()
        ])
    else:
        data_transforms = None

    data = SmokePlumeSegmentationDataset(*args, **kwargs,
                                         transform=data_transforms)
    return data
