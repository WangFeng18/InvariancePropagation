"""
Dataset for ImageNet. Borrowed from
https://github.com/neuroailab/LocalAggregation-Pytorch
"""
from __future__ import division
import os
import logging
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
from PIL import ImageFile
from PIL import ImageFilter
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_DIR = None
DIR_LIST = ['/home/user/ILSVRC2012/',
			'/home/real/ILSVRC2012/',
			'/data/ILSVRC2012/']

for path in DIR_LIST:
	if os.path.exists(path):
		IMAGENET_DIR = path
		break

DIR_LIST = ['/home/user/ILSVRC2012_100/',
			'/home/real/ILSVRC2012_100/',
			'/data/ILSVRC2012_100/']

for path in DIR_LIST:
	if os.path.exists(path):
		HUN_IMAGENET_DIR = path
		break

assert IMAGENET_DIR is not None


def get_labels(dataset):
	train_dataset = ImageNetInstance(train=True, image_transforms=None, imagenet_dir=IMAGENET_DIR if dataset=='imagenet' else HUN_IMAGENET_DIR)
	train_samples = train_dataset.dataset.samples
	train_labels = [train_samples[i][1] for i in range(len(train_samples))]
	train_ordered_labels = np.array(train_labels)

	return train_ordered_labels


class ImageNetInstance(data.Dataset):
	def __init__(self, train=True, imagenet_dir=IMAGENET_DIR, image_transforms=None):
		super(ImageNetInstance, self).__init__()
		split_dir = 'train' if train else 'val'
		self.imagenet_dir = os.path.join(imagenet_dir, split_dir)
		self.dataset = datasets.ImageFolder(self.imagenet_dir, image_transforms)

	def __getitem__(self, index):
		image_data = list(self.dataset.__getitem__(index))
		# important to return the index!
		data = [index] + image_data
		return tuple(data)

	def __len__(self):
		return len(self.dataset)

