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
from utils import colorful
import datasets.list as listData
import random
import cv2
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
			'/home/real/ILSVRC2012_100/']

for path in DIR_LIST:
	if os.path.exists(path):
		HUN_IMAGENET_DIR = path
		break

assert IMAGENET_DIR is not None

class GaussianBlur(object):
	"""Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

	def __init__(self, sigma=[.1, 2.]):
		self.sigma = sigma

	def __call__(self, x):
		sigma = random.uniform(self.sigma[0], self.sigma[1])
		x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
		return x

def get_instance_dataloader(args):
	minimum_crop = 0.2
	logging.info(colorful('ResizedCrop from {} to 1'.format(minimum_crop)))
	if args.blur:
		train_transforms = transforms.Compose([
				transforms.RandomResizedCrop(224, scale=(minimum_crop, 1.)),
				transforms.RandomApply([
					transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
				], p=0.8),
				transforms.RandomGrayscale(p=0.2),
				transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			])
	else:
		train_transforms = transforms.Compose([
				transforms.RandomResizedCrop(224, scale=(minimum_crop, 1.)),
				transforms.RandomGrayscale(p=0.2),
				transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			])
	val_transforms = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
	train_dataset = ImageNetInstance(train=True, image_transforms=train_transforms, imagenet_dir=IMAGENET_DIR if args.dataset=='imagenet' else HUN_IMAGENET_DIR)
	val_dataset = ImageNetInstance(train=False, image_transforms=val_transforms, imagenet_dir=IMAGENET_DIR if args.dataset == 'imagenet' else HUN_IMAGENET_DIR)
	train_samples = train_dataset.dataset.samples
	train_labels = [train_samples[i][1] for i in range(len(train_samples))]
	train_ordered_labels = np.array(train_labels)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
						shuffle=True, pin_memory=True,
						num_workers=args.n_workers)	
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
						shuffle=False, pin_memory=True,
						num_workers=args.n_workers)

	return train_loader, val_loader, train_ordered_labels, train_dataset, val_dataset

def get_semi_dataloader(args):
	minimum_crop = 0.08
	logging.info(colorful('ResizedCrop from {} to 1'.format(minimum_crop)))
	train_transforms = transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(minimum_crop, 1.)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
	val_transforms = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

	train_dataset = listData.ListData(os.path.join(IMAGENET_DIR, 'train'), args.list, transforms=train_transforms)
	val_dataset = datasets.ImageFolder(os.path.join(IMAGENET_DIR, 'val'), val_transforms)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
						shuffle=True, pin_memory=True,
						num_workers=args.n_workers)	
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
						shuffle=False, pin_memory=True,
						num_workers=args.n_workers)

	return train_loader, val_loader

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

