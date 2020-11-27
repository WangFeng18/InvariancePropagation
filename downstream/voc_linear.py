# fine-tune on different dataset
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import math
import time
import glob
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from sklearn import metrics
from models.resnet import resnet50, resnet18
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')


def accimage_loader(path):
	try:
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)

class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, feat):
		return feat.view(feat.size(0), -1)


parser = argparse.ArgumentParser()
parser.add_argument('--vocdir', type=str, required=False,
					default=os.path.expanduser('/data/VOC/VOCdevkit/VOC2007/'), help='pascal voc 2007 dataset')
parser.add_argument('--split', type=str, required=False, default='trainval',
					choices=['train', 'trainval'], help='training split')

parser.add_argument('--pretrain_path', default='', type=str)
parser.add_argument('--model', default='resnet50', type=str)
parser.add_argument('--exp', default='', type=str)

parser.add_argument('--dropout', default=0, type=int)
parser.add_argument('--nit', type=int, default=19*100,
					help='Number of training iterations')
parser.add_argument('--batch_size', type=int, default=16,
					help='Number of Batch Size')
parser.add_argument('--fc6_8', type=int, default=0,
					help='If true, train only the final classifier')
parser.add_argument('--train_batchnorm', type=int, default=0,
					help='If true, train batch-norm layer parameters')
parser.add_argument('--eval_random_crops', type=int, default=1,
					help='If true, eval on 10 random crops, otherwise eval on 10 fixed crops')
parser.add_argument('--stepsize', type=int, default=19*40, help='Decay step') 
parser.add_argument('--lr', type=float, required=False,
					default=3.0, help='learning rate')
parser.add_argument('--wd', type=float, required=False,
					default=1e-6, help='weight decay')
parser.add_argument('--seed', type=int, default=31, help='random seed')

os.system('ulimit -n 10000')

def main():
	args = parser.parse_args()
	print(args)

	# fix random seeds
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	# create model and move it to gpu
	if args.model == 'resnet50':
		model = resnet50(non_linear_head=True)
		model = nn.DataParallel(model)
		classifier = nn.Linear(7*7*2048, 20).cuda()
		classifier = nn.Sequential(
			nn.AdaptiveAvgPool2d((2,2)),
			Flatten(),
			nn.Linear(2*2*2048, 20),
		).cuda()
	elif args.model == 'resnet18':
		model = resnet18()
		model = nn.DataParallel(model)
		classifier = nn.Sequential(
			nn.Linear(512, 20),
		).cuda()

	ckpt = torch.load(args.pretrain_path)
	model.load_state_dict(ckpt['state_dict'])

	model.eval()

	# model.cuda()
	cudnn.benchmark = True

	# what partition of the data to use
	if args.split == 'train':
		args.test = 'val'
	elif args.split == 'trainval':
		args.test = 'test'
	# data loader
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
	dataset = VOC2007_dataset(args.vocdir, split=args.split, transform=transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomResizedCrop(224, scale=(0.2,1.0)),
		transforms.ToTensor(),
		normalize,
	]))

	loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=24, pin_memory=False)

	print('PASCAL VOC 2007 ' + args.split + ' dataset loaded')

	if args.fc6_8:
	   # freeze some layers
		for param in model.parameters():
			param.requires_grad = False
		# unfreeze batchnorm scaling
		if args.train_batchnorm:
			for layer in model.modules():
				if isinstance(layer, torch.nn.BatchNorm2d):
					for param in layer.parameters():
						param.requires_grad = True

	device = torch.device('cuda:0')
	model.to(device)
	# set optimizer
	# optimizer = torch.optim.SGD(
	# 	filter(lambda x: x.requires_grad, model.parameters()),
	# 	lr=args.lr,
	# 	momentum=0.9,
	# 	weight_decay=args.wd,
	# )
	cls_optimizer = torch.optim.SGD(
		filter(lambda x: x.requires_grad, classifier.parameters()),
		lr=args.lr,
		momentum=0.9,
		weight_decay=args.wd,
	)
	criterion = nn.BCEWithLogitsLoss(reduction='none')

	transform_eval = [
		transforms.Resize(256),
		transforms.RandomHorizontalFlip(),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize,
	]

	print('Test set')
	test_dataset = VOC2007_dataset(
		args.vocdir, split=args.test, transform=transforms.Compose(transform_eval))
	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=256,
		shuffle=False,
		num_workers=4,
		pin_memory=False,
	)

	print('Start training')
	it = 0
	losses = AverageMeter()
	count = 0
	while it < args.nit:
		print('Epoch {}'.format(count))
		it = train(
			loader,
			model,
			classifier,
			None,
			cls_optimizer,
			criterion,
			args.fc6_8,
			losses,
			it=it,
			total_iterations=args.nit,
			stepsize=args.stepsize,
			device=device
		)
		count += 1
		if count % 1 == 0:
			evaluate2(test_loader, model, classifier, args.eval_random_crops, device)

	print('Evaluation')
	map = evaluate2(test_loader, model, classifier, args.eval_random_crops, device)


def evaluate(loader, model, classifier, eval_random_crops, device):
	model.eval()
	classifier.eval()
	gts = []
	scr = []
	for crop in range(9 * eval_random_crops + 1):
		for i, (input, target) in enumerate(loader):
			# move input to gpu and optionally reshape it
			if len(input.size()) == 5:
				bs, ncrops, c, h, w = input.size()
				input = input.view(-1, c, h, w)
			#input = input.cuda(non_blocking=True)
			input = input.to(device)

			# forward pass without grad computation
			with torch.no_grad():
				output = model(input, 5)
				output.to(device)
				output = classifier(output)
			if crop < 1:
				scr.append(torch.sum(output, 0, keepdim=True).cpu().numpy())
				gts.append(target)
			else:
				scr[i] += output.cpu().numpy()
	gts = np.concatenate(gts, axis=0).T
	scr = np.concatenate(scr, axis=0).T
	aps = []
	for i in range(20):
		# Subtract eps from score to make AP work for tied scores
		ap = metrics.average_precision_score(
			gts[i][gts[i] <= 1], scr[i][gts[i] <= 1]-1e-5*gts[i][gts[i] <= 1])
		aps.append(ap)
	print(np.mean(aps), '  ', ' '.join(['%0.2f' % a for a in aps]))
	return np.mean(aps)


def evaluate2(loader, model, classifier, eval_random_crops, device):
	model.eval()
	classifier.eval()
	gts = []
	scr = []
	# for crop in range(9 * eval_random_crops + 1):
	for i, (input, target) in enumerate(loader):
		# move input to gpu and optionally reshape it
		#input = input.cuda(non_blocking=True)
		input = input.to(device)

		# forward pass without grad computation
		with torch.no_grad():
			# output = model(input, 5)
			output = model(input, 5, all_blocks=True)[-2]
			output.to(device)
			output = classifier(output)
		scr.append(output.cpu().numpy())
		gts.append(target)
	gts = np.concatenate(gts, axis=0).T
	scr = np.concatenate(scr, axis=0).T
	aps = []
	for i in range(20):
		# Subtract eps from score to make AP work for tied scores
		ap = metrics.average_precision_score(
			gts[i][gts[i] <= 1], scr[i][gts[i] <= 1]-1e-5*gts[i][gts[i] <= 1])
		aps.append(ap)
	print(np.mean(aps), '  ', ' '.join(['%0.2f' % a for a in aps]))
	return np.mean(aps)


def train(loader, model, classifier, optimizer, cls_optimizer, criterion, fc6_8, losses, it=0, total_iterations=None, stepsize=None, verbose=True, device=None):
	# to log
	batch_time = AverageMeter()
	data_time = AverageMeter()
	top1 = AverageMeter()
	end = time.time()

	current_iteration = it

	# use dropout for the MLP
	model.eval()
	# in the batch norms always use global statistics

	for (input, target) in loader:
		# measure data loading time
		data_time.update(time.time() - end)

		# adjust learning rate
		if current_iteration != 0 and current_iteration % stepsize == 0:
			for param_group in cls_optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * 0.5
				print('iter {0} learning rate is {1}'.format(
					current_iteration, param_group['lr']))

		# move input to gpu
		input = input.to(device)
		target = target.float().to(device)
		#input = input.cuda(non_blocking=True)

		# forward pass with or without grad computation
		output = model(input, 5, all_blocks=True)[-2]
		output.to(device)
		output = classifier(output)

		#target = target.float().cuda()
		mask = (target == 255)
		loss = torch.sum(criterion(output, target).masked_fill_(
			mask, 0)) / target.size(0)

		# backward
		# optimizer.zero_grad()
		cls_optimizer.zero_grad()
		loss.backward()
		# optimizer.step()
		cls_optimizer.step()
		# clip gradients
		# torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
		# and weights update

		# measure accuracy and record loss
		losses.update(loss.item(), input.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		if verbose is True and current_iteration % 1 == 0:
			print('Iteration[{0}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
					  current_iteration, batch_time=batch_time,
					  data_time=data_time, loss=losses))
		current_iteration = current_iteration + 1
		if total_iterations is not None and current_iteration == total_iterations:
			break
	return current_iteration


class VOC2007_dataset(torch.utils.data.Dataset):
	def __init__(self, voc_dir, split='train', transform=None):
		# Find the image sets
		image_set_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
		image_sets = glob.glob(os.path.join(
			image_set_dir, '*_' + split + '.txt'))
		assert len(image_sets) == 20
		# Read the labels
		self.n_labels = len(image_sets)
		images = defaultdict(lambda: -np.ones(self.n_labels, dtype=np.uint8))
		for k, s in enumerate(sorted(image_sets)):
			for l in open(s, 'r'):
				name, lbl = l.strip().split()
				lbl = int(lbl)
				# Switch the ignore label and 0 label (in VOC -1: not present, 0: ignore)
				if lbl < 0:
					lbl = 0
				elif lbl == 0:
					lbl = 255
				images[os.path.join(voc_dir, 'JPEGImages',
									name + '.jpg')][k] = lbl
		self.images = [(k, images[k]) for k in images.keys()]
		np.random.shuffle(self.images)
		self.transform = transform

	def __len__(self):
		return len(self.images)

	def __getitem__(self, i):
		#img = Image.open(self.images[i][0])
		#img = img.convert('RGB')
		#img = accimage.Image(self.images[i][0])
		img = pil_loader(self.images[i][0])
		if self.transform is not None:
			img = self.transform(img)
		return img, self.images[i][1]


if __name__ == '__main__':
	main()

