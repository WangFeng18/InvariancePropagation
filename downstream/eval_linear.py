"""
copied and modified from https://github.com/HobbitLong/CMC

"""
from __future__ import print_function

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from utils import getLogger, colorful
import logging

import tensorboard_logger as tb_logger

from torchvision import transforms, datasets

from models.preact_resnet import PreActResNet18, PreActResNet50
from models.resnet import resnet50, resnet18
from models.resnet_cifar import ResNet18 as resnet18_cifar
from models.alexnet import AlexNet, AlexNet_cifar
from models.LinearModel import LinearClassifierResNet, LinearClassifierAlexNet
from PIL import ImageFile
import datasets.cifar as cifar
import datasets.svhn as svhn
ImageFile.LOAD_TRUNCATED_IMAGES = True

def adjust_learning_rate(epoch, opt, optimizer):
	"""Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
	steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
	if steps > 0:
		new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr
			print(new_lr)


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


def parse_option():
	parser = argparse.ArgumentParser('argument for training')

	parser.add_argument('--save_freq', type=int, default=5)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--n_workers', type=int, default=32)
	parser.add_argument('--epochs', type=int, default=100)

	# optimization
	parser.add_argument('--learning_rate', type=float, default=30)
	parser.add_argument('--lr_decay_epochs', type=str, default='40,60,80')
	parser.add_argument('--lr_decay_rate', type=float, default=0.2)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--weight_decay', type=float, default=0)
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
	parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

	# model definition
	parser.add_argument('--data_folder', type=str, default='/home/user/ILSVRC2012/')
	parser.add_argument('--model', type=str, default='resnet50')
	parser.add_argument('--save_folder', default='', type=str)
	parser.add_argument('--model_path', type=str, default='')
	parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')

	# crop
	parser.add_argument('--crop', type=float, default=0.08, help='minimum crop')
	parser.add_argument('--dataset', type=str, default='imagenet')
	parser.add_argument('--resume', default='', type=str, metavar='PATH')
	parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ'])
	parser.add_argument('--bn', action='store_true', help='use parameter-free BN')
	parser.add_argument('--adam', action='store_true', help='use adam optimizer')
	parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
	opt = parser.parse_args()

	opt.tb_folder = os.path.join(opt.save_folder, 'runs')
	if not os.path.exists(opt.save_folder):
		os.makedirs(opt.save_folder)
	if not os.path.exists(opt.tb_folder):
		os.makedirs(opt.tb_folder)
	if not os.path.exists(os.path.join(opt.save_folder, 'logs')):
		os.makedirs(os.path.join(opt.save_folder, 'logs'))
	if opt.dataset == 'imagenet':
		if 'alexnet' not in opt.model:
			opt.crop = 0.08

	iterations = opt.lr_decay_epochs.split(',')
	opt.lr_decay_epochs = list([])
	for it in iterations:
		opt.lr_decay_epochs.append(int(it))

	opt.model_name = opt.model_path.split('/')[-2]
	opt.model_name = '{}_bsz_{}_lr_{}_decay_{}_crop_{}'.format(opt.model_name, opt.batch_size, opt.learning_rate, opt.weight_decay, opt.crop)
	opt.model_name = '{}_aug_{}'.format(opt.model_name, opt.aug)
	if opt.bn:
		opt.model_name = '{}_useBN'.format(opt.model_name)
	if opt.adam:
		opt.model_name = '{}_useAdam'.format(opt.model_name)

	if opt.dataset == 'imagenet100':
		opt.n_label = 100
	if opt.dataset == 'imagenet':
		opt.n_label = 1000
	if opt.dataset == 'cifar10':
		opt.n_label = 10
	if opt.dataset == 'cifar100':
		opt.n_label = 100
	if opt.dataset == 'svhn':
		opt.n_label = 10

	return opt


def main():

	global best_acc1
	best_acc1 = 0

	args = parse_option()

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	# set the data loader
	train_folder = os.path.join(args.data_folder, 'train')
	val_folder = os.path.join(args.data_folder, 'val')


	logger = getLogger(args.save_folder)
	if args.dataset.startswith('imagenet'):
		image_size = 224
		crop_padding = 32
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		normalize = transforms.Normalize(mean=mean, std=std)
		if args.aug == 'NULL':
			train_transform = transforms.Compose([
				transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			])
		elif args.aug == 'CJ':
			train_transform = transforms.Compose([
				transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
				transforms.RandomGrayscale(p=0.2),
				transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			])
		else:
			raise NotImplemented('augmentation not supported: {}'.format(args.aug))

		train_dataset = datasets.ImageFolder(train_folder, train_transform)
		val_dataset = datasets.ImageFolder(
			val_folder,
			transforms.Compose([
				transforms.Resize(image_size + crop_padding),
				transforms.CenterCrop(image_size),
				transforms.ToTensor(),
				normalize,
			])
		)

		print(len(train_dataset))
		train_sampler = None

		train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.n_workers, pin_memory=False, sampler=train_sampler)

		val_loader = torch.utils.data.DataLoader( val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=False)
	elif args.dataset.startswith('cifar'):
		train_loader, val_loader = cifar.get_linear_dataloader(args)
	elif args.dataset.startswith('svhn'):
		train_loader, val_loader = svhn.get_linear_dataloader(args)

	# create model and optimizer
	if args.model == 'alexnet':
		if args.layer == 6:
			args.layer = 5
		model = AlexNet(128)
		model = nn.DataParallel(model)
		classifier = LinearClassifierAlexNet(args.layer, args.n_label, 'avg')
	elif args.model == 'alexnet_cifar':
		if args.layer == 6:
			args.layer = 5
		model = AlexNet_cifar(128)
		model = nn.DataParallel(model)
		classifier = LinearClassifierAlexNet(args.layer, args.n_label, 'avg', cifar=True)
	elif args.model == 'resnet50':
		model = resnet50()
		model = nn.DataParallel(model)
		classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 1)
	elif args.model == 'resnet18':
		model = resnet18()
		model = nn.DataParallel(model)
		classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 1, bottleneck=False)
	elif args.model == 'resnet18_cifar':
		model = resnet18_cifar()
		model = nn.DataParallel(model)
		classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 1, bottleneck=False)
	elif args.model == 'resnet50x2':
		model = InsResNet50(width=2)
		classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 2)
	elif args.model == 'resnet50x4':
		model = InsResNet50(width=4)
		classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 4)
	else:
		raise NotImplementedError('model not supported {}'.format(args.model))

	print('==> loading pre-trained model')
	ckpt = torch.load(args.model_path)
	model.load_state_dict(ckpt['state_dict'])
	print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
	print('==> done')

	model = model.cuda()
	classifier = classifier.cuda()

	criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

	if not args.adam:
		optimizer = torch.optim.SGD(classifier.parameters(),
									lr=args.learning_rate,
									momentum=args.momentum,
									weight_decay=args.weight_decay)
	else:
		optimizer = torch.optim.Adam(classifier.parameters(),
									 lr=args.learning_rate,
									 betas=(args.beta1, args.beta2),
									 weight_decay=args.weight_decay,
									 eps=1e-8)

	model.eval()
	cudnn.benchmark = True

	# optionally resume from a checkpoint
	args.start_epoch = 1
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume, map_location='cpu')
			# checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch'] + 1
			classifier.load_state_dict(checkpoint['classifier'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			best_acc1 = checkpoint['best_acc1']
			print(best_acc1.item())
			best_acc1 = best_acc1.cuda()
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
			if 'opt' in checkpoint.keys():
				# resume optimization hyper-parameters
				print('=> resume hyper parameters')
				if 'bn' in vars(checkpoint['opt']):
					print('using bn: ', checkpoint['opt'].bn)
				if 'adam' in vars(checkpoint['opt']):
					print('using adam: ', checkpoint['opt'].adam)
				#args.learning_rate = checkpoint['opt'].learning_rate
				# args.lr_decay_epochs = checkpoint['opt'].lr_decay_epochs
				args.lr_decay_rate = checkpoint['opt'].lr_decay_rate
				args.momentum = checkpoint['opt'].momentum
				args.weight_decay = checkpoint['opt'].weight_decay
				args.beta1 = checkpoint['opt'].beta1
				args.beta2 = checkpoint['opt'].beta2
			del checkpoint
			torch.cuda.empty_cache()
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	# tensorboard
	tblogger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

	# routine
	best_acc = 0.0
	for epoch in range(args.start_epoch, args.epochs + 1):

		adjust_learning_rate(epoch, args, optimizer)
		print("==> training...")

		time1 = time.time()
		train_acc, train_acc5, train_loss = train(epoch, train_loader, model, classifier, criterion, optimizer, args)
		time2 = time.time()
		logging.info('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

		logging.info('Epoch: {}, lr:{} , train_loss: {:.4f}, train_acc: {:.4f}/{:.4f}'.format(epoch, optimizer.param_groups[0]['lr'], train_loss, train_acc, train_acc5))

		tblogger.log_value('train_acc', train_acc, epoch)
		tblogger.log_value('train_acc5', train_acc5, epoch)
		tblogger.log_value('train_loss', train_loss, epoch)
		tblogger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

		test_acc, test_acc5, test_loss = validate(val_loader, model, classifier, criterion, args)

		if test_acc >= best_acc:
			best_acc = test_acc

		logging.info(colorful('Epoch: {}, val_loss: {:.4f}, val_acc: {:.4f}/{:.4f}, best_acc: {:.4f}'.format(epoch, test_loss, test_acc, test_acc5, best_acc)))
		tblogger.log_value('test_acc', test_acc, epoch)
		tblogger.log_value('test_acc5', test_acc5, epoch)
		tblogger.log_value('test_loss', test_loss, epoch)

		# save the best model
		if test_acc > best_acc1:
			best_acc1 = test_acc
			state = {
				'opt': args,
				'epoch': epoch,
				'classifier': classifier.state_dict(),
				'best_acc1': best_acc1,
				'optimizer': optimizer.state_dict(),
			}
			save_name = '{}_layer{}.pth'.format(args.model, args.layer)
			save_name = os.path.join(args.save_folder, save_name)
			print('saving best model!')
			torch.save(state, save_name)

		# save model
		if epoch % args.save_freq == 0:
			print('==> Saving...')
			state = {
				'opt': args,
				'epoch': epoch,
				'classifier': classifier.state_dict(),
				'best_acc1': test_acc,
				'optimizer': optimizer.state_dict(),
			}
			save_name = 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
			save_name = os.path.join(args.save_folder, save_name)
			print('saving regular model!')
			torch.save(state, save_name)

		# tensorboard logger
		pass


def set_lr(optimizer, lr):
	"""
	set the learning rate
	"""
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def train(epoch, train_loader, model, classifier, criterion, optimizer, opt):
	"""
	one epoch training
	"""

	model.eval()
	classifier.train()

	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	end = time.time()
	pbar = tqdm(train_loader)
	for idx, (input, target) in enumerate(pbar):
		# measure data loading time
		input = input.cuda(opt.gpu, non_blocking=True).float()
		target = target.cuda(opt.gpu, non_blocking=True)

		# ===================forward=====================
		with torch.no_grad():
			feat = model(input, opt.layer)
			feat = feat.detach()

		output = classifier(feat)
		loss = criterion(output, target)

		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), input.size(0))
		top1.update(acc1[0], input.size(0))
		top5.update(acc5[0], input.size(0))

		# ===================backward=====================
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# ===================meters=====================

		# print info
		lr = optimizer.param_groups[0]['lr']
		pbar.set_description("Epoch:{} [lr:{}]".format(epoch, lr))

		info = 'loss: {:.4f}, acc: {:.4f} ({:.4f})'.format(losses.avg, top1.avg, top5.avg)
		pbar.set_postfix(info=info)

	return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, classifier, criterion, opt):
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()
	classifier.eval()

	with torch.no_grad():
		end = time.time()
		pbar = tqdm(val_loader)
		for idx, (input, target) in enumerate(pbar):
			input = input.cuda(opt.gpu, non_blocking=True).float()
			target = target.cuda(opt.gpu, non_blocking=True)

			# compute output
			feat = model(input, opt.layer)
			feat = feat.detach()
			output = classifier(feat)
			loss = criterion(output, target)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), input.size(0))
			top1.update(acc1[0], input.size(0))
			top5.update(acc5[0], input.size(0))
			
			info = 'loss: {:.4f}, acc: {:.4f} ({:.4f})'.format(losses.avg, top1.avg, top5.avg)
			pbar.set_postfix(info=info)

	return top1.avg, top5.avg, losses.avg


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


if __name__ == '__main__':
	best_acc1 = 0
	main()

#CUDA_VISIBLE_DEVICES=0,1 python rotation_instance.py --dataset 'imagenet_100' --batch_size 64 --gpus '0,1' --weight_decay 1e-4 --max_epoch 400 --network 'pre-resnet50' --exp '/data/HierachicalAggregation_exp/ImageNet_100/rotation_instance_aug[preresnet50][lam3.0][AB]/' --n_workers 32 --t 0.07 --dist_t 0.07 --lam 3.0 --scheme 'AB' --infonce

# CUDA_VISIBLE_DEVICES=1 python -m downstream.eval_linear --learning_rate 1.0 --model 'resnet18_cifar' --save_folder '/data/HierachicalAggregation_exp/cifar10/Ablation_EasyPos/linear/' --model_path '/data/HierachicalAggregation_exp/cifar10/Ablation_EasyPos/[mix=False][network=resnet18_cifar][lam_inv=0.6][lam_mix=1.0][diffusion_layer=3][K_nearst=4][n_pos=20][exclusive=0][max_epoch=200][ramp_up=binary][nonlinearhead=0]/models/best.pth' --dataset 'cifar10' --gpu '0'
# CUDA_VISIBLE_DEVICES=0,1 python -m downstream.eval_linear --save_folder '/data/HierachicalAggregation_exp/ImageNet/InvariancePropagation_160190/[mix=False][network=resnet50][lam_inv=0.6][lam_mix=1.0][diffusion_layer=3][K_nearst=4][n_pos=50][exclusive=1][max_epoch=800][ramp_up=binary][nonlinearhead=0][t=0.07][weight_decay=0.0001]/linear/' --model_path '/data/HierachicalAggregation_exp/ImageNet/InvariancePropagation_160190/[mix=False][network=resnet50][lam_inv=0.6][lam_mix=1.0][diffusion_layer=3][K_nearst=4][n_pos=50][exclusive=1][max_epoch=800][ramp_up=binary][nonlinearhead=0][t=0.07][weight_decay=0.0001]/models/best.pth' --dataset 'imagenet' --gpu 0