from __future__ import division
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
import datasets.cifar as cifar
import datasets.imagenet as imagenet
from models.alexnet import AlexNet_cifar, AlexNet
from models.resnet_cifar_sup import ResNet18 as ResNet18_cifar
from models.resnet_cifar_sup import ResNet50 as ResNet50_cifar
from models.wideresnet import WideResNet
from models.resnet import resnet18, resnet50
from torch.utils.data import DataLoader
from utils import *
import objective
import logging

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

def adjust_learning_rate(lr_decay_steps, optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
	#steps = [120,160,200]
	steps = list(map(int, lr_decay_steps.split(',')))
	logging.info(steps)
	lr = args.lr
	if epoch < steps[0]:
		lr = args.lr
	elif epoch >= steps[0] and epoch < steps[1]:
		lr = args.lr * 0.1
	elif epoch >=steps[1] and epoch < steps[2]:
		lr = args.lr * 0.01
	else:
		lr = args.lr * 0.001
	#lr = args.lr * (0.1 ** (epoch // 100))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--stage', default='train', type=str)
	parser.add_argument('--dataset', default='imagenet', type=str)
	parser.add_argument('--lr', default=0.0012, type=float)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--gpus', default='0,1,2,3', type=str)
	parser.add_argument('--weight_decay', default=1e-5, type=float)
	parser.add_argument('--max_epoch', default=30, type=int)
	parser.add_argument('--lr_decay_steps', default='15,20,25', type=str)
	parser.add_argument('--exp', default='', type=str)
	parser.add_argument('--list', default='', type=str)
	parser.add_argument('--resume_path', default='', type=str)
	parser.add_argument('--pretrain_path', default='', type=str)
	parser.add_argument('--n_workers', default=32, type=int)

	parser.add_argument('--network', default='resnet50', type=str)

	global args
	args = parser.parse_args()

	if not os.path.exists(args.exp):
		os.makedirs(args.exp)
	if not os.path.exists(os.path.join(args.exp, 'runs')):
		os.makedirs(os.path.join(args.exp, 'runs'))
	if not os.path.exists(os.path.join(args.exp, 'models')):
		os.makedirs(os.path.join(args.exp, 'models'))
	if not os.path.exists(os.path.join(args.exp, 'logs')):
		os.makedirs(os.path.join(args.exp, 'logs'))

	# logger initialize
	logger = getLogger(args.exp)

	device_ids = list(map(lambda x: int(x), args.gpus.split(',')))
	device = torch.device('cuda: 0')


	train_loader, val_loader = cifar.get_semi_dataloader(args) if args.dataset.startswith('cifar') else imagenet.get_semi_dataloader(args)

	# create model
	if args.network == 'alexnet':
		network = AlexNet(128)
	elif args.network == 'alexnet_cifar':
		network = AlexNet_cifar(128)
	elif args.network == 'resnet18_cifar':
		network = ResNet18_cifar()
	elif args.network == 'resnet50_cifar':
		network = ResNet50_cifar()
	elif args.network == 'wide_resnet28':
		network = WideResNet(28, args.dataset == 'cifar10' and 10 or 100, 2)
	elif args.network == 'resnet18':
		network = resnet18()
	elif args.network == 'resnet50':
		network = resnet50()
	network = nn.DataParallel(network, device_ids=device_ids)
	network.to(device)

	classifier = nn.Linear(2048, 1000).to(device)
	# create optimizer

	parameters = network.parameters()
	optimizer = torch.optim.SGD(
		parameters,
		lr=args.lr,
		momentum=0.9,
		weight_decay=args.weight_decay,
	)

	cls_optimizer = torch.optim.SGD(
		classifier.parameters(),
		lr=args.lr*50,
		momentum=0.9,
		weight_decay=args.weight_decay,
	)


	cudnn.benchmark = True

	# create memory_bank
	global writer
	writer = SummaryWriter(comment='SemiSupervised', logdir=os.path.join(args.exp, 'runs'))

	# create criterion
	criterion = nn.CrossEntropyLoss()

	logging.info(beautify(args))
	start_epoch = 0
	if args.pretrain_path!= '' and args.pretrain_path!= 'none':
		logging.info('loading pretrained file from {}'.format(args.pretrain_path))
		checkpoint = torch.load(args.pretrain_path)
		state_dict = checkpoint['state_dict']
		valid_state_dict = {k: v for k, v in state_dict.items() if k in network.state_dict() and 'fc.' not in k}
		for k,v in network.state_dict().items():
			if k not in valid_state_dict:
				logging.info('{}: Random Init'.format(k))
				valid_state_dict[k] = v
		# logging.info(valid_state_dict.keys())
		network.load_state_dict(valid_state_dict)
	else:
		logging.info('Training SemiSupervised Learning From Scratch')

	logging.info('start training')
	best_acc = 0.0
	try:
		for i_epoch in range(start_epoch, args.max_epoch):
			train(i_epoch, network, classifier, criterion, optimizer, cls_optimizer, train_loader, device)

			checkpoint = {
				'epoch': i_epoch + 1,
				'state_dict': network.state_dict(),
				'optimizer': optimizer.state_dict(),
			}
			torch.save(checkpoint, os.path.join(args.exp, 'models', 'checkpoint.pth'))
			adjust_learning_rate(args.lr_decay_steps, optimizer, i_epoch)
			if i_epoch % 2 == 0:
				acc1, acc5 = validate(i_epoch, network, classifier, val_loader, device)
				if acc1 >= best_acc:
					best_acc = acc1
					torch.save(checkpoint, os.path.join(args.exp, 'models', 'best.pth'))
				writer.add_scalar('acc1', acc1, i_epoch+1)
				writer.add_scalar('acc5', acc5, i_epoch+1)

			if i_epoch in [30, 60, 120, 160, 200]:
				torch.save(checkpoint, os.path.join(args.exp, 'models', '{}.pth'.format(i_epoch+1)))

			logging.info(colorful('[Epoch: {}] val acc: {:.4f}/{:.4f}'.format(i_epoch, acc1, acc5)))
			logging.info(colorful('[Epoch: {}] best acc: {:.4f}'.format(i_epoch, best_acc)))

			with torch.no_grad():
				for name, param in network.named_parameters():
					if 'bn' not in name:
						writer.add_histogram(name, param, i_epoch)

			# cluster
	except KeyboardInterrupt as e:
		logging.info('KeyboardInterrupt at {} Epochs'.format(i_epoch))
		exit()

def train(i_epoch, network, classifier, criterion, optimizer, cls_optimizer, dataloader, device):
	network.train()
	losses = AvgMeter()
	pbar = tqdm(dataloader)
	for data in pbar:
		img = data[0].to(device)
		target = data[1].to(device)
		output = network(img, layer=6).to(device)
		output = classifier(output)
		loss = criterion(output, target)
		losses.add(loss.item())

		optimizer.zero_grad()
		cls_optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		cls_optimizer.step()

		lr = optimizer.param_groups[0]['lr']
		pbar.set_description("Epoch:{} [lr:{}]".format(i_epoch, lr))
		info = 'loss: {:.4f}'.format(losses.get())
		pbar.set_postfix(info=info)

	writer.add_scalar('loss', losses.get(), i_epoch)
	logging.info('Epoch {}: loss: {:.4f}'.format(i_epoch, losses.get()))

def validate(i_epoch, network, classifier, val_loader, device):
	# For validation, for each image, we find the closest neighbor in the
	# memory bank (training data), take its class! We compute the accuracy.

	network.eval()

	top1 = AverageMeter()
	top5 = AverageMeter()
	with torch.no_grad():
		pbar = tqdm(val_loader)
		for images, labels in pbar:
			batch_size = images.size(0)

			images = images.to(device)
			output = network(images, layer=6).to(device)
			output = classifier(output)
			acc1, acc5 = accuracy(output, labels.to(device), topk=(1,5))

			top1.update(acc1[0], images.size(0))
			top5.update(acc5[0], images.size(0))

			info = 'acc: {:.4f}/{:.4f}'.format(top1.avg, top5.avg)
			pbar.set_postfix(info=info)

	return top1.avg, top5.avg


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
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

if __name__ == '__main__':
	main()




