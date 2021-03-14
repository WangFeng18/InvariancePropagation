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
import datasets.svhn as svhn
import datasets.imagenet as imagenet
import datasets.wm811.loaders as wm811loader
from models.alexnet import AlexNet
from models.alexnet import AlexNet_cifar
from models.resnet_cifar import ResNet18 as ResNet18_cifar
from models.resnet_wm811 import ResNet18 as ResNet18_wm811
from models.resnet_cifar import ResNet50 as ResNet50_cifar
from models.resnet import resnet18, resnet50
from models.preact_resnet import PreActResNet18
from models.preact_resnet import PreActResNet50
from torch.utils.data import DataLoader
from utils import *
import objective
import logging
import torch.distributions.beta as beta
from models.wideresnet import WideResNetInstance

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--stage', default='train', type=str)

	parser.add_argument('--gpus', default='0,1,2,3', type=str)
	parser.add_argument('--max_epoch', default=800, type=int)
	parser.add_argument('--lr_decay_steps', default='400,600,700', type=str)
	parser.add_argument('--exp', default='', type=str)
	parser.add_argument('--res_path', default='', type=str)
	parser.add_argument('--resume_path', default='', type=str)

	parser.add_argument('--dataset', default='imagenet', type=str)
	parser.add_argument('--lr', default=0.03, type=float)
	parser.add_argument('--lr_decay_rate', default=0.1, type=float)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--weight_decay', default=5e-4, type=float)
	parser.add_argument('--n_workers', default=32, type=int)
	parser.add_argument('--n_background', default=4096, type=int)
	parser.add_argument('--t', default=0.07, type=float)
	parser.add_argument('--m', default=0.5, type=float)
	parser.add_argument('--dropout', action='store_true')
	parser.add_argument('--blur', action='store_true')
	parser.add_argument('--cos', action='store_true')

	parser.add_argument('--network', default='resnet18', type=str)
	parser.add_argument('--not_hardpos', action='store_true')
	parser.add_argument('--InvP', type=int, default=1)
	parser.add_argument('--ramp_up', default='binary', type=str)
	parser.add_argument('--lam_inv', default=0.6, type=float)
	parser.add_argument('--diffusion_layer', default=3, type=int)
	# for cifar 10 the best diffusion_layer is 3 and cifar 100 is 4
	# for imagenet I have only tested when diffusion_layer = 3
	parser.add_argument('--K_nearst', default=4, type=int)
	parser.add_argument('--n_pos', default=50, type=int)
	# for cifar10 the best n_pos is 20, for cifar 100 the best is 10 or 20
	parser.add_argument('--exclusive', default=1, type=int)
	parser.add_argument('--nonlinearhead', default=0, type=int)

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

	logger = getLogger(args.exp)

	device_ids = list(map(lambda x: int(x), args.gpus.split(',')))
	device = torch.device('cuda: 0')


	if args.dataset.startswith('cifar'):
		train_loader, val_loader, train_ordered_labels, train_dataset, val_dataset = cifar.get_dataloader(args) 
	elif args.dataset.startswith('imagenet'):
		train_loader, val_loader, train_ordered_labels, train_dataset, val_dataset = imagenet.get_instance_dataloader(args)
	elif args.dataset == 'svhn':
		train_loader, val_loader, train_ordered_labels, train_dataset, val_dataset = svhn.get_dataloader(args)
	elif args.dataset == 'wm811':
		train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_ordered_labels = wm811loader.get_dataloader(args)

	# create model
	if args.network == 'alexnet':
		network = alexnet(128)
	if args.network == 'alexnet_cifar':
		network = AlexNet_cifar(128)
	elif args.network == 'resnet18_cifar':
		network = ResNet18_cifar(128, dropout=args.dropout, non_linear_head=args.nonlinearhead)
	elif args.network == 'resnet18_wm811':
		network = ResNet18_wm811(128, dropout=args.dropout, non_linear_head=args.nonlinearhead)
	elif args.network == 'resnet50_cifar':
		network = ResNet50_cifar(128, dropout=args.dropout)
	elif args.network == 'wide_resnet28':
		network = WideResNetInstance(28, 2)
	elif args.network == 'resnet18':
		network = resnet18(non_linear_head=args.nonlinearhead)
	elif args.network == 'pre-resnet18':
		network = PreActResNet18(128)
	elif args.network == 'resnet50':
		network = resnet50(non_linear_head=args.nonlinearhead)
	elif args.network == 'pre-resnet50':
		network = PreActResNet50(128)
	network = nn.DataParallel(network, device_ids=device_ids)
	network.to(device)

	# create optimizer

	parameters = network.parameters()
	optimizer = torch.optim.SGD(
		parameters,
		lr=args.lr,
		momentum=0.9,
		weight_decay=args.weight_decay,
	)

	cudnn.benchmark = True

	# create memory_bank
	global writer
	writer = SummaryWriter(comment='InvariancePropagation', logdir=os.path.join(args.exp, 'runs'))
	memory_bank = objective.MemoryBank_v1(len(train_dataset), writer, device, m=args.m)

	# create criterion
	criterionA = objective.InvariancePropagationLoss(args.t, n_background=args.n_background, diffusion_layer=args.diffusion_layer, k=args.K_nearst, n_pos=args.n_pos, exclusive=args.exclusive, InvP=args.InvP, hard_pos=(not args.not_hardpos))

	if args.ramp_up == 'binary':
		ramp_up = lambda i_epoch: objective.BinaryRampUp(i_epoch, 30)
	elif args.ramp_up == 'gaussian':
		ramp_up = lambda i_epoch: objective.GaussianRampUp(i_epoch, 30, 5)
	elif args.ramp_up == 'zero':
		ramp_up = lambda i_epoch: 1

	logging.info(beautify(args))
	start_epoch = 0
	if args.resume_path != '':
		logging.info('loading pretrained file from {}'.format(args.resume_path))
		checkpoint = torch.load(args.resume_path)
		network.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		_memory_bank = checkpoint['memory_banks']
		try:
			_neigh = checkpoint['neigh']
			memory_bank.neigh = _neigh
		except:
			logging.info(colorful('The Pretrained Path has No NEIGH and require a epoch to re-calculate'))
		memory_bank.points = _memory_bank
		start_epoch = checkpoint['epoch']
	else:
		initialize_memorybank(network, train_loader, device, memory_bank)
	logging.info('start training')
	best_acc = 0.0

	try:
		for i_epoch in range(start_epoch, args.max_epoch):
			adjust_learning_rate(args.lr, args.lr_decay_steps, optimizer, i_epoch, lr_decay_rate=args.lr_decay_rate, cos=args.cos, max_epoch=args.max_epoch)
			acc = kNN(i_epoch, network, memory_bank, val_loader, train_ordered_labels, K=200, sigma=0.07)
			train(i_epoch, network, criterionA, optimizer, train_loader, device, memory_bank, ramp_up)

			save_name = 'checkpoint.pth'
			checkpoint = {
				'epoch': i_epoch + 1,
				'state_dict': network.state_dict(),
				'optimizer': optimizer.state_dict(),
				'memory_banks': memory_bank.points,
				'neigh': memory_bank.neigh,
			}
			torch.save(checkpoint, os.path.join(args.exp, 'models', save_name))

			# scheduler.step()
			# validate(network, memory_bank, val_loader, train_ordered_labels, device)
			# acc = kNN(i_epoch, network, memory_bank, val_loader, train_ordered_labels, K=200, sigma=0.07)
			if acc >= best_acc:
				best_acc = acc
				torch.save(checkpoint, os.path.join(args.exp, 'models', 'best.pth'))
			if i_epoch in [30, 60, 120, 160, 200, 400, 600]:
				torch.save(checkpoint, os.path.join(args.exp, 'models', '{}.pth'.format(i_epoch+1)))

			args.y_best_acc = best_acc
			logging.info(colorful('[Epoch: {}] val acc: {:.4f}'.format(i_epoch, acc)))
			logging.info(colorful('[Epoch: {}] best acc: {:.4f}'.format(i_epoch, best_acc)))
			writer.add_scalar('acc', acc, i_epoch+1)

	except KeyboardInterrupt as e:
		logging.info('KeyboardInterrupt at {} Epochs'.format(i_epoch))
		save_result(args)
		exit()

	save_result(args)


def initialize_memorybank(network, dataloader, device, memory_bank, refill=False):
	logging.info('start memorybank pointing filling')
	# if not args.entropy_loss:
	memory_bank.random_init_bank()
	logging.info('finish memorybank pointing filling')


def train(i_epoch, network, criterionA, optimizer, dataloader, device, memory_bank, ramp_up):
	#all_targets = np.array(dataloader.dataset.targets)
	network.train()
	losses_ins = AvgMeter()
	losses_inv = AvgMeter()
	losses = AvgMeter()
	all_weights = []
	n_neighbour = AvgMeter()
	pbar = tqdm(dataloader)

	for data in pbar:
		img = data[1].to(device)
		# normal_img = img[:,0,:,:,:]
		index = data[0].to(device)
		output = network(img).to(device)

		# Nearst Neighbour Set vs Invariance Propagation Set
		L_ins, L_inv = criterionA(output, index, memory_bank)
		# lossA = lossA_1 + args.lam_inv * lossA_2

		L = L_ins + args.lam_inv * ramp_up(i_epoch) * L_inv
		losses_ins.add(L_ins.item())
		losses_inv.add(0.0 if type(L_inv)==float else L_inv.item())
		losses.add(L.item())

		optimizer.zero_grad()
		L.backward()
		optimizer.step()

		with torch.no_grad():
			memory_bank.update_points(output.detach(), index)

		lr = optimizer.param_groups[0]['lr']
		pbar.set_description("Epoch:{} [lr:{}]".format(i_epoch, lr))
		info = 'L: {:.4f} = L_ins: {:.4f} + {:.3f} * L_inv: {:.4f}'.format(losses.get(), losses_ins.get(), args.lam_inv * ramp_up(i_epoch), losses_inv.get())
		pbar.set_postfix(info=info)

	writer.add_scalar('L', losses.get(), i_epoch)
	writer.add_scalar('L_ins', losses_ins.get(), i_epoch)
	writer.add_scalar('L_inv', losses_inv.get(), i_epoch)
	logging.info('Epoch {}: L: {:.4f}'.format(i_epoch, losses.get()))
	logging.info('Epoch {}: L_ins: {:.4f}'.format(i_epoch, losses_ins.get()))
	logging.info('Epoch {}: L_inv: {:.4f}'.format(i_epoch, losses_inv.get()))

def kNN(epoch, net, memory_bank, val_loader, train_ordered_labels, K=200, sigma=0.1):
	print(train_ordered_labels)
	net.eval()
	total = 0
	testsize = val_loader.dataset.__len__()

	trainLabels = torch.LongTensor(train_ordered_labels).cuda()
	print(trainLabels)
	C = trainLabels.max() + 1
	print(C)
	
	top1 = 0.
	top5 = 0.
	with torch.no_grad():
		retrieval_one_hot = torch.zeros(K, C).cuda()
		pbar = tqdm(val_loader)
		for batch_idx, data in enumerate(pbar):
			indexes = data[0]
			inputs = data[1]
			targets = data[2]
			targets = targets.cuda()
			batchSize = inputs.size(0)
			features = net(inputs).detach()
			features = objective.l2_normalize(features)

			dist = memory_bank.get_all_dot_products(features)

			yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
			candidates = trainLabels.view(1,-1).expand(batchSize, -1)
			retrieval = torch.gather(candidates, 1, yi)

			retrieval_one_hot.resize_(batchSize * K, C).zero_()
			retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
			yd_transform = yd.clone().div_(sigma).exp_()
			probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
			_, predictions = probs.sort(1, True)

			# Find which predictions match the target
			correct = predictions.eq(targets.data.view(-1,1))

			top1 = top1 + correct.narrow(1,0,1).sum().item()
			top5 = top5 + correct.narrow(1,0,5).sum().item()

			total += targets.size(0)

			pbar.set_postfix(info='Top1: {:.2f}  Top5: {:.2f}'.format(top1*100./total, top5*100./total))

	return top1/total

if __name__ == '__main__':
	main()