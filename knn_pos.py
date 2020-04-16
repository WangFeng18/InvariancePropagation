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
from models.alexnet import alexnet
from models.resnet_cifar import ResNet18 as ResNet18_cifar
from models.resnet import resnet18, resnet50
from models.preact_resnet import PreActResNet18
from models.preact_resnet import PreActResNet50
# from models.alexnet_cifar import AlexNet
from torch.utils.data import DataLoader
from utils import *
import objective
import logging
from scipy.special import comb, perm
from kNN import kNN

class HardNegativePositivePointLoss(nn.Module):
	def __init__(self, t):
		super(HardNegativePositivePointLoss, self).__init__()
		self.t = t
		self.k = 50

	def forward(self, points, point_indices, memory_bank, first):
		norm_points = objective.l2_normalize(points)
		points_sim = self._exp(memory_bank.get_all_dot_products(norm_points))
		positive_sim = points_sim[list(range(points_sim.size(0))), point_indices]
		hard_negatives_sim, hn_indices = points_sim.topk(k=4096, dim=1, largest=True, sorted=True)

		if first:
			return -(positive_sim/hard_negatives_sim.sum(dim=1) + 1e-7).log().mean(), None

		has_self = (hn_indices[:, :self.k+1] == point_indices.unsqueeze(dim=1)).sum(dim=1)
		all_other_positive_sim = hard_negatives_sim[:, :self.k+1].sum(dim=1) - has_self * positive_sim

		# return -(positive_sim/hard_negatives_sim.sum(dim=1) + 1e-7).log().mean(), -(all_other_positive_sim/(hard_negatives_sim.sum(dim=1)-positive_sim)).log().mean()
		return -(positive_sim/hard_negatives_sim.sum(dim=1) + 1e-7).log().mean(), -(all_other_positive_sim/hard_negatives_sim.sum(dim=1)).log().mean()


	def _exp(self, dot_prods):
		# Z = 2876934.2 / 1281167 * self.data_len
		return torch.exp(dot_prods / self.t)


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
	steps = [120,160,200]
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
	parser.add_argument('--lr', default=0.03, type=float)
	parser.add_argument('--batch_size', default=256, type=int)
	parser.add_argument('--gpus', default='0,1,2,3', type=str)
	parser.add_argument('--weight_decay', default=5e-4, type=float)
	parser.add_argument('--max_epoch', default=240, type=int)
	parser.add_argument('--network', default='resnet18', type=str)
	parser.add_argument('--exp', default='', type=str)
	parser.add_argument('--resume_path', default='', type=str)
	parser.add_argument('--pretrain_path', default='', type=str)
	parser.add_argument('--n_workers', default=32, type=int)
	parser.add_argument('--t', default=0.07, type=float)

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
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	fh = logging.FileHandler(os.path.join(args.exp, 'logs', 'log.txt'))
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	logger.addHandler(fh)

	device_ids = list(map(lambda x: int(x), args.gpus.split(',')))
	device = torch.device('cuda: 0')


	train_loader, val_loader, train_ordered_labels, train_dataset, val_dataset = cifar.get_dataloader(args) if args.dataset.startswith('cifar') else imagenet.get_instance_dataloader(args)

	# create model
	if args.network == 'alexnet':
		network = alexnet(128)
	elif args.network == 'resnet18_cifar':
		network = ResNet18_cifar(128)
	elif args.network == 'resnet18':
		network = resnet18()
	elif args.network == 'pre-resnet18':
		network = PreActResNet18(128)
	elif args.network == 'resnet50':
		network = resnet50()
	elif args.network == 'pre-resnet50':
		network = PreActResNet50(128)
	network = nn.DataParallel(network, device_ids=device_ids)
	network.to(device)

	
	# create optimizer

	if 'imagenet' in args.dataset:
		args.weight_decay = 1e-4
	if args.network == 'pre-resnet18' or args.network == 'pre-resnet50':
		logging.info(colorful('Exclude bns from weight decay, copied from LocalAggregation proposed by Zhuang et al [ICCV 2019]'))
		parameters = exclude_bn_weight_bias_from_weight_decay(network, weight_decay=args.weight_decay)
	else:
		parameters = network.parameters()

	optimizer = torch.optim.SGD(
		parameters,
		lr=args.lr,
		momentum=0.9,
		weight_decay=args.weight_decay,
	)

	# create scheduler
	# if args.dataset == 'imagenet':
	# 	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160, 190], gamma=0.1)
	# else:
	# 	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

	cudnn.benchmark = True

	# create memory_bank
	global writer
	writer = SummaryWriter(comment='InvariancePropagation', logdir=os.path.join(args.exp, 'runs'))
	memory_bank = objective.MemoryBank_v1(len(train_dataset), train_ordered_labels, writer, device, m=0.5)

	# create criterion
	criterion = HardNegativePositivePointLoss(args.t)

	print(args)
	start_epoch = 0
	if args.pretrain_path!= '' and args.pretrain_path!= 'none':
		logging.info('loading pretrained file from {}'.format(args.pretrain_path))
		checkpoint = torch.load(args.pretrain_path)
		network.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		_memory_bank = checkpoint['memory_banks']
		memory_bank.points = _memory_bank
		start_epoch = checkpoint['epoch']
	else:
		initialize_memorybank(network, train_loader, device, memory_bank)
	logging.info('start training')
	best_acc = 0.0
	for i_epoch in range(start_epoch, args.max_epoch):
		# memory_bank.cluster_points(args.n_clusters)
		train(i_epoch, network, criterion, optimizer, train_loader, device, memory_bank, first=(i_epoch <= start_epoch+30))

		save_name = 'checkpoint.pth'
		checkpoint = {
			'epoch': i_epoch + 1,
			'state_dict': network.state_dict(),
			'optimizer': optimizer.state_dict(),
			'memory_banks': memory_bank.points,
		}
		torch.save(checkpoint, os.path.join(args.exp, 'models', save_name))

		# scheduler.step()
		adjust_learning_rate(optimizer, i_epoch)
		# validate(network, memory_bank, val_loader, train_ordered_labels, device)
		acc = kNN(i_epoch, network, memory_bank, val_loader, train_ordered_labels, K=200, sigma=args.t)
		if acc >= best_acc:
			best_acc = acc
			torch.save(checkpoint, os.path.join(args.exp, 'models', 'best.pth'))
		if i_epoch in [120, 160, 200]:
			torch.save(checkpoint, os.path.join(args.exp, 'models', '{}.pth'.format(i_epoch+1)))

			
		logging.info(colorful('[Epoch: {}] val acc: {:.4f}'.format(i_epoch, acc)))
		logging.info(colorful('[Epoch: {}] best acc: {:.4f}'.format(i_epoch, best_acc)))
		writer.add_scalar('acc', acc, i_epoch+1)

		with torch.no_grad():
			for name, param in network.named_parameters():
				if 'bn' not in name:
					writer.add_histogram(name, param, i_epoch)

		# cluster
	


def initialize_memorybank(network, dataloader, device, memory_bank, refill=False):
	logging.info('start memorybank pointing filling')
	# if not args.entropy_loss:
	memory_bank.random_init_bank()
	logging.info('finish memorybank pointing filling')


def train(i_epoch, network, criterion, optimizer, dataloader, device, memory_bank, first=False):
	network.train()
	lossesA = AvgMeter()
	lossesB = AvgMeter()
	losses = AvgMeter()
	pbar = tqdm(dataloader)
	for data in pbar:
		img = data[1].to(device)
		# normal_img = img[:,0,:,:,:]
		index = data[0].to(device)
		output = network(img)
		output = output.to(device)

		lossA, lossB = criterion(output, index, memory_bank, first=first)
		if first: loss = lossA
		else: loss = lossA + lossB
		lossesA.add(lossA.item())
		if not first: lossesB.add(lossB.item())
		losses.add(loss.item())


		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		with torch.no_grad():
			memory_bank.update_points(output.detach(), index)

		lr = optimizer.param_groups[0]['lr']
		pbar.set_description("Epoch:{} [lr:{}]".format(i_epoch, lr))
		info = 'loss: {:.4f}(A {:.4f} + B {:.4f}), '.format(losses.get(), lossesA.get(), lossesB.get())
		pbar.set_postfix(info=info)

	writer.add_scalar('loss', losses.get(), i_epoch)
	writer.add_scalar('lossA', lossesA.get(), i_epoch)
	writer.add_scalar('lossB', lossesB.get(), i_epoch)
	logging.info('Epoch {}: loss: {:.4f}'.format(i_epoch, losses.get()))
	logging.info('Epoch {}: lossA: {:.4f}'.format(i_epoch, lossesA.get()))
	logging.info('Epoch {}: lossB: {:.4f}'.format(i_epoch, lossesB.get()))

def validate(network, memory_bank, val_loader, train_ordered_labels, device):
	# For validation, for each image, we find the closest neighbor in the
	# memory bank (training data), take its class! We compute the accuracy.

	network.eval()
	num_correct = 0.
	num_total = 0.

	with torch.no_grad():
		pbar = tqdm(val_loader)
		for _, images, labels in pbar:
			batch_size = images.size(0)

			# cast elements to CUDA
			images = images.to(device)
			outputs = network(images).to(device)

			# use memory bank to ge the top 1 neighbor
			# from the training dataset
			outputs = objective.l2_normalize(outputs)
			all_dps = memory_bank.get_all_dot_products(outputs)
			_, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
			neighbor_idxs = neighbor_idxs.squeeze(1)  # shape: batch_size
			neighbor_idxs = neighbor_idxs.cpu().numpy()  # convert to numpy
			# fetch the actual label of each example
			neighbor_labels = train_ordered_labels[neighbor_idxs]
			neighbor_labels = torch.from_numpy(neighbor_labels).long()

			num_correct += torch.sum(neighbor_labels == labels).item()
			num_total += batch_size

			pbar.set_postfix({"Val Accuracy": num_correct / num_total})

	return (num_correct/num_total)



def kNN(epoch, net, memory_bank, val_loader, train_ordered_labels, K=200, sigma=0.1):
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
	cmd = "python linear_classification.py --backbone {} --pretrained_path '{}' --exp '{}' --dataset {}".format(args.network, os.path.join(args.exp, 'models', 'save_195.pth'), os.path.join(args.exp, 'linear'), args.dataset)
	print(cmd)
	os.system(cmd)


