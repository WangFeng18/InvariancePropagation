from __future__ import division
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from dotmap import DotMap
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from datasets.cifar import CIFAR10Instance
from models.alexnet import AlexNet
from models.resnet_cifar import ResNet18
from torch.utils.data import DataLoader
from utils import *
import objective
import logging

def get_cluster_numbers(i_epoch):
	# 30w 29w 28w 27w 26w ... 21w, (2 epochs for each) 20
	# 20w, 19w, ... 11w (5 epochs for each) 50
	# 10w, 9w, ... 6w (10 epochs for each) 50
	# 5w, 4w, 3w, 2w, 1w (15 epochs for each) 75
	nmbs = np.zeros(201)
	nmbs[:20] = [30-(i//2) for i in range(20)]
	nmbs[20:70] = [20-(i//5) for i in range(50)]
	nmbs[70:120] = [10-(i//10) for i in range(50)]
	nmbs[120:195] = [5-(i//15) for i in range(75)]
	nmbs[195:] = 1
	nmbs = nmbs * 10000
	return 1000
	#return int(nmbs[i_epoch])

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default=256, type=int)
	parser.add_argument('--gpus', default='0', type=str)
	parser.add_argument('--network', default='resnet18', type=str)
	parser.add_argument('--exp', default=os.path.expanduser('/data/HierachicalAggregation_exp/cifar/cent+point/'), type=str)
	parser.add_argument('--pretrain_path', default='', type=str)
	parser.add_argument('--n_workers', default=2, type=int)
	global args
	args = parser.parse_args()
	if not os.path.exists(args.exp):
		os.makedirs(args.exp)
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

	# dataloader initialize
	train_transforms = transforms.Compose([
		transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
		transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
		transforms.RandomGrayscale(p=0.2),
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	val_transforms = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	train_dataset = CIFAR10Instance(root='./data', train=True, download=True, transform=train_transforms)
	val_dataset = CIFAR10Instance(root='./data', train=False, download=True, transform=val_transforms)
	train_ordered_labels = np.array(train_dataset.targets)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
						shuffle=True, pin_memory=False,
						num_workers=args.n_workers)	
	val_loader = DataLoader(val_dataset, batch_size=100,
						shuffle=False, pin_memory=False,
						num_workers=args.n_workers)


	# create model
	if args.network == 'alexnet':
		raise NotImplementedError
	elif args.network == 'resnet18':
		network = ResNet18(128)
	network = nn.DataParallel(network, device_ids=device_ids)
	network.to(device)

	cudnn.benchmark = True

	# create memory_bank
	memory_bank = objective.MemoryBank_v1(len(train_dataset), train_ordered_labels, None, device)

	if args.pretrain_path!= '' and args.pretrain_path!= 'none':
		logging.info('loading pretrained file from {}'.format(args.pretrain_path))
		state_dict = torch.load(args.pretrain_path)
		network.load_state_dict(state_dict)
	initialize_memorybank(network, train_loader, device, memory_bank)
	acc = kNN(0, network, memory_bank, val_loader, train_ordered_labels, K=200, sigma=0.07)
	
	#acc = validate(network, val_loader, memory_bank, device, train_ordered_labels)
	logging.info('[pretrain] val acc: {:.4f}'.format(acc))
	
def initialize_memorybank(network, dataloader, device, memory_bank):
	logging.info('start memorybank pointing filling')
	pbar = tqdm(dataloader)
	with torch.no_grad():
		for data in pbar:
			img = data[1].to(device)
			index = data[0]
			output = network(img).to(device)
			memory_bank.update_points(output.detach(), index)
	logging.info('finish memorybank pointing filling')

def validate(network, val_loader, memory_bank, device, train_ordered_labels):
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






