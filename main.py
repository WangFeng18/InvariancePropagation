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
from models.alexnet import AlexNet
from models.alexnet import AlexNet_cifar
from models.resnet_cifar import ResNet18 as ResNet18_cifar
from models.resnet_cifar import ResNet50 as ResNet50_cifar
from models.resnet import resnet18, resnet50
from models.preact_resnet import PreActResNet18
from models.preact_resnet import PreActResNet50
# from models.alexnet_cifar import AlexNet
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
	parser.add_argument('--max_epoch', default=200, type=int)
	parser.add_argument('--lr_decay_steps', default='160,190,200', type=str)
	parser.add_argument('--exp', default='', type=str)
	parser.add_argument('--res_path', default='', type=str)
	parser.add_argument('--resume_path', default='', type=str)
	parser.add_argument('--pretrain_path', default='', type=str)

	parser.add_argument('--dataset', default='imagenet', type=str)
	parser.add_argument('--lr', default=0.03, type=float)
	parser.add_argument('--lr_decay_rate', default=0.1, type=float)
	parser.add_argument('--batch_size', default=256, type=int)
	parser.add_argument('--weight_decay', default=5e-4, type=float)
	parser.add_argument('--n_workers', default=32, type=int)
	parser.add_argument('--n_background', default=4096, type=int)
	parser.add_argument('--t', default=0.07, type=float)
	parser.add_argument('--m', default=0.5, type=float)
	parser.add_argument('--dropout', action='store_true')
	parser.add_argument('--blur', action='store_true')

	parser.add_argument('--network', default='resnet18', type=str)
	parser.add_argument('--mix', action='store_true')
	parser.add_argument('--not_hardpos', action='store_true')
	parser.add_argument('--InvP', type=int, default=1)
	parser.add_argument('--ramp_up', default='binary', type=str)
	parser.add_argument('--lam_inv', default=0.6, type=float)
	parser.add_argument('--lam_mix', default=1.0, type=float)
	parser.add_argument('--diffusion_layer', default=4, type=int)
	# for cifar 10 the best diffusion_layer is 3 and cifar 100 is 4
	# for imagenet I have only tested when diffusion_layer = 3
	parser.add_argument('--K_nearst', default=4, type=int)
	parser.add_argument('--n_pos', default=50, type=int)
	# for cifar10 the best n_pos is 20, for cifar 100 the best is 10 or 20
	parser.add_argument('--exclusive', default=1, type=int)
	parser.add_argument('--exclusive_easypos', default=0, type=int)
	parser.add_argument('--nonlinearhead', default=0, type=int)
	# exclusive best to be 0

	global args
	args = parser.parse_args()
	if 'imagenet' in args.dataset:
		args.weight_decay = 1e-4
		# args.weight_decay = 1e-6
	else:
		args.weight_decay = 5e-4
	exp_identifier = get_expidentifier(['mix', 'network', 'lam_inv', 'lam_mix', 'diffusion_layer', 'K_nearst', 'n_pos', 'exclusive', 'max_epoch', 'ramp_up', 'nonlinearhead', 't', 'weight_decay'], args)
	if not args.InvP: exp_identifier = 'hard'
	args.exp = os.path.join(args.exp, exp_identifier)

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

	# create model
	if args.network == 'alexnet':
		network = alexnet(128)
	if args.network == 'alexnet_cifar':
		network = AlexNet_cifar(128)
	elif args.network == 'resnet18_cifar':
		network = ResNet18_cifar(128, dropout=args.dropout, non_linear_head=args.nonlinearhead)
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

	cudnn.benchmark = True

	# create memory_bank
	global writer
	writer = SummaryWriter(comment='InvariancePropagation', logdir=os.path.join(args.exp, 'runs'))
	memory_bank = objective.MemoryBank_v1(len(train_dataset), train_ordered_labels, writer, device, m=args.m)

	# create criterion
	criterionA = objective.InvariancePropagationLoss(args.t, diffusion_layer=args.diffusion_layer, k=args.K_nearst, n_pos=args.n_pos, exclusive=args.exclusive, exclusive_easypos=args.exclusive_easypos, InvP=args.InvP, hard_pos=(not args.not_hardpos))
	criterionB = objective.MixPointLoss(args.t)
	if args.ramp_up == 'binary':
		ramp_up = lambda i_epoch: objective.BinaryRampUp(i_epoch, 30)
	elif args.ramp_up == 'gaussian':
		ramp_up = lambda i_epoch: objective.GaussianRampUp(i_epoch, 30, 5)
	elif args.ramp_up == 'zero':
		ramp_up = lambda i_epoch: 1

	logging.info(beautify(args))
	start_epoch = 0
	if args.pretrain_path!= '' and args.pretrain_path!= 'none':
		logging.info('loading pretrained file from {}'.format(args.pretrain_path))
		checkpoint = torch.load(args.pretrain_path)
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
			# memory_bank.cluster_points(args.n_clusters)
			adjust_learning_rate(args.lr_decay_steps, optimizer, i_epoch, lr_decay_rate=args.lr_decay_rate)
			train(i_epoch, network, criterionA, criterionB, optimizer, train_loader, device, memory_bank, ramp_up)

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
			acc = kNN(i_epoch, network, memory_bank, val_loader, train_ordered_labels, K=200, sigma=0.07)
			if acc >= best_acc:
				best_acc = acc
				torch.save(checkpoint, os.path.join(args.exp, 'models', 'best.pth'))
			if i_epoch in [30, 60, 120, 160, 200, 400, 600]:
				torch.save(checkpoint, os.path.join(args.exp, 'models', '{}.pth'.format(i_epoch+1)))

			args.y_best_acc = best_acc
			logging.info(colorful('[Epoch: {}] val acc: {:.4f}'.format(i_epoch, acc)))
			logging.info(colorful('[Epoch: {}] best acc: {:.4f}'.format(i_epoch, best_acc)))
			writer.add_scalar('acc', acc, i_epoch+1)

			with torch.no_grad():
				for name, param in network.named_parameters():
					if 'bn' not in name:
						writer.add_histogram(name, param, i_epoch)

			# cluster
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


def train(i_epoch, network, criterionA, criterionB, optimizer, dataloader, device, memory_bank, ramp_up):
	#all_targets = np.array(dataloader.dataset.targets)
	network.train()
	losses_ins = AvgMeter()
	losses_inv = AvgMeter()
	losses_mix = AvgMeter()
	losses = AvgMeter()
	beta_dist = beta.Beta(0.75,0.75)
	all_weights = []
	n_neighbour = AvgMeter()
	pbar = tqdm(dataloader)

	ipacc = AverageMeter()
	nnacc = AverageMeter()
	for data in pbar:
		img = data[1].to(device)
		# normal_img = img[:,0,:,:,:]
		index = data[0].to(device)
		output = network(img).to(device)

		# Nearst Neighbour Set vs Invariance Propagation Set
		L_ins, L_inv, NNS, IPS = criterionA(output, index, memory_bank)
		# lossA = lossA_1 + args.lam_inv * lossA_2
		if np.random.rand() < 0.0:
			# NNS BSx4096 IPS BSxK index 
			for i_sample in range(NNS.size(0)):
				right_target = all_targets[index[i_sample]]
				this_ips = np.unique(IPS[i_sample].detach().cpu().numpy())
				iptargets = all_targets[this_ips]
				ip_consistency = (iptargets == right_target).sum()/float(len(this_ips))
				ipacc.update(ip_consistency, len(this_ips))

				this_nns = NNS[i_sample].detach().cpu().numpy()[:len(this_ips)+1]
				nntargets = all_targets[this_nns]
				nn_consistency = ((nntargets == right_target).sum()-1)/float(len(this_ips))
				nnacc.update(nn_consistency, len(this_ips))

		if args.mix:
			permutations = np.arange(index.size(0))
			np.random.shuffle(permutations)
			imgB = img[permutations]
			indexB = index[permutations]
			Alphas = beta_dist.sample([index.size(0),]).to(device)
			MixImgs = img * Alphas.view(-1,1,1,1) + imgB * (1-Alphas).view(-1,1,1,1)
			outputMix = network(MixImgs)

			L_mix = criterionB(outputMix, Alphas, index, indexB, memory_bank)
		else:
			L_mix = 0.0

		L = L_ins + args.lam_inv * ramp_up(i_epoch) * L_inv + args.lam_mix * L_mix
		losses_ins.add(L_ins.item())
		losses_inv.add(0.0 if type(L_inv)==float else L_inv.item())
		losses_mix.add(0.0 if type(L_mix)==float else L_mix.item())
		losses.add(L.item())

		optimizer.zero_grad()
		L.backward()
		optimizer.step()

		with torch.no_grad():
			memory_bank.update_points(output.detach(), index)

		lr = optimizer.param_groups[0]['lr']
		pbar.set_description("Epoch:{} [lr:{}] {:.3f}__{:.3f}".format(i_epoch, lr, ipacc.avg, nnacc.avg))
		info = 'L: {:.4f} = L_ins: {:.4f} + {:.3f} * L_inv: {:.4f} + {:.3f} * L_mix: {:.4f}'.format(losses.get(), losses_ins.get(), args.lam_inv * ramp_up(i_epoch), losses_inv.get(), args.lam_mix, losses_mix.get())
		pbar.set_postfix(info=info)

	writer.add_scalar('L', losses.get(), i_epoch)
	writer.add_scalar('L_ins', losses_ins.get(), i_epoch)
	writer.add_scalar('L_inv', losses_inv.get(), i_epoch)
	writer.add_scalar('L_mix', losses_mix.get(), i_epoch)
	writer.add_scalar('ipacc', ipacc.avg, i_epoch)
	writer.add_scalar('nnacc', nnacc.avg, i_epoch)
	logging.info('Epoch {}: L: {:.4f}'.format(i_epoch, losses.get()))
	logging.info('Epoch {}: L_ins: {:.4f}'.format(i_epoch, losses_ins.get()))
	logging.info('Epoch {}: L_inv: {:.4f}'.format(i_epoch, losses_inv.get()))
	logging.info('Epoch {}: L_mix: {:.4f}'.format(i_epoch, losses_mix.get()))
	logging.info('Epoch {}: IPacc: {:.4f}'.format(i_epoch, ipacc.avg))
	logging.info('Epoch {}: NNacc: {:.4f}'.format(i_epoch, nnacc.avg))

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

def run_eval_linear(args):
	lr = 30.0 if args.network.startswith('imagenet') else 1.0
	cmd = "python -m downstream.eval_linear --learning_rate {} --model {} --save_folder '{}' --model_path '{}' --dataset {} --gpu 0".format(lr, args.network, os.path.join(args.exp, 'linear'), os.path.join(args.exp, 'models', 'best.pth'), args.dataset)
	logging.info(cmd)
	os.system(cmd)

def run_semi_supervised(args):
	semi_fraction = [250, 500, 1000, 2000, 4000]
	for fraction in semi_fraction:
		cmd = "python -m downstream.semi_supervised --dataset {} --gpus 0 --exp '{}' --list '{}' --pretrain_path '{}' --network {}".format(args.dataset, os.path.join(args.exp, 'semi_{}'.format(fraction)), 'datasets/lists/cifar_{}.txt'.format(fraction), os.path.join(args.exp, 'models', 'best.pth'), args.network)
		logging.info(cmd)
		os.system(cmd)


if __name__ == '__main__':
	main()
	run_eval_linear(args)
	#run_semi_supervised(args)