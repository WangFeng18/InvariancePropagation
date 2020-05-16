import os
import torch
import torch.backends.cudnn as cudnn
import datasets.cifar as cifar
import datasets.stanford_cars as cars
import datasets.caltech101 as caltech101
import datasets.cub200 as cub200
import datasets.pets as pets
import datasets.flowers as flowers
import logging
import torch.nn as nn
import torchvision
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from models.resnet import resnet18, resnet50
from utils import *
import logging

class MeanPerClassAccuracy(object):
	def __init__(self, n_cls):
		self.n_cls = n_cls
		self.total = np.zeros(n_cls)
		self.correct = np.zeros(n_cls)

	def add(self, pred, target):
		true_judge = (pred == target)
		for t in target:
			self.total[t] += 1
		for i, t in enumerate(pred):
			if true_judge[i]:
				self.correct[t] += 1
		
	def get(self):
		pca = np.zeros(self.n_cls)
		for i in range(self.total.shape[0]):
			if self.total[i] != 0:
				a = self.correct[i] / self.total[i]
				pca[i] = a
		return pca.mean()
			

def adjust_learning_rate(lr_decay_steps, optimizer, epoch, base_lr):
	"""Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
	#steps = [120,160,200]
	steps = list(map(int, lr_decay_steps.split(',')))
	logging.info(steps)
	lr = base_lr
	if epoch < steps[0]:
		lr = base_lr
	elif epoch >= steps[0] and epoch < steps[1]:
		lr = base_lr * 0.1
	elif epoch >=steps[1] and epoch < steps[2]:
		lr = base_lr * 0.01
	else:
		lr = base_lr * 0.001
	#lr = args.lr * (0.1 ** (epoch // 100))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

class FineTune(object):
	def __init__(self, args):
		cudnn.benchmark = True
		self.start_epoch = 0
		self.args = args
		self.create_experiment()
		self.criterion = nn.CrossEntropyLoss()
		self.get_dataloader()
		self.get_network()
		self.get_logger()
		self.get_optimizer()
		if self.args.resume_path != '':
			self.resume()
		logging.info(beautify(args))
		self.main_loop()

	def main_loop(self):
		best_acc = 0.0
		best_mpca = 0.0
		try:
			for i_epoch in range(self.start_epoch, self.args.max_epoch):
				# memory_bank.cluster_points(args.n_clusters)
				self.train(i_epoch)

				save_name = 'checkpoint.pth'
				checkpoint = {
					'epoch': i_epoch + 1,
					'state_dict': self.network.state_dict(),
					'optimizer': self.optimizer.state_dict(),
					'cls_state_dict': self.classifier.state_dict(),
					'cls_optimizer': self.cls_optimizer.state_dict(),
				}
				torch.save(checkpoint, os.path.join(self.args.exp, 'models', save_name))

				# scheduler.step()
				adjust_learning_rate(self.args.lr_decay_steps, self.cls_optimizer, i_epoch, self.args.lr*5)
				adjust_learning_rate(self.args.lr_decay_steps, self.optimizer, i_epoch, self.args.lr)
				if i_epoch in [30, 60, 120, 160, 200, 400, 600]:
					torch.save(checkpoint, os.path.join(self.args.exp, 'models', '{}.pth'.format(i_epoch+1)))

				if i_epoch % self.args.n_val == 0:
					acc, mpca = self.validate(i_epoch)
					if acc >= best_acc:
						best_acc = acc
						torch.save(checkpoint, os.path.join(self.args.exp, 'models', 'best.pth'))
					if mpca >= best_mpca:
						best_mpca = mpca

					self.args.y_best_acc = best_acc
					logging.info(colorful('[Epoch: {}] best acc: {:.4f}'.format(i_epoch, best_acc)))
					logging.info(colorful('[Epoch: {}] best mpca: {:.4f}'.format(i_epoch, best_mpca)))
					self.writer.add_scalar('acc', acc, i_epoch+1)

				# with torch.no_grad():
					# for name, param in self.network.named_parameters():
						# if 'bn' not in name:
							# self.writer.add_histogram(name, param, i_epoch)

				# cluster
		except KeyboardInterrupt as e:
			logging.info('KeyboardInterrupt at {} Epochs'.format(i_epoch))
			exit()

	def create_experiment(self):
		if not os.path.exists(self.args.exp):
			os.makedirs(self.args.exp)
		if not os.path.exists(os.path.join(self.args.exp, 'models')):
			os.makedirs(os.path.join(self.args.exp, 'models'))
		self.device_ids = list(map(lambda x: int(x), self.args.gpus.split(',')))
		self.device = torch.device('cuda: 0')

	def resume(self):
		logging.info('resuming from {}'.format(self.args.resume_path))
		checkpoint = torch.load(self.args.resume_path)
		self.network.load_state_dict(checkpoint['state_dict'])
		self.classifier.load_state_dict(checkpoint['cls_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		self.cls_optimizer.load_state_dict(checkpoint['cls_optimizer'])
		self.start_epoch = checkpoint['epoch']

	def get_dataloader(self):
		if self.args.dataset == 'car':
			self.train_loader, self.val_loader = cars.get_finetune_dataloader(self.args)
			self.args.n_cls = 196
		elif self.args.dataset == 'caltech101':
			self.train_loader, self.val_loader = caltech101.get_finetune_dataloader(self.args)
			self.args.n_cls = 101
		elif self.args.dataset == 'cub200':
			self.train_loader, self.val_loader = cub200.get_finetune_dataloader(self.args)
			self.args.n_cls = 200
		elif self.args.dataset == 'pets':
			self.train_loader, self.val_loader = pets.get_finetune_dataloader(self.args)
			self.args.n_cls = 37
		elif self.args.dataset == 'flowers':
			self.train_loader, self.val_loader = flowers.get_finetune_dataloader(self.args)
			self.args.n_cls = 102
		elif self.args.dataset == 'cifar10' or self.args.dataset == 'cifar100':
			self.train_loader, self.val_loader = cifar.get_finetune_dataloader(self.args)
			self.args.n_cls = 10 if self.args.dataset=='cifar10' else 100
		else:
			raise NotImplementedError

	def get_network(self):
		if self.args.network == 'resnet50':
			self.network = resnet50()
			self.classifier = nn.Linear(2048, args.n_cls).to(self.device)
		elif self.args.network == 'resnet18':
			self.network = resnet18()
			self.classifier = nn.Linear(512, args.n_cls).to(self.device)
		else:
			raise NotImplementedError
		self.network = nn.DataParallel(self.network, device_ids=self.device_ids)
		self.network.to(self.device)
		ckpt = torch.load(self.args.pretrain_path)
		# self.network.load_state_dict(ckpt['state_dict'])
		state_dict = ckpt['state_dict']
		valid_state_dict = {k: v for k, v in state_dict.items() if k in self.network.state_dict() and 'fc.' not in k}
		for k,v in self.network.state_dict().items():
			if k not in valid_state_dict:
				logging.info('{}: Random Init'.format(k))
				valid_state_dict[k] = v
		# logging.info(valid_state_dict.keys())
		self.network.load_state_dict(valid_state_dict)

		if self.args.fc_only: 
			for param in self.network.parameters():
				param.requires_grad = False

	def get_logger(self):
		if not os.path.exists(os.path.join(self.args.exp, 'runs')):
			os.makedirs(os.path.join(self.args.exp, 'runs'))
		if not os.path.exists(os.path.join(self.args.exp, 'logs')):
			os.makedirs(os.path.join(self.args.exp, 'logs'))
		self.writer = SummaryWriter(comment='FineTune', logdir=os.path.join(self.args.exp, 'runs'))
		self.logger = getLogger(self.args.exp)

	def get_optimizer(self):
		self.optimizer = torch.optim.SGD(
			filter(lambda x: x.requires_grad, self.network.parameters()),
			lr=args.lr,
			momentum=0.9,
			weight_decay=args.weight_decay,
		)
		self.cls_optimizer = torch.optim.SGD(
			self.classifier.parameters(),
			lr = 5 * args.lr,
			momentum=0.9,
			weight_decay=args.weight_decay,
		)

	def train(self, i_epoch):
		# TODO Investigate if the bn requires training
		self.network.train()
		self.classifier.train()
		losses = AvgMeter()
		pbar = tqdm(self.train_loader)
		for data in pbar:
			img = data[0].float().to(self.device)
			target = data[1].long().to(self.device)
			output = self.network(img, 6)
			output = self.classifier(output)
			loss = self.criterion(output, target)

			self.optimizer.zero_grad()
			self.cls_optimizer.zero_grad()
			loss.backward()
			if not self.args.fc_only: self.optimizer.step()
			self.cls_optimizer.step()
			losses.add(loss.item())
			lr = self.cls_optimizer.param_groups[0]['lr']
			pbar.set_description('Epoch: {}, lr: {}'.format(i_epoch, lr))
			pbar.set_postfix(info='loss: {:.4f}'.format(losses.get()))
		self.writer.add_scalar('loss', losses.get(), i_epoch)
		logging.info('Epoch: {}, loss: {}'.format(i_epoch, losses.get()))

	def validate(self, i_epoch):
		with torch.no_grad():
			self.network.eval()
			self.classifier.eval()
			losses = AvgMeter()
			mpca = MeanPerClassAccuracy(self.args.n_cls)
			num_correct = 0.
			num_total = 0.
			pbar = tqdm(self.val_loader)
			for data in pbar:
				img = data[0].float().to(self.device)
				target = data[1].long().to(self.device)
				output = self.network(img, 6)
				output = self.classifier(output)
				loss = self.criterion(output, target)
				losses.add(loss.item())
				correct = (output.argmax(dim=1) == target).sum()
				total = output.size(0)
				num_correct += correct
				num_total += total
				mpca.add(output.argmax(dim=1).detach().cpu().numpy(), target.detach().cpu().numpy())
				# print('{}/{}'.format(num_correct, num_total))
				acc = float(num_correct)/float(num_total)
				pbar.set_description('Epoch: {}'.format(i_epoch))
				pbar.set_postfix(info='loss: {:.4f}, acc: {:.4f}, mpca: {:.4f}'.format(losses.get(), acc, mpca.get()))
			self.writer.add_scalar('acc', i_epoch, acc)
			logging.info('Epoch: {}, val_loss: {:.3f}, val_acc: {}, mpca: {:.4f}'.format(i_epoch, losses.get(), acc, mpca.get()))
		return acc, mpca.get()
			
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='car', type=str)
	parser.add_argument('--gpus', default='0,1', type=str)
	parser.add_argument('--max_epoch', default=200, type=int)
	parser.add_argument('--lr_decay_steps', default='160,190,200', type=str)
	parser.add_argument('--exp', default='', type=str)
	parser.add_argument('--resume_path', default='', type=str)
	parser.add_argument('--pretrain_path', default='', type=str)
	parser.add_argument('--n_cls', default=196, type=int)
	parser.add_argument('--n_val', default=10, type=int)

	parser.add_argument('--lr', default=0.01, type=float)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--weight_decay', default=1e-6, type=float)
	parser.add_argument('--n_workers', default=32, type=int)

	parser.add_argument('--network', default='resnet50', type=str)
	parser.add_argument('--fc_only', action='store_true')
	args = parser.parse_args()

	machine = FineTune(args)

