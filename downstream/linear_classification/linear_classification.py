import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# import data.Transforms as transforms
from torchvision.datasets import ImageFolder
import argparse
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
import torch.backends.cudnn as cudnn
from models.LinearModelMulti import LinearModel

from datasets.ImageList import ImageList
from utils import AvgMeter, AccuracyMeter
from models.resnet import resnet50
from PIL import ImageFile
import csv
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Logger(object):

	def __init__(self, path, header):
		self.log_file = open(path, 'w')
		self.logger = csv.writer(self.log_file, delimiter='\t')

		self.logger.writerow(header)
		self.header = header

	def __del(self):
		self.log_file.close()

	def log(self, values):
		write_values = []
		for col in self.header:
			assert col in values
			write_values.append(values[col])

		self.logger.writerow(write_values)
		self.log_file.flush()

class LinearNetwork(nn.Module):
	def __init__(self, pretrained_path, backbone, pool_type):
		super(LinearNetwork, self).__init__()
		if backbone == 'resnet50':
			self.features = resnet50()
		else:
			raise NotImplementedError("ERROR: network not implemented!")

		for param in self.features.parameters():
			param.requires_grad = False

		self.linear = nn.ModuleList()

		if 'resnet50' in backbone:
			# config = [(12, 64, pool_type), (6, 256, pool_type), (6, 256, pool_type), (6, 256, pool_type), (4, 512, pool_type), (4, 512, pool_type), (4, 512, pool_type), (4, 512, pool_type), (3, 1024, pool_type), (3, 1024, pool_type), (3, 1024, pool_type), (3, 1024, pool_type), (3, 1024, pool_type), (3, 1024, pool_type), (2, 2048, pool_type), (2, 2048, pool_type), (2, 2048, pool_type)]
			config = [(12, 64, pool_type), (6, 256, pool_type), (6, 256, pool_type), (6, 256, pool_type), (4, 512, pool_type), (4, 512, pool_type), (4, 512, pool_type), (4, 512, pool_type), (3, 1024, pool_type), (3, 1024, pool_type), (3, 1024, pool_type), (3, 1024, pool_type), (3, 1024, pool_type), (3, 1024, pool_type), (1, 2048, pool_type), (1, 2048, pool_type), (1, 2048, pool_type)]
		else:
			raise NotImplementedError("ERROR: network not implemented!")

		for i_head in range(len(config)):
			self.linear.append(LinearModel(*config[i_head]))

		self.pretrained_path = pretrained_path
		if pretrained_path != '' and pretrained_path != 'none':
			print('LOADING pretrained model')
			self._init_features()

	def forward(self, x):
		feats = self.features.forward_convs(x)
		outputs = []
		for idx, feat in enumerate(feats):
			outputs.append(self.linear[idx](feat))
		return outputs

	def _init_features(self):
		state_dict = torch.load(self.pretrained_path)
		if 'state_dict' in state_dict.keys():
			state_dict = state_dict['state_dict']

		tstate_dict = self.features.state_dict()
		valid_keys = self.features.state_dict().keys()
		for key, value in state_dict.items():
			if 'module.' in key:
				tkey = key.replace('module.', '')
				if tkey in valid_keys:
					print(tkey)
					tstate_dict[tkey] = value
			else:
				if key in valid_keys:
					print(key)
					tstate_dict[key] = value

		self.features.load_state_dict(tstate_dict)
		self.features.eval()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--stage', default='train', type=str)
	parser.add_argument('--root', default=os.path.expanduser('~/ILSVRC2012/'), type=str)
	parser.add_argument('--lr', default=1., type=float)
	parser.add_argument('--batch_size', default=256, type=int)
	parser.add_argument('--gpus', default='0', type=str)
	parser.add_argument('--weight_decay', default=0, type=float)
	parser.add_argument('--max_epoch', default=120, type=int)
	parser.add_argument('--exp', default=os.path.expanduser('~/ClusterFusion_exp/rotation/'), type=str)
	parser.add_argument('--pretrained_path', default='', type=str)
	parser.add_argument('--resume_path', default='', type=str)
	parser.add_argument('--backbone', default='resnet50', type=str)
	parser.add_argument('--tencrop', default=0, type=int)
	parser.add_argument('--pool_type', default='mean', type=str)
	parser.add_argument('--nesterov', default=0, type=int)
	parser.add_argument('--dataset', default='imagenet', type=str)
	global args
	global n_layers
	args = parser.parse_args()
	args.exp = os.path.expanduser(args.exp)
	if 'resnet50' in args.backbone:
		n_layers = 17
	if not os.path.exists(args.exp):
		os.makedirs(args.exp)
	if args.stage == 'train':
		logger = Logger(os.path.join(args.exp, 'record.log'), ['epoch', 'lr', 'loss/acc[1]', 'loss/acc[2]', 'loss/acc[3]', 'loss/acc[4]', 'loss/acc[5]'])

	if args.dataset == 'places205':
		args.root = '/data/data/vision/torralba/deeplearning/images256'

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	normalize = torchvision.transforms.Normalize(mean=mean,std=std)

	t = torchvision.transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
			transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor(),
			normalize
		])
	if args.tencrop:
		t_test = torchvision.transforms.Compose([
			transforms.TenCrop(224),
			transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
		])
	else:
		t_test = torchvision.transforms.Compose([
					transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					normalize
				])
	if args.dataset != 'places205':
		train_dataset = ImageFolder(os.path.join(args.root, '{}'.format('train')), transform=t)
		val_dataset = ImageFolder(os.path.join(args.root, '{}'.format('val')), transform=t_test)
	else:
		train_dataset = ImageList('/data/trainvalsplit_places205/train_places205.csv', args.root, transform=t, symbol_split=' ')
		val_dataset = ImageList('/data/trainvalsplit_places205/val_places205.csv', args.root, transform=t_test, symbol_split=' ')

	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32, shuffle=True)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=32, shuffle=True)
	cudnn.benchmark = True
	network = LinearNetwork(args.pretrained_path, args.backbone, args.pool_type)
	device_ids = list(map(lambda x: int(x), args.gpus.split(',')))
	device = torch.device('cuda: 0')
	network = nn.DataParallel(network, device_ids=device_ids)
	network.to(device)
	criterion = nn.CrossEntropyLoss().to(device)
	
	params = []
	for i in range(n_layers):
		params.append({'params':network.module.linear[i].parameters(), 'lr':args.lr})
	optimizer = torch.optim.SGD(
		params,
		momentum=0.9,
		nesterov=args.nesterov,
		weight_decay=args.weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80, 100], gamma=0.1)
	if args.stage == 'train':
		if args.resume_path != '' and args.resume_path != 'none':
			state_dict = torch.load(args.resume_path)
			network.load_state_dict(state_dict)
		writer = SummaryWriter()
		for i_epoch in range(args.max_epoch):
			val_accs = eval(network, val_dataloader, device, args.tencrop)
			epoch_losses, epoch_accs = train(i_epoch, network, criterion, optimizer, train_dataloader, device)
			log_dic = {
				'epoch': 'train {}:'.format(i_epoch),
				'lr'   : optimizer.param_groups[0]['lr'],
			}
			acc_dic = {
				'epoch': 'eval  {}:'.format(i_epoch),
				'lr'   : optimizer.param_groups[0]['lr'],
			}
			for idx in range(13, n_layers):
				writer.add_scalar('loss_{}'.format(idx), epoch_losses[idx].get(), i_epoch)
				log_dic.update({'loss/acc[{}]'.format(idx+1): '{:.3f}/{:.3f}'.format(epoch_losses[idx].get(), epoch_accs[idx].get())})
				acc_dic.update({'loss/acc[{}]'.format(idx+1): '{:.3f}'.format(val_accs[idx].get())})

			save_name = 'save_{}.pth'.format(int(i_epoch/5)*5)
			torch.save(network.state_dict(), os.path.join(args.exp, save_name))
			scheduler.step()
	else:
		state_dict = torch.load(args.resume_path)
		network.load_state_dict(state_dict)
		with torch.no_grad():
			network.eval()
			eval(network, val_dataloader, device, args.tencrop)

def accuracy(output, label, accmeter):
	judge = output.argmax(dim=1)
	correct = (judge == label).sum().item()
	total = output.size()[0]
	accmeter.add(correct, total)

def generate_lossacc(num, start=0):
	ts = ''
	for i in range(start, num):
		s = 'c{}:'.format(i+1)+'{:.4f}/{:.4f}, '
		ts += s
	ts += 'lr: {}'
	return ts

def generate_acc(num, start=0):
	ts = ''
	for i in range(start, num):
		s = 'c{}:'.format(i+1)+'{:.4f}, '
		ts += s
	return ts

def train(i_epoch, network, criterion, optimizer, dataloader, device):
	network.eval()
	losses = []
	accs = []
	for idx in range(n_layers):
		losses.append(AvgMeter())
		accs.append(AccuracyMeter())
	pbar = tqdm(dataloader)
	for data in pbar:
		img = data[0].to(device)
		rot = data[1].long().to(device)

		outputs = network(img)
		for idx in range(n_layers):
			outputs[idx].to(device)

		optimizer.zero_grad()
		all_loss = []
		for idx in range(n_layers):
			all_loss.append(criterion(outputs[idx], rot))
			accuracy(outputs[idx], rot, accs[idx])

		loss = 0
		for idx in range(n_layers):
			loss += all_loss[idx]
			#all_loss[idx].backward()
			losses[idx].add(all_loss[idx].item())
		loss.backward()
		optimizer.step()

		
		lr = optimizer.param_groups[0]['lr']
		str_content = generate_lossacc(n_layers, start=13)
		# str_content = 'c1:{:.4f}/{:.4f} c2:{:.4f}/{:.4f} c3:{:.4f}/{:.4f} c4:{:.4f}/{:.4f} c5:{:.4f}/{:.4f}, lr:{}'
		flt_content = []
		for idx in range(13, n_layers):
			flt_content.append(losses[idx].get())
			flt_content.append(accs[idx].get())
		flt_content.append(lr)
		pbar.set_description("Epoch:{}".format(i_epoch))
		pbar.set_postfix(info=str_content.format(*flt_content))

	return losses, accs

def eval(network, dataloader, device, tencrop):
	network.eval()
	softmax = nn.Softmax(dim=1).cuda()
	accs = []
	for idx in range(n_layers):
		accs.append(AccuracyMeter())

	pbar = tqdm(dataloader)
	for data in pbar:
		img = data[0].to(device)
		rot = data[1].long().to(device)
		if tencrop:
			bs, ncrops, c, h, w = img.size()
			img = img.view(-1, c, h, w)

		outputs = network(img)

		for idx in range(n_layers):
			outputs[idx].to(device)
			if tencrop:
				outputs[idx] = softmax(outputs[idx])
				outputs[idx] = torch.squeeze(outputs[idx].view(bs, ncrops, -1).mean(1))

		for idx in range(13, n_layers):
			accuracy(outputs[idx], rot, accs[idx])

		str_content = generate_acc(n_layers, start=13)
		flt_content = []
		for idx in range(13, n_layers):
			flt_content.append(accs[idx].get())

		pbar.set_postfix(info=str_content.format(*flt_content))
	return accs


if __name__ == '__main__':
	main()






