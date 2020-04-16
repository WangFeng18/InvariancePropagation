from __future__ import division
import torch
import numpy as np
import os
import math
import argparse
import logging
from collections import OrderedDict
import pandas as pd
import json


'''
Histogram of simalarities:
a) positive
b) Top-k percent
'''
def histogram(sim, top_k_percents, writer, i_epoch, name):
	K = np.array([int(sim.size(0) * top_k_percent) for top_k_percent in top_k_percents])
	max_K = K.max()
	y, yi = torch.topk(sim, int(max_K), largest=True, sorted=True)
	try:
		for i, top_k_percent in enumerate(top_k_percents):
			writer.add_histogram(
				'{}/top_{}'.format(name, int(top_k_percent*100)),
				y[:int(K[i])],
				i_epoch,
			)
	except:
		print('histogram wrong')
	

'''
The 100 ImageNet classes are copied from https://github.com/HobbitLong/CMC/issues/21
'''

imagenet_cls_100 = \
'n02869837 n01749939 n02488291 n02107142 n13037406 n02091831 n04517823 n04589890 n03062245 n01773797 n01735189 n07831146 n07753275 n03085013 n04485082 n02105505 n01983481 n02788148 n03530642 n04435653 n02086910 n02859443 n13040303 n03594734 n02085620 n02099849 n01558993 n04493381 n02109047 n04111531 n02877765 n04429376 n02009229 n01978455 n02106550 n01820546 n01692333 n07714571 n02974003 n02114855 n03785016 n03764736 n03775546 n02087046 n07836838 n04099969 n04592741 n03891251 n02701002 n03379051 n02259212 n07715103 n03947888 n04026417 n02326432 n03637318 n01980166 n02113799 n02086240 n03903868 n02483362 n04127249 n02089973 n03017168 n02093428 n02804414 n02396427 n04418357 n02172182 n01729322 n02113978 n03787032 n02089867 n02119022 n03777754 n04238763 n02231487 n03032252 n02138441 n02104029 n03837869 n03494278 n04136333 n03794056 n03492542 n02018207 n04067472 n03930630 n03584829 n02123045 n04229816 n02100583 n03642806 n04336792 n03259280 n02116738 n02108089 n03424325 n01855672 n02090622'.split(' ')

class data_prefetcher():
	def __init__(self, loader):
		self.loader = iter(loader)
		self.stream = torch.cuda.Stream()
		# With Amp, it isn't necessary to manually convert data to half.
		# if args.fp16:
		#     self.mean = self.mean.half()
		#     self.std = self.std.half()
		self.preload()

	def preload(self):
		try:
			self.next_data = next(self.loader)
		except StopIteration:
			self.next_data = None
			return
		with torch.cuda.stream(self.stream):
			for i in range(2):
				self.next_data[i] = self.next_data[i].cuda(non_blocking=True)
			# With Amp, it isn't necessary to manually convert data to half.
			# if args.fp16:
			#     self.next_input = self.next_input.half()
			# else:
			
	def next(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		data = self.next_data
		self.preload()
		return data

def batch_addscalar(writer, allloss, lossname, i):
	for loss, lossname in zip(allloss, lossname):
		writer.add_scalar(lossname, loss, i)

def batch_logging(allloss, lossname, i):
	for loss, lossname in zip(allloss, lossname):
		logging.info('[Epoch: {}] {}: {:.4f}'.format(i, lossname, loss))

def colorful(text):
	return '\033[1;33m {} \033[0m'.format(text)

def exclude_bn_weight_bias_from_weight_decay(model, weight_decay):
	decay = []
	no_decay = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		# if len(param.shape) == 1 or name in skip_list:
		if 'bn' in name:
			no_decay.append(param)
		else:
			decay.append(param)
	return [
		{'params': no_decay, 'weight_decay': 0.},
		{'params': decay, 'weight_decay': weight_decay}
	]

class GroupAvgMeter(object):
	def __init__(self, n, name=''):
		self.n = n
		self.group_names = [name + '_' + str(i) for i in range(n)]
		self.avg_meters = {}
		for i in range(len(self.group_names)):
			self.avg_meters.update({self.group_names[i]: AvgMeter()})

	def add(self, values):
		for value, group_name in zip(values, self.group_names):
			self.avg_meters[group_name].add(value)

	def get(self, group_name):
		return self.avg_meters[group_name].get()

	def get_all(self):
		return [self.avg_meters[group_name].get() for group_name in self.group_names]

	def s(self):
		return ','.join(['{:.4f}'.format(value) for value in self.get_all()])

class AvgMeter(object):
	def __init__(self):
		self.clear()

	def add(self, value):
		self.value += value
		self.n += 1

	def get(self):
		if self.n == 0:
			return 0
		return self.value/self.n

	def clear(self):
		self.n = 0
		self.value = 0.

class AccuracyMeter(object):
	def __init__(self):
		self.clear()

	def add(self, correct, total):
		self.correct += correct
		self.total += total

	def get(self):
		return self.correct/self.total
		
	def clear(self):
		self.correct = 0.
		self.total = 0.

def getLogger(path):
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	fh = logging.FileHandler(os.path.join(path, 'logs', 'log.txt'))
	fh.setLevel(logging.INFO)
	fh.setFormatter(formatter)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	logger.addHandler(fh)
	return logger

def beautify(dic):
	if type(dic) == argparse.Namespace:
		dic = vars(dic)
	return json.dumps(dic, indent=4, sort_keys=True)

def get_expidentifier(keys, args):
	args = vars(args)
	all_pairs = []
	for key in keys: 
		all_pairs.append(key)
		all_pairs.append(args[key])
	print(all_pairs)
	ret = ('[{}={}]'*len(keys)).format(*all_pairs)
	return ret

def save_result(args):
	res_path = args.res_path
	args = sorted(vars(args).items(), key=lambda obj: obj[0])
	a = OrderedDict()
	for key, value in args:
		a[key] = [value,]
	df = pd.DataFrame(a)
	df.to_csv(res_path, mode='a')

	
	
	
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
		self.avg = self.sum / float(self.count)



def adjust_learning_rate(lr, lr_decay_steps, optimizer, epoch, lr_decay_rate=0.1, cos=False, max_epoch=800):
	"""Decay the learning rate based on schedule"""
	if cos:
		lr *= 0.5 * (1. + math.cos(math.pi * epoch / max_epoch))
	else:
		steps = list(map(int, lr_decay_steps.split(',')))
		for milestone in steps:
			lr *= lr_decay_rate if epoch >= milestone else 1.
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr