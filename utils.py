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



def adjust_learning_rate(lr, lr_decay_steps, optimizer, epoch, lr_decay_rate=0.1, cos=False, max_epoch=800, warmup_epoch=0):
	"""Decay the learning rate based on schedule"""
	if epoch <= warmup_epoch:
		lr = lr * epoch / warmup_epoch
	else:
		if cos:
			epoch = epoch - warmup_epoch
			lr *= 0.5 * (1. + math.cos(math.pi * epoch / max_epoch))
		else:
			steps = list(map(int, lr_decay_steps.split(',')))
			for milestone in steps:
				lr *= lr_decay_rate if epoch >= milestone else 1.
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
