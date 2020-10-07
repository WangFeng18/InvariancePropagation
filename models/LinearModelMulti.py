import torch
import torch.nn as nn
import numpy as np

class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		bs = x.size(0)
		return x.view(bs, -1)


class LinearModel(nn.Module):
	def __init__(self, target_size, n_filter, pool_type):
		super(LinearModel, self).__init__()
		# Adaptive Average Pooling -> Flatten -> Linear
		if pool_type == 'max':
			print('Using MaxPooling to downsample conv size')
			pool_layer = nn.AdaptiveMaxPool2d(target_size)
		elif pool_type == 'mean':
			print('Using AvgPooling to downsample conv size')
			pool_layer = nn.AdaptiveAvgPool2d(target_size)

		self.model = nn.Sequential(
				pool_layer,
				Flatten(),
				nn.BatchNorm1d(target_size*target_size*n_filter),
				nn.Linear(target_size*target_size*n_filter, 1000),
			)
		self.initilize()

	# def init(self):
	# 	for m in self.modules():
	# 		if isinstance(m, nn.Linear):
	# 			print('start initialization!')
	# 			fin = m.in_features
	# 			fout = m.out_features
	# 			std_val = np.sqrt(2.0/fout)
	# 			m.weight.data.normal_(0.0, std_val)
	# 			if m.bias is not None:
	# 				m.bias.data.fill_(0.0)

	def initilize(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.fill_(0.0)


	def forward(self, x):
		return self.model(x)

