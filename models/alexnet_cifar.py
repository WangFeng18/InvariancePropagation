from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, feat):
		return feat.view(feat.size(0), -1)

class AlexNet(nn.Module):
	def __init__(self, low_dim=128):
		super(AlexNet, self).__init__()
		n_filters = [96, 256, 384, 384, 256]

		conv1 = nn.Sequential(
			nn.Conv2d(3, n_filters[0], kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(n_filters[0]),
			nn.ReLU(inplace=True),
		)
		pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		conv2 = nn.Sequential(
			nn.Conv2d(n_filters[0], n_filters[1], kernel_size=3, padding=1),
			nn.BatchNorm2d(n_filters[1]),
			nn.ReLU(inplace=True),
		)
		pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		conv3 = nn.Sequential(
			nn.Conv2d(n_filters[1], n_filters[2], kernel_size=3, padding=1),
			nn.BatchNorm2d(n_filters[2]),
			nn.ReLU(inplace=True),
		)
		conv4 = nn.Sequential(
			nn.Conv2d(n_filters[2], n_filters[3], kernel_size=3, padding=1),
			nn.BatchNorm2d(n_filters[3]),
			nn.ReLU(inplace=True),
		)
		conv5 = nn.Sequential(
			nn.Conv2d(n_filters[3], n_filters[4], kernel_size=3, padding=1),
			nn.BatchNorm2d(n_filters[4]),
			nn.ReLU(inplace=True),
		)
		pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

		num_pool5_feats = 4 * 4 * 256
		fc_block = nn.Sequential(
			Flatten(),
			nn.Linear(num_pool5_feats, low_dim),
		)

		self._feature_blocks = nn.ModuleList([
			conv1,
			pool1,
			conv2,
			pool2,
			conv3,
			conv4,
			conv5,
			pool5,
			fc_block,
		])
		self.all_feat_names = [
			'conv1',
			'pool1',
			'conv2',
			'pool2',
			'conv3',
			'conv4',
			'conv5',
			'pool5',
			'fc_block',
		]
		assert(len(self.all_feat_names) == len(self._feature_blocks))

	def forward(self, x, layer=-1):
		feat_idxs = [0,2,4,5,6]
		x = self._feature_blocks[0](x)
		if layer == 1:
			return x
		x = self._feature_blocks[1](x)
		x = self._feature_blocks[2](x)
		if layer == 2:
			return x
		x = self._feature_blocks[3](x)
		x = self._feature_blocks[4](x)
		if layer == 3:
			return x
		x = self._feature_blocks[5](x)
		if layer == 4:
			return x
		x = self._feature_blocks[6](x)
		if layer == 5:
			return x
		x = self._feature_blocks[7](x)
		x = self._feature_blocks[8](x)

		return x

	def forward_convs(self, x):
		feats = []
		feat_idxs = [0,2,4,5,6]
		for idx, subnet in enumerate(self._feature_blocks):
			x = subnet(x)
			if idx in feat_idxs:
				feats.append(x)
		return feats

	def get_L1filters(self):
		convlayer = self._feature_blocks[0][0]
		batchnorm = self._feature_blocks[0][1]
		filters = convlayer.weight.data
		scalars = (batchnorm.weight.data / torch.sqrt(batchnorm.running_var + 1e-05))
		filters = (filters * scalars.view(-1, 1, 1, 1).expand_as(filters)).cpu().clone()

		return filters

def alexnet(low_dim):
	return AlexNet(low_dim)

