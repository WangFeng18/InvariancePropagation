from __future__ import division
import logging
import torch
import torch.nn as nn
import numpy as np
#from Kmeans import *
from sklearn.metrics.cluster import normalized_mutual_info_score
import random
import math
import queue
import multiprocessing

DEFAULT_KMEANS_SEED = 1234

def GaussianRampUp(i_epoch, end_epoch, weight=5):
	m = max(1-i_epoch/end_epoch, 0)**2
	v = np.exp(-weight * m)
	return v

def BinaryRampUp(i_epoch, end_epoch):
	return int(i_epoch > end_epoch)
	
def l2_normalize(x):
	return x / torch.sqrt(torch.sum(x**2, dim=1).unsqueeze(1))

def propagate_single(diffusion_layer, point_index, memory_bank_neigh):
		neigh = memory_bank_neigh[point_index]
		# print(neigh)
		q = queue.Queue()
		mark = neigh[-1]
		for ele in neigh: 
			q.put(ele)
		neigh = set(neigh)
		layer = 1
		# tranverse through queue
		last_put = None
		while not q.empty() and layer < diffusion_layer:
			curr_node = q.get()
			curr_node_neigh = memory_bank_neigh[curr_node]
			for curr_node_ele in curr_node_neigh:
				if curr_node_ele not in neigh:
					neigh.add(curr_node_ele)
					q.put(curr_node_ele)
					last_put = curr_node_ele
			if curr_node == mark:
				layer += 1
				mark = last_put
		return list(neigh)

class MixPointLoss(nn.Module):
	def __init__(self, t):
		super(MixPointLoss, self).__init__()
		self.t = t

	def forward(self, points, Alphas, point_indices_A, point_indices_B, memory_bank):
		norm_points = l2_normalize(points)
		points_sim = self._exp(memory_bank.get_all_dot_products(norm_points))
		positive_sim_A = points_sim[list(range(points_sim.size(0))), point_indices_A]
		positive_sim_B = points_sim[list(range(points_sim.size(0))), point_indices_B]
		positive_sim = positive_sim_A * Alphas + positive_sim_B * (1-Alphas)
		# hard_negatives_sim, hn_indices = points_sim.topk(k=4096, dim=1, largest=True, sorted=True)

		return -(positive_sim/points_sim.sum(dim=1) + 1e-7).log().mean()

	def _exp(self, dot_prods):
		# Z = 2876934.2 / 1281167 * self.data_len
		return torch.exp(dot_prods / self.t)

class RatioLoss(nn.Module):
	def __init__(self, t):
		super(RatioLoss, self).__init__()
		self.t = t

	def forward(self, points, point_indices, memory_bank):
		norm_points = l2_normalize(points)
		similarities = memory_bank.get_all_dot_products(norm_points)
		points_sim = self._exp(similarities)
		positive_sim = points_sim[list(range(points_sim.size(0))), point_indices]
		negative_sim = points_sim.sum(dim=1) - positive_sim
		return -(positive_sim/negative_sim + 1e-7).log().mean(), similarities

	def _exp(self, dot_prods):
		# Z = 2876934.2 / 1281167 * self.data_len
		return torch.exp(dot_prods / self.t)

def calculate_entropy(similarities, point_indices, t):
	F = torch.exp(similarities/t.unsqueeze(dim=1))
	F[list(range(F.size(0))), point_indices] = 0
	r = F/F.sum(dim=1, keepdim=True)
	entropy = -(r * (r + 1e-7).log()).sum(dim=1)
	return entropy

def binary_search_entropy(similarities, target_entropy, point_indices, iterations=13):
	left = 0.0
	right = 10.0
	scale = (right - left) / 4.0
	vcenter = (right + left) / 2.0
	centers = torch.ones(similarities.size(0), device=similarities.device) * vcenter
	# scale = 0.25
	# centers = torch.ones(similarities.size(0), device=similarities.device) * 0.5

	for i_iteration in range(iterations):
		#exps = torch.exp(similarities/centers.unsqueeze(dim=1))
		#distribution = exps / exps.sum(dim=1, keepdim=True)
		entropy = calculate_entropy(similarities, point_indices, centers)

		indication = 2*(entropy < target_entropy) - 1.0
		centers = centers + scale * indication
		scale = scale / 2.0

	distribution = torch.nn.functional.softmax(similarities/centers.unsqueeze(dim=1))
	entropy = calculate_entropy(similarities, point_indices, centers)

	return distribution, centers, entropy

class FixedEntropyPointLoss(nn.Module):
	def __init__(self, target_entropy):
		super(FixedEntropyPointLoss, self).__init__()
		self.target_entropy = target_entropy

	def forward(self, points, point_indices, memory_bank):
		norm_points = l2_normalize(points)
		similarities = memory_bank.get_all_dot_products(norm_points)

		distribution, centers, entropy = binary_search_entropy(similarities, self.target_entropy, point_indices)
		positive_probability = distribution[list(range(distribution.size(0))), point_indices]
		loss = -(positive_probability + 1e-7).log().mean()

		return loss, similarities, centers.mean().item(), entropy.mean().item()

class FixedEntropyHardNegativeLoss(nn.Module):
	def __init__(self, target_entropy, n_background=4096):
		super(FixedEntropyHardNegativeLoss, self).__init__()
		self.target_entropy = target_entropy
		self.n_background = n_background

	def forward(self, points, point_indices, memory_bank, target_centers=None):
		norm_points = l2_normalize(points)
		similarities = memory_bank.get_all_dot_products(norm_points)
		hard_similarities, hn_indices = similarities.topk(k=self.n_background, dim=1, largest=True, sorted=True)

		_, centers, entropy = binary_search_entropy(hard_similarities, self.target_entropy, point_indices)

		positive_similarities = similarities[list(range(similarities.size(0))), point_indices]

		if target_centers is not None:
			centers = torch.ones(similarities.size(0), device=similarities.device) * target_centers
			
		condition_p = torch.exp(positive_similarities/centers - 1.0/centers) / torch.exp(hard_similarities/centers.unsqueeze(dim=1) - 1.0/centers.unsqueeze(dim=1)).sum(dim=1)

		loss = - (condition_p + 1e-7).log().mean()

		return loss, similarities, centers.mean().item(), entropy.mean().item()

class PointLoss(nn.Module):
	def __init__(self, t):
		super(PointLoss, self).__init__()
		self.t = t

	def forward(self, points, point_indices, memory_bank):
		norm_points = l2_normalize(points)
		similarities = memory_bank.get_all_dot_products(norm_points)
		points_sim = self._exp(similarities)
		positive_sim = points_sim[list(range(points_sim.size(0))), point_indices]
		return -(positive_sim/points_sim.sum(dim=1) + 1e-7).log().mean(), similarities

	def _exp(self, dot_prods):
		# Z = 2876934.2 / 1281167 * self.data_len
		return torch.exp(dot_prods / self.t)

class SingleLoss(nn.Module):
	def __init__(self):
		super(SingleLoss, self).__init__()

	def forward(self, points, point_indices, memory_bank):
		norm_points = l2_normalize(points)
		similarities = memory_bank.get_all_dot_products(norm_points)
		positive_sim = similarities[list(range(similarities.size(0))), point_indices]
		maximum, _ = similarities.max(dim=1)
		loss = (maximum - positive_sim).mean()

		return loss, similarities

class FlatLoss(nn.Module):
	def __init__(self):
		super(FlatLoss, self).__init__()

	def forward(self, points, point_indices, memory_bank):
		norm_points = l2_normalize(points)
		similarities = memory_bank.get_all_dot_products(norm_points)
		positive_sim = similarities[list(range(similarities.size(0))), point_indices]
		N = float(similarities.size(1))
		loss = (- positive_sim + 1/N * similarities.sum(dim=1)).mean()
		return loss, similarities

class HardFlatLoss(nn.Module):
	def __init__(self, n_background):
		super(HardFlatLoss, self).__init__()
		self.n_background = n_background

	def forward(self, points, point_indices, memory_bank):
		norm_points = l2_normalize(points)
		similarities = memory_bank.get_all_dot_products(norm_points)
		points_sim = similarities
		positive_sim = points_sim[list(range(points_sim.size(0))), point_indices]
		hard_negatives_sim, hn_indices = points_sim.topk(k=self.n_background, dim=1, largest=True, sorted=True)

		N = float(hard_negatives_sim.size(1))
		loss = (- positive_sim + 1/N * hard_negatives_sim.sum(dim=1)).mean()
		return loss, similarities

class OraclePointLoss(nn.Module):
	def __init__(self, t):
		super(OraclePointLoss, self).__init__()
		self.t = t

	def forward(self, points, point_indices, index, memory_bank):
		norm_points = l2_normalize(points)
		similarities = memory_bank.get_all_dot_products(norm_points)
		points_sim = self._exp(similarities)
		positive_sim = points_sim[list(range(points_sim.size(0))), point_indices]
		return -(positive_sim/points_sim.sum(dim=1) + 1e-7).log().mean(), similarities

	def _exp(self, dot_prods):
		# Z = 2876934.2 / 1281167 * self.data_len
		return torch.exp(dot_prods / self.t)

class RingLoss(nn.Module):
	def __init__(self, t, n_potential_positive=100, n_background=4096):
		super(RingLoss, self).__init__()
		self.t = t
		self.n_background = n_background
		self.n_potential_positive = n_potential_positive

	def forward(self, points, point_indices, memory_bank):
		norm_points = l2_normalize(points)
		similarities = memory_bank.get_all_dot_products(norm_points)
		points_sim = self._exp(similarities)
		positive_sim = points_sim[list(range(points_sim.size(0))), point_indices]
		hard_negatives_sim, hn_indices = points_sim.topk(k=self.n_background, dim=1, largest=True, sorted=True)
		potential_positive_sim = hard_negatives_sim[:,:self.n_potential_positive]
		total_positive_sim = positive_sim + potential_positive_sim.sum(dim=1)

		return -(total_positive_sim/hard_negatives_sim.sum(dim=1) + 1e-7).log().mean(), similarities.detach()

	def _exp(self, dot_prods):
		# Z = 2876934.2 / 1281167 * self.data_len
		return torch.exp(dot_prods / self.t)

class AlternativeRingLoss(nn.Module):
	def __init__(self, t, n_potential_positive=100, n_background=4096):
		super(AlternativeRingLoss, self).__init__()
		self.t = t
		self.n_background = n_background
		self.n_potential_positive = n_potential_positive
		self.lastneighbour = False

	def forward(self, points, point_indices, memory_bank):
		norm_points = l2_normalize(points)
		similarities = memory_bank.get_all_dot_products(norm_points)
		points_sim = self._exp(similarities)
		positive_sim = points_sim[list(range(points_sim.size(0))), point_indices]
		hard_negatives_sim, hn_indices = points_sim.topk(k=self.n_background, dim=1, largest=True, sorted=True)
		potential_positive_sim = hard_negatives_sim[:,:self.n_potential_positive]
		if not self.lastneighbour:
			total_positive_sim = positive_sim + potential_positive_sim.sum(dim=1)
			self.lastneighbour = True
		else:
			total_positive_sim = positive_sim 
			self.lastneighbour = False

		return -(total_positive_sim/hard_negatives_sim.sum(dim=1) + 1e-7).log().mean(), similarities.detach()

	def _exp(self, dot_prods):
		# Z = 2876934.2 / 1281167 * self.data_len
		return torch.exp(dot_prods / self.t)

class HardNegativePointLoss(nn.Module):
	def __init__(self, t, n_background=4096):
		super(HardNegativePointLoss, self).__init__()
		self.t = t
		self.n_background = n_background

	def forward(self, points, point_indices, memory_bank):
		norm_points = l2_normalize(points)
		similarities = memory_bank.get_all_dot_products(norm_points)
		points_sim = self._exp(similarities)
		positive_sim = points_sim[list(range(points_sim.size(0))), point_indices]
		hard_negatives_sim, hn_indices = points_sim.topk(k=self.n_background, dim=1, largest=True, sorted=True)

		return -(positive_sim/hard_negatives_sim.sum(dim=1) + 1e-7).log().mean(), similarities.detach()

	def _exp(self, dot_prods):
		# Z = 2876934.2 / 1281167 * self.data_len
		return torch.exp(dot_prods / self.t)

class NearstNeighbourLoss(object):
	def __init__(self, t, n_background=4096, k=80):
		self.t = t
		self.n_background = n_background
		pass

class InvariancePropagationLoss(nn.Module):
	def __init__(self, t, n_background=4096, diffusion_layer=3, k=4, n_pos=50, exclusive=True, InvP=True, hard_pos=True):
		super(InvariancePropagationLoss, self).__init__()
		self.t = t
		self.n_background = n_background
		self.diffusion_layer = diffusion_layer
		self.k = k
		self.n_pos = n_pos
		self.exclusive = exclusive
		self.InvP = InvP
		self.hard_pos = hard_pos
		if self.hard_pos == False:
			logging.info('WARNING: Not Using Hard Postive Samples')
		print('DIFFUSION_LAYERS: {}'.format(self.diffusion_layer))
		print('K_nearst: {}'.format(self.k))
		print('N_POS: {}'.format(self.n_pos))

	def update_nn(self, background_indices, point_indices, memory_bank):
		nei = background_indices[:, :self.k+1]
		condition = (nei == point_indices.unsqueeze(dim=1))
		backup = nei[:, self.k:self.k+1].expand_as(nei)
		nei_exclusive = torch.where(
			condition,
			backup,
			nei,
		)
		nei_exclusive = nei_exclusive[:, :self.k]
		memory_bank.neigh[point_indices] = nei_exclusive

	def propagate(self, point_indices, memory_bank):
		cur_point = 0
		matrix = memory_bank.neigh[point_indices] # 256 x 4
		end_point = matrix.size(1) - 1
		layer = 2
		while layer <= self.diffusion_layer:
			current_nodes = matrix[:, cur_point] # 256

			sub_matrix = memory_bank.neigh[current_nodes] # 256 x 4
			matrix = torch.cat([matrix, sub_matrix], dim=1)

			if cur_point == end_point:
				layer += 1
				end_point = matrix.size(1) - 1
			cur_point += 1
		return matrix


	def forward(self, points, point_indices, memory_bank, return_neighbour=False):
		norm_points = l2_normalize(points)
		all_sim = self._exp(memory_bank.get_all_dot_products(norm_points))
		self_sim = all_sim[list(range(all_sim.size(0))), point_indices]
		background_sim, background_indices = all_sim.topk(k=self.n_background, dim=1, largest=True, sorted=True)

		lossA = -(self_sim/background_sim.sum(dim=1) + 1e-7).log().mean()

		if self.InvP:
			# invariance propagation
			neighs = self.propagate(point_indices, memory_bank)

			lossB = 0
			background_exclusive_sim = background_sim.sum(dim=1) - self_sim

			## one
			pos_sim = torch.gather(all_sim, index=neighs, dim=1)
			if self.hard_pos:
				hard_pos_sim, hp_indices = pos_sim.topk(k=min(self.n_pos, pos_sim.size(1)), dim=1, largest=False, sorted=True)
			else:
				hard_pos_sim, hp_indices = pos_sim.topk(k=min(self.n_pos, pos_sim.size(1)), dim=1, largest=True, sorted=True)

			if self.exclusive:	
				lossB = -( hard_pos_sim.sum(dim=1) / background_exclusive_sim + 1e-7).log().mean()
			else:
				# print('no exclusive')
				lossB = -( hard_pos_sim.sum(dim=1) / (background_exclusive_sim + self_sim) + 1e-7).log().mean()

		else:
			lossB = 0.0
			neighs = None

		self.update_nn(background_indices, point_indices, memory_bank)

		if return_neighbour:
			return lossA, lossB, neighs, torch.gather(neighs, index=hp_indices, dim=1)
		else:
			return lossA, lossB, background_indices, neighs

	def _exp(self, dot_prods):
		# Z = 2876934.2 / 1281167 * self.data_len
		return torch.exp(dot_prods / self.t)

class MemoryBank_v1(object):
	def __init__(self, n_points, train_ordered_labels, writer, device, m=0.5):
		self.m = m
		logging.info('M: {}'.format(self.m))
		self.device = device
		logging.info('memery bank initialize with {} points'.format(n_points))
		self.n_points = n_points
		self.points = torch.zeros(n_points, 128).to(device).detach()
		self.cluster_number = 0
		self.point_centroid = None
		self.writer = writer
		self.train_ordered_labels = train_ordered_labels
		self.k = 4
		self.neigh = torch.zeros(n_points, self.k, dtype=torch.long).to(device).detach()
		self.neigh_sim = torch.zeros(n_points, self.k).to(device).detach()

	def clear(self):
		self.points = torch.zeros(self.n_points, 128).to(self.device).detach()
		
	def random_init_bank(self):
		logging.info('memery bank random initialize with {} points'.format(self.n_points))
		stdv = 1. / math.sqrt(128/3)
		self.points = torch.rand(self.n_points, 128).mul_(2*stdv).add_(-stdv).to(self.device).detach()

	def update_points(self, points, point_indices):
		norm_points = l2_normalize(points)
		data_replace = self.m * self.points[point_indices,:] + (1-self.m) * norm_points
		self.points[point_indices,:] = l2_normalize(data_replace)

	def sample_points(self):
		point_indices = []
		for centroid, points in enumerate(self.centroid_points):
			point_indices.append(random.choice(points))
		return self.points[point_indices]

	def cluster_points(self, nmb_clusters):
		# clustering and allocate the new centroids to points
		logging.info('clustering points into {} clusters'.format(nmb_clusters))
		self.last_point_centroid = self.point_centroid
		self.point_centroid, loss, self.centroids = run_kmeans(self.points.cpu().numpy(), nmb_clusters, verbose=True, seed=DEFAULT_KMEANS_SEED+self.cluster_number, gpu_device=0)

		self.centroid_points = [[] for i in range(nmb_clusters)]
		for index_point, centroid in enumerate(self.point_centroid):
			self.centroid_points[centroid].append(index_point)

		self.centroids = torch.from_numpy(self.centroids).to(self.device).detach()
		self.centroids = l2_normalize(self.centroids)
		try:
			nmi_with_god = normalized_mutual_info_score(self.train_ordered_labels, self.point_centroid, average_method='arithmetic')
			logging.info('[NMI] with god {}'.format(nmi_with_god))
			self.writer.add_scalar('nmi_with_god', nmi_with_god, self.cluster_number)

			nmi_with_last = normalized_mutual_info_score(self.last_point_centroid, self.point_centroid, average_method='arithmetic')
			logging.info('[NMI] with last {}'.format(nmi_with_last))
			self.writer.add_scalar('nmi_with_last', nmi_with_last, self.cluster_number)
		except:
			logging.info('[NMI calculate false]')
			pass
		self.cluster_number += 1
		logging.info('finish clustering'.format(nmb_clusters))

	def get_all_dot_products(self, points):
		assert len(points.size()) == 2
		return torch.matmul(points, torch.transpose(self.points, 1, 0))




class InvariancePropagationLoss_v2(nn.Module):
	def __init__(self, t, n_background=4096, diffusion_layer=3, k=4, n_pos=50, exclusive=True, exclusive_easypos=False, InvP=True, hard_pos=True):
		super(InvariancePropagationLoss, self).__init__()
		self.t = t
		self.n_background = n_background
		self.diffusion_layer = diffusion_layer
		self.k = k
		self.n_pos = n_pos
		self.exclusive = exclusive
		self.exclusive_easypos = exclusive_easypos
		self.InvP = InvP
		self.hard_pos = hard_pos
		if self.hard_pos == False:
			logging.info('WARNING: Not Using Hard Postive Samples')
		print('DIFFUSION_LAYERS: {}'.format(self.diffusion_layer))
		print('K_nearst: {}'.format(self.k))
		print('N_POS: {}'.format(self.n_pos))

	def update_nn(self, background_indices, point_indices, memory_bank):
		nei = background_indices[:, :self.k+1]
		condition = (nei == point_indices.unsqueeze(dim=1))
		backup = nei[:, self.k:self.k+1].expand_as(nei)
		nei_exclusive = torch.where(
			condition,
			backup,
			nei,
		)
		nei_exclusive = nei_exclusive[:, :self.k]
		memory_bank.neigh[point_indices] = nei_exclusive

	def propagate(self, point_indices, memory_bank):
		cur_point = 0
		matrix = memory_bank.neigh[point_indices] # 256 x 4
		end_point = matrix.size(1) - 1
		layer = 2
		while layer <= self.diffusion_layer:
			current_nodes = matrix[:, cur_point] # 256

			sub_matrix = memory_bank.neigh[current_nodes] # 256 x 4
			matrix = torch.cat([matrix, sub_matrix], dim=1)

			if cur_point == end_point:
				layer += 1
				end_point = matrix.size(1) - 1
			cur_point += 1
		
		matrix_rows = []
		for i in range(point_indices.size(0)):
			matrix_rows.append(torch.unique(matrix[i]))
		return matrix


	def forward(self, points, point_indices, memory_bank, return_neighbour=False):
		norm_points = l2_normalize(points)
		all_sim = self._exp(memory_bank.get_all_dot_products(norm_points))
		self_sim = all_sim[list(range(all_sim.size(0))), point_indices]
		background_sim, background_indices = all_sim.topk(k=self.n_background, dim=1, largest=True, sorted=True)

		lossA = -(self_sim/background_sim.sum(dim=1) + 1e-7).log().mean()

		if self.InvP:
			# invariance propagation
			neighs = self.propagate(point_indices, memory_bank)

			lossB = 0
			background_exclusive_sim = background_sim.sum(dim=1) - self_sim

			## one
			pos_sim = torch.gather(all_sim, index=neighs, dim=1)
			if self.hard_pos:
				hard_pos_sim, hp_indices = pos_sim.topk(k=min(self.n_pos, pos_sim.size(1)), dim=1, largest=False, sorted=True)
			else:
				hard_pos_sim, hp_indices = pos_sim.topk(k=min(self.n_pos, pos_sim.size(1)), dim=1, largest=True, sorted=True)

			if self.exclusive:	
				if self.exclusive_easypos:
					lossB = -( hard_pos_sim.sum(dim=1) / \
						(background_exclusive_sim - pos_sim.sum(dim=1) + hard_pos_sim.sum(dim=1)) + 1e-7)\
						.log().mean()
				else:
					lossB = -( hard_pos_sim.sum(dim=1) / background_exclusive_sim + 1e-7).log().mean()
			else:
				# print('no exclusive')
				lossB = -( hard_pos_sim.sum(dim=1) / (background_exclusive_sim + self_sim) + 1e-7).log().mean()

		else:
			lossB = 0.0
			neighs = None

		self.update_nn(background_indices, point_indices, memory_bank)

		if return_neighbour:
			return lossA, lossB, neighs, torch.gather(neighs, index=hp_indices, dim=1)
		else:
			return lossA, lossB, background_indices, neighs

	def _exp(self, dot_prods):
		# Z = 2876934.2 / 1281167 * self.data_len
		return torch.exp(dot_prods / self.t)