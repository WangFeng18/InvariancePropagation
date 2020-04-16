from __future__ import division
import faiss
import numpy as np
DEFAULT_KMEANS_SEED = 1234
def run_kmeans(x, nmb_clusters, verbose=False,
			   seed=DEFAULT_KMEANS_SEED, gpu_device=0):
	"""
	Runs kmeans on 1 GPU.
	
	Args:
	-----
	x: data
	nmb_clusters (int): number of clusters
	
	Returns:
	--------
	list: ids of data in each cluster
	"""
	n_data, d = x.shape

	# faiss implementation of k-means
	clus = faiss.Clustering(d, nmb_clusters)
	clus.niter = 20
	clus.max_points_per_centroid = 10000000
	clus.seed = seed
	res = faiss.StandardGpuResources()
	flat_config = faiss.GpuIndexFlatConfig()
	flat_config.useFloat16 = False
	flat_config.device = gpu_device

	index = faiss.GpuIndexFlatL2(res, d, flat_config)

	# perform the training
	clus.train(x, index)
	centroids = faiss.vector_to_array(clus.centroids).reshape(nmb_clusters, d)
	_, I = index.search(x, 1)
	losses = faiss.vector_to_array(clus.obj)
	if verbose:
		print('k-means loss evolution: {0}'.format(losses))

	return [int(n[0]) for n in I], losses[-1], centroids


def run_kmeans_multi_gpu(x, nmb_clusters, verbose=False,
			   seed=DEFAULT_KMEANS_SEED, gpu_device=0):

	"""
	Runs kmeans on multi GPUs.

	Args:
	-----
	x: data
	nmb_clusters (int): number of clusters

	Returns:
	--------
	list: ids of data in each cluster
	"""
	n_data, d = x.shape
	ngpus = len(gpu_device)
	assert ngpus > 1

	# faiss implementation of k-means
	clus = faiss.Clustering(d, nmb_clusters)
	clus.niter = 20
	clus.max_points_per_centroid = 10000000
	clus.seed = seed
	res = [faiss.StandardGpuResources() for i in range(ngpus)]
	flat_config = []
	for i in gpu_device:
		cfg = faiss.GpuIndexFlatConfig()
		cfg.useFloat16 = False
		cfg.device = i
		flat_config.append(cfg)

	indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(ngpus)]
	index = faiss.IndexReplicas()
	for sub_index in indexes:
		index.addIndex(sub_index)

	# perform the training
	clus.train(x, index)
	centroids = faiss.vector_to_array(clus.centroids).reshape(nmb_clusters, d)
	_, I = index.search(x, 1)
	losses = faiss.vector_to_array(clus.obj)
	if verbose:
		print('k-means loss evolution: {0}'.format(losses))

	return [int(n[0]) for n in I], losses[-1], centroids


class Kmeans(object):
	"""
	Train <k> different k-means clusterings with different 
	random seeds. These will be used to compute close neighbors
	for a given encoding.
	"""
	def __init__(self, k, memory_bank, gpu_device=0):
		super().__init__()
		self.k = k
		self.memory_bank = memory_bank
		self.gpu_device = gpu_device

	def compute_clusters(self):
		"""
		Performs many k-means clustering.
		
		Args:
			x_data (np.array N * dim): data to cluster
		"""
		data = self.memory_bank.as_tensor()
		data_npy = data.cpu().detach().numpy()
		clusters = self._compute_clusters(data_npy)
		return clusters

	def _compute_clusters(self, data):
		pred_labels = []
		for k_idx, each_k in enumerate(self.k):
			# cluster the data

			if len(self.gpu_device) == 1: # single gpu
				I, _ = run_kmeans(data, each_k, seed=k_idx + DEFAULT_KMEANS_SEED,
								  gpu_device=self.gpu_device[0])
			else: # multigpu
				I, _ = run_kmeans_multi_gpu(data, each_k, seed=k_idx + DEFAULT_KMEANS_SEED,
								  gpu_device=self.gpu_device)

			clust_labels = np.asarray(I)
			pred_labels.append(clust_labels)
		pred_labels = np.stack(pred_labels, axis=0)
		pred_labels = torch.from_numpy(pred_labels).long()
		
		return pred_labels
