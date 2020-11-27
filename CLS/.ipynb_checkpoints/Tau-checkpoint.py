import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

class TauObserver(object):
	def __init__(self, exp):
		self.exp = exp
		self.device = torch.device('cpu')

	def read_data(self, step=0, name=None):
		self.similarities = []
		self.indices = []
		self.positive_similarities = []
		self.sorted_sim = []

		files = os.listdir(self.exp)
		files = sorted(files, key=lambda name:int(name.split('.')[0][9:]))
		if name is not None:
			ckpt = torch.load(os.path.join(self.exp, name), map_location=self.device)
		else:
			print(files[step], end='\n')
			ckpt = torch.load(os.path.join(self.exp, files[step]), map_location=torch.device('cpu'))

		self.similarities = ckpt['sim'].to(self.device).detach()
		self.indices = ckpt['index'].to(self.device).detach()
		self.similarities = self.similarities[:1000,:].to(self.device)
		self.indices = self.indices[:1000].to(self.device)
		self.positive_similarities = self.similarities[
			range(self.similarities.shape[0]), self.indices
		]

	def softmax(self, tau):
		return torch.exp(self.positive_similarities/tau) / torch.exp(self.similarities/tau).sum(dim=1)
		
	def mean_sp(self):
		# sim Sn x N
		# ind Sn
		return self.positive_similarities.mean().item()

	def mean_sn(self):
		son = self.similarities.sum(dim=1) - self.positive_similarities
		par = self.similarities.shape[1]-1
		a = son.mean()/float(par)
		print(a)
		return a

	def meanP(self, tau):
		return self.softmax(tau).mean().item()

	def DlDsp(self, tau):
		return (- 1/tau * (1-self.softmax(tau))).mean().item()

	def E(self, similarities, tau):
		with torch.no_grad():
			respect_E = (similarities * torch.exp(similarities/tau)/torch.exp(similarities/tau).sum(dim=1, keepdim=True)).sum(dim=1, keepdim=True).detach()
			a = respect_E.mean().item()
		return a, respect_E.cpu()

	def delta(self, tau):
		P = self.softmax(tau)
		return (tau * (1-P).mean()/P.mean() + self.E(tau)).item()

	def plot_dsp(self):
		alldsp = []
		for tau in np.linspace(0,1,100):
			alldsp.append(self.DlDsp(tau))
		alldsp = np.array(alldsp)
		plt.plot(np.linspace(0,1,100), alldsp)
		plt.show()

	# The above method are based on positive statistics
	def rank(self):
		with torch.no_grad():
			similarities = self.similarities.clone()
			similarities[list(range(similarities.size(0))), self.indices] = 0.0
			sorted_sim, _ = torch.sort(similarities.to(torch.device('cuda:3')), dim=1, descending=True)
			sorted_sim = sorted_sim
		return sorted_sim.detach().cpu()

	def show(self, df, name, tau=None, topk=20):
		mean_sp = float('{:.4f}'.format(self.mean_sp()))
		mean_sn = float('{:.4f}'.format(self.mean_sn()))
		mean_P = float('{:.4f}'.format(self.meanP(tau=0.07)))
		sortedsim = self.rank()
		E, respect_E = self.E(sortedsim, tau)
		n_superhard = (sortedsim > respect_E ).sum(dim=1).float().mean().item()

		hd2048_E, hd2048_respect_E = self.E(sortedsim[:,:2048], tau)
		n_hd2048superhard = (sortedsim[:,:2048] > hd2048_respect_E ).sum(dim=1).float().mean().item()

		hd4096_E, hd4096_respect_E = self.E(sortedsim[:,:4096], tau)
		n_hd4096superhard = (sortedsim[:, :4096] > hd4096_respect_E ).sum(dim=1).float().mean().item()

		hd8192_E, hd8192_respect_E = self.E(sortedsim[:, :8192], tau)
		n_hd8192superhard = (sortedsim[:, :8192] > hd8192_respect_E ).sum(dim=1).float().mean().item()

		sortedsim = sortedsim.mean(dim=0).numpy()
		sortedsim = np.around(sortedsim, decimals=4)

		if df is None:	
			df = pd.DataFrame(
				[[name, mean_sp, mean_sn, mean_P, E, n_superhard, hd2048_E, n_hd2048superhard, hd4096_E, n_hd4096superhard, hd8192_E, n_hd8192superhard]+ list(sortedsim[:topk])],
				columns=['name', 'MeanPosSim', 'MeanNegSim', 'MeanPoss', 'E', 'nSuperHard', 'hd2048_E', 'nhd2048SuperHard', 'hd4096_E', 'nhd4096SuperHard', 'hd8192_E', 'nhd8192SuperHard', ] + ['rank_{}'.format(i) for i in range(1, topk+1)]
			)
		else:
			new_row = {
				'name': name,
				'MeanPosSim': mean_sp,
				'MeanNegSim': mean_sn,
				'MeanPoss': mean_P,
				'E': E,
				'hd2048_E': hd2048_E,
				'hd4096_E': hd4096_E,
				'hd8192_E': hd8192_E,
				'nSuperHard': n_superhard,
				'nhd2048SuperHard': n_hd2048superhard,
				'nhd4096SuperHard': n_hd4096superhard,
				'nhd8192SuperHard': n_hd8192superhard,
			}
			for i, ele in enumerate(list(sortedsim[:topk])):
				new_row.update({'rank_{}'.format(i+1): ele})
			df = df.append(new_row, ignore_index=True)
		return df

