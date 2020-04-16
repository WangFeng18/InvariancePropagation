# according to list file
# each row of the list file:
# relative_path, class
import logging
import torch
import os
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_loader(path):
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')

class ListData(data.Dataset):
	def __init__(self, root_folder, listfilename, transforms, label_type='str'):
		self.label_type = label_type
		self.listfilename = listfilename
		self.root_folder = root_folder
		self.transforms = transforms
		self.samples, self.targets = self.create_sample()
		
	def create_sample(self):
		samples = []
		targets = []
		names = []
		with open(self.listfilename, 'r') as f:
			lines = f.readlines()

		for line in lines:
			line = line.strip()
			relpath, name = line.split(',')
			if name not in names:
				names.append(name)
			abspath = os.path.join(self.root_folder, relpath)
			samples.append(abspath)
			targets.append(name)

		if self.label_type == 'str':
			names = sorted(names)
			name2id = {}
			for cid, name in enumerate(names):
				name2id[name] = cid
			for i, target in enumerate(targets):
				targets[i] = name2id[target]
			logging.info('Total of {} Classes'.format(len(name2id)))
		else:
			for i, target in enumerate(targets):
				targets[i] = int(target)

		logging.info('Total of {} Samples'.format(len(samples)))
		return samples, targets

	def __getitem__(self, index):
		path = self.samples[index]
		target = self.targets[index]
		sample = pil_loader(path)
		if self.transforms is not None:
			sample = self.transforms(sample)
		return sample, target

	def __len__(self):
		return len(self.samples)
