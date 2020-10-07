import torch
from torch.utils.data import Dataset
from PIL import Image
import os
	
def loader(path):
	return pil_loader(path)

def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')


def accimage_loader(path):
	try:
		import accimage
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)

class MultiImageList(Dataset):
	def __init__(self, list_files, root, transform=None):
		self.samples = []
		if type(list_files) != list:
			list_files = [list_files,]
		lines = self.merge(list_files)
		for line in lines:
			temp = line.split(',')
			relpath = temp[0]
			label = list(map(lambda x:int(x), temp[1:]))
			if len(label) == 1:
				label = label[0]
			path = os.path.join(root, relpath)
			self.samples.append((path, label))
		self.transform = transform

	def merge(self, list_files):
		multi_lines = []
		L = None
		for list_file in list_files:
			with open(list_file) as f:
				lines = f.readlines()
				multi_lines.append(lines)
			if L is None:
				L = len(lines)
			else:
				assert(L == len(lines))

		target_lines = []
		for i_line in range(L):
			labels = []
			for i_file in range(len(list_files)):
				path, label = multi_lines[i_file][i_line].strip().split(',')
				if i_file == 0:
					labels.append(path)
					P = path
				else:
					assert P == path
				labels.append(label)
			target_lines.append(','.join(labels)+'\n')
		with open('temp.csv', 'w') as f:
			f.writelines(target_lines)
		return target_lines


	def __getitem__(self, index):
		path, target = self.samples[index]
		sample = loader(path)
		if self.transform is not None:
			sample = self.transform(sample)

		return sample, target

	def __len__(self):
		return len(self.samples)

class ImageList(Dataset):
	def __init__(self, list_file, root, transform=None, symbol_split=','):
		self.samples = []
		with open(list_file) as f:
			lines = f.readlines()
		for line in lines:
			relpath, label = line.split(symbol_split)
			path = os.path.join(root, relpath)
			label = int(label)
			self.samples.append((path, label))
		self.transform = transform

	def __getitem__(self, index):
		path, target = self.samples[index]
		sample = loader(path)
		if self.transform is not None:
			sample = self.transform(sample)

		return sample, target

	def __len__(self):
		return len(self.samples)

