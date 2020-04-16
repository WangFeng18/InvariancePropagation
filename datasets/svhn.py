from __future__ import division
from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np

def get_linear_dataloader(args):
	train_transforms = transforms.Compose([
		# transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
		transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
		transforms.RandomGrayscale(p=0.2),
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	val_transforms = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=train_transforms)

	val_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=val_transforms) 

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
						shuffle=True, pin_memory=False,
						num_workers=args.n_workers)	
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
						shuffle=False, pin_memory=False,
						num_workers=args.n_workers)
	return train_loader, val_loader


def get_dataloader(args):
	train_transforms = transforms.Compose([
		transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
		transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
		transforms.RandomGrayscale(p=0.2),
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	val_transforms = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	train_dataset = SVHNInstance(root='./data', split='train', download=True, transform=train_transforms)

	val_dataset = SVHNInstance(root='./data', split='test', download=True, transform=val_transforms) 

	train_ordered_labels = np.array(train_dataset.labels)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
						shuffle=True, pin_memory=False,
						num_workers=args.n_workers)	
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
						shuffle=False, pin_memory=False,
						num_workers=args.n_workers)
	return train_loader, val_loader, train_ordered_labels, train_dataset, val_dataset, 



class SVHNInstance(datasets.SVHN):
	"""SVHNInstance Dataset.
	"""
	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], int(self.labels[index])

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(np.transpose(img, (1, 2, 0)))

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return index, img, target


