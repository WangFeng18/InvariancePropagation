import datasets.list as listdata
from torch.utils.data import DataLoader
from torchvision import transforms

def get_finetune_dataloader(args):
	train_transforms = transforms.Compose([
		transforms.RandomResizedCrop(size=224, scale=(0.2,1.)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	val_transforms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	train_dataset = listdata.ListData(root_folder='/data/stanford_cars/', listfilename='/data/stanford_cars/car_train.txt', transforms=train_transforms, label_type='int')

	val_dataset = listdata.ListData(root_folder='/data/stanford_cars/', listfilename='/data/stanford_cars/car_test.txt', transforms=val_transforms, label_type='int')

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.n_workers)	

	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=args.n_workers)
	return train_loader, val_loader