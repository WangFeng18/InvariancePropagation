# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from datasets.wm811.samplers import ImbalancedDatasetSampler
from datasets.wm811.transforms import *
from datasets.wm811.wm811k import WM811K


def balanced_loader(dataset: torch.utils.data.Dataset,
                    batch_size: int,
                    shuffle: bool = True,
                    num_workers: int = 0,
                    drop_last: bool = False,
                    pin_memory: bool = False
                    ):
    """Returns a `DataLoader` instance, which yields a class-balanced minibatch of samples."""

    sampler = ImbalancedDatasetSampler(dataset)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      sampler=sampler,
                      num_workers=num_workers,
                      drop_last=drop_last,
                      pin_memory=pin_memory)

def balanced_loader(dataset: torch.utils.data.Dataset,
                    batch_size: int,
                    shuffle: bool = True,
                    num_workers: int = 0,
                    drop_last: bool = False,
                    pin_memory: bool = False
                    ):
    """Returns a `DataLoader` instance, which yields a class-balanced minibatch of samples."""

    sampler = ImbalancedDatasetSampler(dataset)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      sampler=sampler,
                      num_workers=num_workers,
                      drop_last=drop_last,
                      pin_memory=pin_memory)

def get_dataloader(args):
    # Data
    data_kwargs = {
        'transform': WM811KTransform(size=96, mode='crop'),
        'decouple_input': True,
    }
    train_set = torch.utils.data.ConcatDataset([
        WM811K('./data/wm811k/unlabeled/train/', **data_kwargs),
        WM811K('./data/wm811k/labeled/train/', **data_kwargs),
    ])
    valid_set = torch.utils.data.ConcatDataset([
        WM811K('./data/wm811k/unlabeled/valid/', **data_kwargs),
        WM811K('./data/wm811k/labeled/valid/', **data_kwargs),
    ])
    test_set = torch.utils.data.ConcatDataset([
        WM811K('./data/wm811k/unlabeled/test/', **data_kwargs),
        WM811K('./data/wm811k/labeled/test/', **data_kwargs),
    ])

    train_ordered_labels = np.array([sample[1] for sample in train_set.datasets[0].samples] + [sample[1] for sample in train_set.datasets[1].samples])
    # train_ordered_labels = np.array([sample[1] for sample in train_set.datasets[0].samples])

    train_loader = DataLoader(train_set, args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=False)
    val_loader = DataLoader(valid_set, args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_set, args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=False)

    # train_loader = balanced_loader(train_set, args.batch_size, True, num_workers=args.n_workers)
    # val_loader = balanced_loader(valid_set, args.batch_size, True, num_workers=args.n_workers)
    # test_loader = balanced_loader(test_set, args.batch_size, True, num_workers=args.n_workers)
    return train_loader, val_loader, test_loader, train_set, valid_set, test_set, train_ordered_labels


def get_linear_dataloader(args):
    train_transform = WM811KTransform(size=config.input_size, mode='crop')
    test_transform  = WM811KTransform(size=config.input_size, mode='test')
    train_set = WM811K('./data/wm811k/labeled/train/',
                       transform=train_transform,
                       proportion=config.label_proportion,
                       decouple_input=config.decouple_input)
    valid_set = WM811K('./data/wm811k/labeled/valid/',
                       transform=test_transform,
                       decouple_input=config.decouple_input)

    train_loader = balanced_loader(train_set, args.batch_size, True, num_workers=args.n_workers)
    val_loader = balanced_loader(valid_set, args.batch_size, False, num_workers=args.n_workers)
    return train_loader, val_loader