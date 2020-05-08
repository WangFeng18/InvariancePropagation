# Unsupervised Learning by Invariance Propagation

This repository is the official implementation of [Unsupervised Learning by Invariance Propagation](https://arxiv.org/abs/---). 
![concept](paper/concept.pdf)
<!-- > 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
To train the model(s) in the paper, run this command:
### CIFAR10
```train
python main.py --exp 'your_exp_folder_path' --dataset 'cifar10' --weight_decay 5e-4 --t 0.1 --network 'resnet18_cifar' --n_pos 20 --mix
```
### CIFAR100
```train
python main.py --exp 'your_exp_folder_path' --dataset 'cifar100' --weight_decay 5e-4 --t 0.1 --network 'resnet18_cifar' --n_pos 20 --diffusion_layer 4 --mix
```
### SVHN
```train
python main.py --exp 'your_exp_folder_path' --dataset 'svhn' --weight_decay 5e-4 --t 0.1 --network 'resnet18_cifar' --n_pos 20 --mix
```

### ImageNet
```train
python main.py --exp 'your_exp_folder_path' --dataset 'imagenet' --weight_decay 1e-4 --t 0.07 --network 'resnet50'
```

Note that we do not use mixcontrast augmentations in ImageNet due to it will slow down the training process.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```
> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).
## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.
## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 