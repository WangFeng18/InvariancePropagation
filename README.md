# Unsupervised Learning by Invariance Propagation

This repository is the official implementation of [Unsupervised Learning by Invariance Propagation](https://arxiv.org/abs/2010.11694). 

<img src="img/graph.png" alt="" align=center />

## Pretraining on Natual Images
### Train on ImageNet
To train the model(s) in the paper, run this command:

```
python main.py --exp 'your_path' --n_background 4096 --t 0.2 --blur --cos --network 'resnet50' --nonlinearhead 1 --weight_decay 1e-4
```

### Evaluation

To evaluate the model on ImageNet, run:

```eval
python -m downstream.linear_classification.linear_classification --gpus '0,1' --exp 'your_exp_path' --pretrained_path 'pretrain_path' --backbone 'resnet50'
```

Notice that in the paper, to calculate the BFS results, we require to record the id of neighbours of each anchor point. For computational efficiency, we apprximate the BFS results by only concatenating the neighbours of each point, up to L steps. This results may be a little different with the real BFS results due to there exists repeated samples, however it works pretty well, both effectively and efficiently. Pretrained model can be found [here](https://drive.google.com/file/d/1mE1WafTJbEdJav5wF98I_RvTksSxuRpU/view?usp=sharing).

### Train on Cifar
To train the model(s) in cifar10 and cifar100 or svhn, run this command:

```
# cifar10
python main.py --exp 'your_path' -n_background 4096 --t 0.2 --blur --cos --network 'resnet18_cifar' --nonlinearhead 1 --weight_decay 5e-4 --n_pos 20 --dataset 'cifar10'
# cifar100
python main.py --exp 'your_path' -n_background 4096 --t 0.2 --blur --cos --network 'resnet18_cifar' --nonlinearhead 1 --weight_decay 5e-4 --n_pos 20 --dataset 'cifar100'
# svhn
python main.py --exp 'your_path' -n_background 4096 --t 0.2 --blur --cos --network 'resnet18_cifar' --nonlinearhead 1 --weight_decay 5e-4 --n_pos 20 --dataset 'svhn'
```

### Evaluation
To train the model(s) in cifar10 and cifar100 run this command:

```eval
# cifar10
python -m downstream.linear_classification.eval_linear --gpus '0,1' --exp 'your_exp_path' --pretrained_path 'pretrain_path' --backbone 'resnet18_cifar' --dataset 'cifar10'
# cifar100
python -m downstream.linear_classification.eval_linear --gpus '0,1' --exp 'your_exp_path' --pretrained_path 'pretrain_path' --backbone 'resnet18_cifar' --dataset 'cifar100'
# svhn
python -m downstream.linear_classification.eval_linear --gpus '0,1' --exp 'your_exp_path' --pretrained_path 'pretrain_path' --backbone 'resnet18_cifar' --dataset 'svhn'
```

## Pretraining on Defect Classification Dataset
For validate the effectiveness and practicabilities of the proposed algorithms, we can also train and evaluate our method on Defect Detection Dataset.

### Train on WM811.
```
python main.py --gpus '0,1,2' --exp 'output/' --n_background 4096 --t 0.07 --cos --network 'resnet18_wm811' --dataset 'wm811' --nonlinearhead 0 --weight_decay 5e-4
```

### Evaluation

To evaluate the model on WM811, run:

```eval
python -m downstream.fine_tune_wm811 --save_folder 'your_output_folder' --model_path 'your_pretrain_model' --model 'resnet18_wm811' --dataset 'wm811' --weight_decay 1e-3 --learning_rate1 0.001 --learning_rate2 0.002 --label_smoothing 0.1 --dropout 0.5
```

