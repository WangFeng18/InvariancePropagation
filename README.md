# Unsupervised Learning by Invariance Propagation

This repository is the official implementation of [Unsupervised Learning by Invariance Propagation](https://arxiv.org/abs/2010.11694). 

## Model Training
To train the model(s) in the paper, run this command:

### ImageNet
```train
python main.py --exp 'your_path' --n_background 4096 --t 0.2 --blur --cos --network 'resnet50' --nonlinearhead 1 --weight_decay 1e-4
```

## Evaluation

To evaluate the model on ImageNet, run:

```eval
python -m downstream.linear_classification.linear_classification --gpus '0,1' --exp 'your_exp_path' --pretrained_path 'pretrain_path' --backbone 'resnet50'
```

