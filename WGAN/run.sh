#!/usr/bin/env bash

cd /home/hlcv_team019/Text2Image/WGAN/
wandb login 7a44e6f35f9bf51e15cefc85c9c65093fc9c5d87

/home/hlcv_team019/miniconda3/envs/myenv1/bin/python train.py --experiment_name "WGAN_BIRDS_CLS" --num_epochs 300 --log_interval 100 --ngpu 1 --dataset birds --num_workers 8 --wandb=True --cls=True

/home/hlcv_team019/miniconda3/envs/myenv1/bin/python train.py --experiment_name "WGAN_FLOWERS_CLS" --num_epochs 300 --log_interval 100 --ngpu 1 --dataset flowers --num_workers 8 --wandb=True --cls=True

/home/hlcv_team019/miniconda3/envs/myenv1/bin/python train.py --experiment_name "WGAN_BIRDS" --num_epochs 300 --log_interval 100 --ngpu 1 --dataset birds --num_workers 8 --wandb=True --cls=False

/home/hlcv_team019/miniconda3/envs/myenv1/bin/python train.py --experiment_name "WGAN_FLOWERS" --num_epochs 300 --log_interval 100 --ngpu 1 --dataset flowers --num_workers 8 --wandb=True --cls=False

