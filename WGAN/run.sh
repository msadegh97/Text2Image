#!/usr/bin/env bash

cd /home/hlcv_team047/Text2Image/WGAN/
wandb login 7a44e6f35f9bf51e15cefc85c9c65093fc9c5d87

/home/hlcv_team047/miniconda3/envs/myenv1/bin/python train.py --experiment_name "WGAN_BIRDS_emb" --num_epochs 601 --log_interval 200 --lr 0.0001 --ngpu 1 --dataset birds --num_workers 8 --wandb=True --cls=False

#/home/hlcv_team019/miniconda3/envs/myenv1/bin/python train.py --experiment_name "WGAN_FLOWERS_CLS2" --num_epochs 300 --lr 0.00005 --log_interval 100 --ngpu 1 --dataset flowers --num_workers 8 --wandb=True --cls=True

#/home/hlcv_team019/miniconda3/envs/myenv1/bin/python train.py --experiment_name "WGAN_BIRDS2" --num_epochs 300 --lr 0.0001 --log_interval 100 --ngpu 1 --dataset birds --num_workers 8 --wandb=True --cls=False

#/home/hlcv_team019/miniconda3/envs/myenv1/bin/python train.py --experiment_name "WGAN_FLOWERS_2" --num_epochs 300 --lr 0.0001 --log_interval 100 --ngpu 1 --dataset flowers --num_workers 8 --wandb=True --cls=False

