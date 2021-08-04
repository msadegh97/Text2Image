#!/usr/bin/env bash

cd /home/hlcv_team047/Text2Image/GAN/
wandb login 7a44e6f35f9bf51e15cefc85c9c65093fc9c5d87

/home/hlcv_team047/miniconda3/envs/myenv1/bin/python train.py --experiment_name "GAN_CLS_INT_BIRDS_emb" --num_epochs 601 --log_interval 200 --ngpu 1 --dataset birds --num_workers 8 --wandb=True --cls=True --inter=True

/home/hlcv_team047/miniconda3/envs/myenv1/bin/python train.py --experiment_name "GAN_INT_BIRDS_emb" --num_epochs 601 --log_interval 200 --ngpu 1 --dataset birds --num_workers 8 --wandb=True --cls=False --inter=True

/home/hlcv_team047/miniconda3/envs/myenv1/bin/python train.py --experiment_name "GAN_CLS_BIRDS_emb" --num_epochs 601 --log_interval 200 --ngpu 1 --dataset birds --num_workers 8 --wandb=True --cls=True --inter=False


#/home/hlcv_team019/miniconda3/envs/myenv1/bin/python train.py --experiment_name "GAN_CLS_INT_FLOWERS" --num_epochs 600 --log_interval 100 --ngpu 1 --dataset flowers --num_workers 8 --wandb=True --cls=True --inter=True

