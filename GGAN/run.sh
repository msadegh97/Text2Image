#!/usr/bin/env bash

cd /home/hlcv_team047/Text2Image/GGAN/

/home/hlcv_team019/miniconda3/envs/myenv1/bin/python train.py --experiment_name "GGAN_BIRDS_EMB3" --num_epochs 601 --log_interval 200 --ngpu 1 --lr 0.0001 --dataset birds --num_workers 8 --wandb=True --cls=False
#/home/hlcv_team019/miniconda3/envs/myenv1/bin/python train.py --experiment_name "GGAN_INT_FLOWERS" --num_epochs 600 --log_interval 100 --ngpu 1 --dataset flowers --num_workers 8 --wandb=True --cls=False

