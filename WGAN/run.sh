#!/usr/bin/env bash

cd /home/hlcv_team019/Text2Image/
wandb login 7a44e6f35f9bf51e15cefc85c9c65093fc9c5d87
/home/hlcv_team019/miniconda3/envs/myenv1/bin/python train.py --log_interval 50 --ngpu 1 --dataset birds --num_workers 10
