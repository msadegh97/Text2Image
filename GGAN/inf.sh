#!/usr/bin/env bash

cd /home/hlcv_team019/Text2Image/WGAN/
wandb login 7a44e6f35f9bf51e15cefc85c9c65093fc9c5d87

/home/hlcv_team019/miniconda3/envs/myenv1/bin/python inference.py --dataset flowers --num_workers 10 "./models/cls_WGAN_flowers/WGAN_gen_190.pth.pth"
