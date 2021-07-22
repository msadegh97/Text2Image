#!/usr/bin/env bash

cd /home/hlcv_team047/Text2Image/GAN/
# wandb login 7a44e6f35f9bf51e15cefc85c9c65093fc9c5d87

/home/hlcv_team047/miniconda3/envs/myenv1/bin/python inference.py --dataset birds --num_workers 10 --gen_dir="models/GAN_CLS_INT_BIRDS_emb/GAN_CLS_INT_BIRDS_emb_gen_600.pth" --experiment_name "GAN_CLS_INT_BIRDS_emb"
/home/hlcv_team047/miniconda3/envs/myenv1/bin/python inference.py --dataset birds --num_workers 10 --gen_dir="models/GAN_INT_BIRDS_emb/GAN_INT_BIRDS_emb_gen_600.pth" --experiment_name "GAN_INT_BIRDS_emb"
/home/hlcv_team047/miniconda3/envs/myenv1/bin/python inference.py --dataset birds --num_workers 10 --gen_dir="models/GAN_CLS_BIRDS_emb/GAN_CLS_BIRDS_emb_gen_600.pth" --experiment_name "GAN_CLS_BIRDS_emb"
