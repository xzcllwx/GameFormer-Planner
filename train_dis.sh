#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
# CFG=$1                                               #
                                              #
# -------------------------------------------------- #
# GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}


# python -m torch.distributed.launch train_predictor_dis.py \
# --train_set /root/data/alstar/nuplan/dataset/nuplan-v1.1/splits/mini_process \
# --valid_set /root/data/alstar/nuplan/dataset/nuplan-v1.1/splits/val_process \
# --batch_size 192 \
# --train_epochs 100 \

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=1 \
    --node_rank=${RANK} \
    train_predictor_dis.py \
    --train_set /root/xzcllwx_ws/nuplan_dataset_process/train_4M \
    --valid_set /root/xzcllwx_ws/nuplan_dataset_process/val_process \
    --batch_size 256 \
    --train_epochs 30 \
    --name Exp3 \
    --workers=8 \
    --resume /root/xzcllwx_ws/GameFormer-Planner/training_log/Exp2/model_epoch_28_valADE_1.1344.pth 