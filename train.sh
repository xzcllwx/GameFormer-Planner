


python  train_predictor.py \
--train_set /root/xzcllwx_ws/nuplan_dataset_process/train_1M \
--valid_set /root/xzcllwx_ws/nuplan_dataset_process/val_process \
--batch_size 128 \
--train_epochs 30 \
--device cuda:1 \
--name Exp1 \
--workers=8
