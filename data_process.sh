# echo "start"
# python /root/xzcllwx_ws/GameFormer-Planner/data_process.py \
#     --data_path /root/data/alstar/nuplan/dataset/nuplan-v1.1/splits/train \
#     --map_path /root/data/alstar/nuplan/dataset/maps \
#     --save_path /root/data/alstar/nuplan/dataset/nuplan-v1.1/splits/train_1M \  
#     --total_scenarios 1000000 \
#     --shuffle_scenarios True

# echo "__________train done___________"

echo "start"
python /root/xzcllwx_ws/GameFormer-Planner/data_process_dis.py \
    --data_path /root/xzcllwx_ws/nuplan_dataset_process/train_half \
    --map_path /root/xzcllwx_ws/nuplan_dataset_process/maps \
    --save_path /root/xzcllwx_ws/nuplan_dataset_process/train_3M \
    --total_scenarios 2000000 
    # --shuffle_scenarios True

echo "__________data process done___________"

# echo "start"
# python /root/xzcllwx_ws/GameFormer-Planner/data_process.py \
#     --data_path /root/data/alstar/nuplan/dataset/nuplan-v1.1/splits/mini_1 \
#     --map_path /root/data/alstar/nuplan/dataset/maps \
#     --save_path /root/data/alstar/nuplan/dataset/nuplan-v1.1/splits/mini_1_process \
#     --total_scenarios 1
#     # --shuffle_scenarios True

# echo "__________train done___________"