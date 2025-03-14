import os
import csv
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from GameFormer.predictor import GameFormer
from torch.utils.data import DataLoader
from GameFormer.train_utils import *
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

def is_main_process():
    return dist.get_rank() == 0

def train_epoch(data_loader, model, optimizer):
    epoch_loss = []
    epoch_metrics = []
    model.train()
    with tqdm(data_loader, desc=f"Training—{str(dist.get_rank())}", unit="batch") as data_epoch:
        for batch in data_epoch:
            # prepare data
            inputs = {
                'ego_agent_past': batch[0].to(args.device, non_blocking=True),
                'neighbor_agents_past': batch[1].to(args.device, non_blocking=True),
                'map_lanes': batch[2].to(args.device, non_blocking=True),
                'map_crosswalks': batch[3].to(args.device, non_blocking=True),
                'route_lanes': batch[4].to(args.device, non_blocking=True)
            }

            ego_future = batch[5].to(args.device, non_blocking=True)
            neighbors_future = batch[6].to(args.device, non_blocking=True)
            neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

            # call the mdoel
            optimizer.zero_grad()
            level_k_outputs, ego_plan = model(inputs)
            loss, results = level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)
            prediction = results[:, 1:]
            plan_loss = planning_loss(ego_plan, ego_future)
            loss += plan_loss

            # loss backward
            loss.backward()

            # if is_main_process():
            #     ls = [name for name,para in model.named_parameters() if para.grad==None]
            #     print(ls)

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # compute metrics
            metrics = motion_metrics(ego_plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    planningAHE, planningFHE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE]
    if is_main_process():
        logging.info(f"plannerADE: {planningADE:.4f}, plannerFDE: {planningFDE:.4f}, " +
                    f"plannerAHE: {planningAHE:.4f}, plannerFHE: {planningFHE:.4f}, " +
                    f"predictorADE: {predictionADE:.4f}, predictorFDE: {predictionFDE:.4f}\n")
        
    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, model):
    epoch_loss = []
    epoch_metrics = []
    model.eval()

    with tqdm(data_loader, desc=f"Validation—{str(dist.get_rank())}", unit="batch") as data_epoch:
        for batch in data_epoch:
           # prepare data
            inputs = {
                'ego_agent_past': batch[0].to(args.device, non_blocking=True),
                'neighbor_agents_past': batch[1].to(args.device, non_blocking=True),
                'map_lanes': batch[2].to(args.device, non_blocking=True),
                'map_crosswalks': batch[3].to(args.device, non_blocking=True),
                'route_lanes': batch[4].to(args.device, non_blocking=True)
            }

            ego_future = batch[5].to(args.device, non_blocking=True)
            neighbors_future = batch[6].to(args.device, non_blocking=True)
            neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

            # call the mdoel
            with torch.no_grad():
                level_k_outputs, ego_plan = model(inputs)
                loss, results = level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)
                prediction = results[:, 1:]
                plan_loss = planning_loss(ego_plan, ego_future)
                loss += plan_loss

            # compute metrics
            metrics = motion_metrics(ego_plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    planningAHE, planningFHE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE]
    if is_main_process():
        logging.info(f"val-plannerADE: {planningADE:.4f}, val-plannerFDE: {planningFDE:.4f}, " +
                    f"val-plannerAHE: {planningAHE:.4f}, val-plannerFHE: {planningFHE:.4f}, " +
                    f"val-predictorADE: {predictionADE:.4f}, val-predictorFDE: {predictionFDE:.4f}\n")

    return np.mean(epoch_loss), epoch_metrics


def model_training():
    # Logging
    if is_main_process():
        log_path = f"./training_log/{args.name}/"
        os.makedirs(log_path, exist_ok=True)
        initLogging(log_file=log_path+'train.log')
        logging.info("------------- {} -------------".format(args.name))
        logging.info("Batch size: {}".format(args.batch_size))
        logging.info("Learning rate: {}".format(args.learning_rate))
        logging.info("Use device: {}".format(args.device))
        tensorboard_writer = SummaryWriter(log_dir=log_path)
        
    # set seed
    set_seed(args.seed)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    train_set = DrivingData(args.train_set + '/*.npz', args.num_neighbors)
    valid_set = DrivingData(args.valid_set + '/*.npz', args.num_neighbors)

    train_sampler = DistributedSampler(train_set, shuffle=True)
    # train_loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=args.batch_size, sampler=train_sampler
    # )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=args.workers, shuffle=False,
        pin_memory=True, prefetch_factor=16, persistent_workers=True
    )
    # train_loader = DataLoader(
    #     train_set, batch_size=batch_size, sampler=train_sampler, num_workers=os.cpu_count(), shuffle=False,
    # )

    # valid_sampler = DistributedSampler(valid_set, shuffle=False)
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_set, batch_size=args.batch_size, sampler=valid_sampler
    # )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size*4, num_workers=args.workers, shuffle=False,
        pin_memory=True
    )
    # valid_loader = DataLoader(
    #     valid_set, batch_size=batch_size, sampler=valid_sampler, num_workers=os.cpu_count(), shuffle=False,
    # )
    if is_main_process():
        logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    # set up model
    gameformer = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels, neighbors=args.num_neighbors)
    gameformer = gameformer.to(args.device)

    start_epoch = 0
    # 加载预训练模型（如果指定）
    if args.checkpoint is not None:
        epoch_str = os.path.basename(args.checkpoint).split('model_epoch_')[1].split('_')[0]
        if args.resume:   
            # 从文件名解析epoch信息
            start_epoch = int(epoch_str)
            if is_main_process():
                logging.info(f"Start training from epoch {start_epoch+1}")

        logging.info(f"Loading pre-trained model from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        
        # 尝试直接加载模型权重
        try:
            gameformer.load_state_dict(checkpoint)
            if is_main_process():
                logging.info("Successfully loaded model weights")
        except Exception as e:
            # 如果直接加载失败，尝试提取模型状态字典（可能是保存了完整的checkpoint）
            if 'model_state_dict' in checkpoint:
                gameformer.load_state_dict(checkpoint['model_state_dict'])
                if is_main_process():
                    logging.info("Successfully loaded model weights from checkpoint state dict")
            else:
                if is_main_process():
                    logging.warning(f"Failed to load weights: {str(e)}")
        
    gameformer=torch.nn.parallel.DistributedDataParallel(
        gameformer, 
        device_ids=[args.local_rank], 
        output_device=args.local_rank,
        find_unused_parameters= args.stage == 2,
    )
    
    total_params = sum(p.numel() for p in gameformer.module.parameters())

    if args.stage == 1:
        frozen_params = 0
        if is_main_process():
            logging.info(f"stage 1：frozen parameters {frozen_params}/{total_params} ({frozen_params/total_params:.2%})")
    elif args.stage == 2:
        frozen_params = 0
        for name, param in gameformer.named_parameters():
            if 'planner' not in name:
                param.requires_grad = False
                frozen_params += param.numel()
        if is_main_process():
            logging.info(f"stage 2：frozen parameters {frozen_params}/{total_params} ({frozen_params/total_params:.2%})")
    else :
        logging.info("stage error")
        return
    
    if is_main_process():
        # for name, param in gameformer.module.named_parameters():
        #     logging.info(f"{name}: {param.requires_grad}")
        logging.info("Model Params: {}".format(total_params))
        logging.info("Start epoch: {}".format(start_epoch))
        logging.info("Training for {} epochs\n".format(train_epochs))

    # set up optimizer
    # optimizer = optim.AdamW(gameformer.parameters(), lr=args.learning_rate)
    optimizer = optim.AdamW(filter(lambda p : p.requires_grad, gameformer.parameters()), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 13, 16, 19, 22, 25, 28], gamma=0.5)

    # begin training
    for epoch in range(start_epoch, train_epochs):
        if is_main_process():
            for _ in range(start_epoch):
                scheduler.step()
            logging.info(f"Epoch {epoch+1}/{train_epochs}")
        train_loader.sampler.set_epoch(epoch)
        # valid_loader.sampler.set_epoch(epoch)
        if not args.val:
            train_loss, train_metrics = train_epoch(train_loader, gameformer, optimizer)
        dist.barrier()
        if is_main_process():
            val_loss, val_metrics = valid_epoch(valid_loader, gameformer.module)
        dist.barrier()
        if is_main_process():
            tensorboard_writer.add_scalar('Loss/train', train_loss, epoch)
            tensorboard_writer.add_scalar('Loss/val', val_loss, epoch)
            tensorboard_writer.add_scalar('Metrics/train_planningADE', train_metrics[0], epoch)
            tensorboard_writer.add_scalar('Metrics/train_planningFDE', train_metrics[1], epoch)
            tensorboard_writer.add_scalar('Metrics/train_planningAHE', train_metrics[2], epoch)
            tensorboard_writer.add_scalar('Metrics/train_planningFHE', train_metrics[3], epoch)
            tensorboard_writer.add_scalar('Metrics/train_predictionADE', train_metrics[4], epoch)
            tensorboard_writer.add_scalar('Metrics/train_predictionFDE', train_metrics[5], epoch)
            
            tensorboard_writer.add_scalar('Metrics/val_planningADE', val_metrics[0], epoch)
            tensorboard_writer.add_scalar('Metrics/val_planningFDE', val_metrics[1], epoch)
            tensorboard_writer.add_scalar('Metrics/val_planningAHE', val_metrics[2], epoch)
            tensorboard_writer.add_scalar('Metrics/val_planningFHE', val_metrics[3], epoch)
            tensorboard_writer.add_scalar('Metrics/val_predictionADE', val_metrics[4], epoch)
            tensorboard_writer.add_scalar('Metrics/val_predictionFDE', val_metrics[5], epoch)
            
            tensorboard_writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            # save to training log
            log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss, 
                'train-planningADE': train_metrics[0], 'train-planningFDE': train_metrics[1], 
                'train-planningAHE': train_metrics[2], 'train-planningFHE': train_metrics[3], 
                'train-predictionADE': train_metrics[4], 'train-predictionFDE': train_metrics[5],
                'val-planningADE': val_metrics[0], 'val-planningFDE': val_metrics[1], 
                'val-planningAHE': val_metrics[2], 'val-planningFHE': val_metrics[3],
                'val-predictionADE': val_metrics[4], 'val-predictionFDE': val_metrics[5]}

            if epoch == 0:
                with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file: 
                    writer = csv.writer(csv_file) 
                    writer.writerow(log.keys())
                    writer.writerow(log.values())
            else:
                with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file: 
                    writer = csv.writer(csv_file)
                    writer.writerow(log.values())

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        if is_main_process():
            torch.save(gameformer.module.state_dict(), f'training_log/{args.name}/model_epoch_{epoch+1}_valADE_{val_metrics[0]:.4f}.pth')
            logging.info(f"Model saved in training_log/{args.name}\n")
        # print(f"thread {args.local_rank} finished epoch {epoch+1}\n")
        dist.barrier()
        
    if is_main_process():
        tensorboard_writer.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--train_set', type=str, help='path to train data')
    parser.add_argument('--valid_set', type=str, help='path to validation data')
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--encoder_layers', type=int, help='number of encoding layers', default=3)
    parser.add_argument('--decoder_levels', type=int, help='levels of reasoning', default=2)
    parser.add_argument('--num_neighbors', type=int, help='number of neighbor agents to predict', default=10)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=20)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=32)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 1e-4)', default=1e-4)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument("--workers", type=int, default=8, help="number of workers used for dataloader")
    parser.add_argument('--stage', type=int, default=1, help='训练阶段 (1: 预测, 2: 规划)')
    parser.add_argument('--checkpoint', type=str, help='path to pre-trained model (default: None)', default=None)
    parser.add_argument('--resume', action='store_true', help='set start epoch to resume training')
    parser.add_argument('--val', action='store_true', help='only validate the model')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # distributed training
    dist.init_process_group(
        'nccl', init_method='env://'
    )
    args.local_rank = dist.get_rank()
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    args.device = device
    print(f"---------Rank {args.local_rank} is running on {args.device}---------")
    # Run
    model_training()