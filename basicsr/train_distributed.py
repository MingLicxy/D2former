import argparse
import datetime
import logging
import math
import random
import time
import torch

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

# Xformer 主目录
project_dir = os.path.expanduser('~/Xformer-main')

# 插入 sys.path 最前面，让 Python 优先查找这个目录
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# 测试是否生效
try:
    import basicsr
    #print("Using basicsr from:", basicsr.__file__)
except ModuleNotFoundError:
    print("basicsr not found!")

from tqdm import tqdm
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np

# 解析训练或测试的选项配置
def parse_options(is_train=True):
    # 对应三个命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'], # 一般采用pytorch
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # 分布式训练 distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)
    
    # 返回进程排名与总进程数（所有节点上的进程数）
    opt['rank'], opt['world_size'] = get_dist_info() 

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt

# 初始化日志
def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        #TODO 指定日志保存路径：/home/caoxinyu/UNet-based/Xformer-main/tb_logger
        tb_logger = init_tb_logger(log_dir=osp.join('/home/caoxinyu/Xformer-main/tb_logger', opt['name']))
    return logger, tb_logger

# 创建训练/验证数据加载器
def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])
            
            #TODO (600*100)/(8*2)=3750，不论是否开启分布式训练
            # opt['world_size']只与分布式训练采用的GPU数量有关，与.yaml中的num_gpu无关
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter']) # 300000
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch)) # 300000/3750=80

            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)
    import torch.distributed as dist
    rank, world_size = get_dist_info()

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #TODO automatic resume ..自动恢复训练
    state_folder_path = '/home/caoxinyu/UNet-based/Xformer-main/experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary 手动设置从哪一点恢复训练
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            #BUG
            mkdir_and_rename(osp.join('/home/caoxinyu/UNet-based/Xformer-main/tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    #TODO create train and validation dataloaders 获取所有数据加载器
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    #TODO create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        #################################### 创建模型 ####################################
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher 有关数据预取
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):

    iters = opt['datasets']['train'].get('iters') # [300000]
    
    #TODO batch_size与mini_batch_sizes  gt_size与mini_gt_sizes的关系？
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')

    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']
    
    ############################################# 训练过程 #############################################
    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        
        # 进度条
        with tqdm(total=total_iters, desc=f'Epoch {epoch}/{total_epochs}', unit='iter') as pbar:
          while train_data is not None:
              data_time = time.time() - data_time

              current_iter += 1
              if current_iter > total_iters:
                  break
              # update learning rate
              model.update_learning_rate(
                  current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            
              ##################### ------Progressive learning ------ #####################
              j = ((current_iter>groups) !=True).nonzero()[0]
              if len(j) == 0:
                  bs_j = len(groups) - 1
              else:
                  bs_j = j[0]

              mini_gt_size = mini_gt_sizes[bs_j] # 固定 16
              mini_batch_size = mini_batch_sizes[bs_j]
            
              if logger_j[bs_j]:
                  #TODO 
                  logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size, mini_batch_size*torch.cuda.device_count())) 
                  logger_j[bs_j] = False

              lq = train_data['lq']
              gt = train_data['gt']
              
              #TODO mini_batch_size与batch_size都考虑的是单个GPU（随机采样）
              if mini_batch_size < batch_size:
                  if opt['dist']:
                      torch.manual_seed(current_iter)   # 保证所有 rank 一致
                  indices = torch.randperm(batch_size)[:mini_batch_size]
                  indices = indices.to(lq.device)
                  lq = lq[indices]
                  gt = gt[indices]
              #TODO 在固定尺度训练过程中不需要（随机裁剪）
              if mini_gt_size < gt_size:
                  if opt['dist']:
                     torch.manual_seed(current_iter)
                  x0 = int((gt_size - mini_gt_size) * torch.rand(1).item())
                  y0 = int((gt_size - mini_gt_size) * torch.rand(1).item())
                  x1 = x0 + mini_gt_size
                  y1 = y0 + mini_gt_size
                  lq = lq[:,:,x0:x1,y0:y1]
                  gt = gt[:,:,x0*scale:x1*scale,y0*scale:y1*scale] # scale
              ##############################################################################

              # 将数据注入模型
              model.feed_train_data({'lq': lq, 'gt':gt})

              #TODO 模型训练
              model.optimize_parameters(current_iter)

              iter_time = time.time() - iter_time
              # log
              if current_iter % opt['logger']['print_freq'] == 0:
                  log_vars = {'epoch': epoch, 'iter': current_iter}
                  log_vars.update({'lrs': model.get_current_learning_rate()})
                  log_vars.update({'time': iter_time, 'data_time': data_time})
                  log_vars.update(model.get_current_log())
                  msg_logger(log_vars)

              # save models and training states
              if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                  if opt['dist']:
                      dist.barrier()   # 所有卡同步
                  if rank == 0:
                      logger.info('Saving models and training states.')
                      model.save(epoch, current_iter)
                  if opt['dist']:
                      dist.barrier()   # 等 rank0 保存完成
              ################################ TODO 验证 validation ################################
              if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                  rgb2bgr = opt['val'].get('rgb2bgr', True)
                  # wheather use uint8 image to compute metrics
                  use_image = opt['val'].get('use_image', True) # 默认true
                  
                  #TODO 模型验证
                  if opt['dist']:
                      dist.barrier()
                  # ======================================================
                  # ✅ ALL RANKS ENTER VALIDATION (CRITICAL FIX)
                  # ======================================================
                  if rank == 0:
                      model.validation(
                          val_loader,
                          current_iter,
                          tb_logger,
                          opt['val']['save_img'],
                          rgb2bgr,
                          use_image
                      )
                  else:
                      # ❗关键：其他 rank 执行“等价路径”，避免 DDP 卡死
                      # 不做 IO / log / save，但必须调用同一函数结构
                      model.validation(
                          val_loader,
                          current_iter,
                          None,
                          False,
                          rgb2bgr,
                          use_image
                      )
                  if opt['dist']:
                      dist.barrier()


              data_time = time.time()
              iter_time = time.time()
              train_data = prefetcher.next()

              # 更新进度条
              pbar.update(1)

        # end of iter
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')

    if opt['dist']:
        dist.barrier()
    if rank == 0:
        model.save(epoch=-1, current_iter=-1)
    if opt['dist']:
        dist.barrier()  # -1 stands for the latest

    if opt.get('val') is not None:
        if opt['dist']:
            dist.barrier()
        if rank == 0:
            model.validation(
                val_loader,
                current_iter,
                tb_logger,
                opt['val']['save_img'],
                opt['val'].get('rgb2bgr', True),
                opt['val'].get('use_image', True)
            )
        else:
            model.validation(
                val_loader,
                current_iter,
                None,
                False,
                opt['val'].get('rgb2bgr', True),
                opt['val'].get('use_image', True)
            )
        if opt['dist']:
            dist.barrier()
        
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
