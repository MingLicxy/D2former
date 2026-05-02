from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2

# 引入医学图像数据读取和预处理方法
#from .amir_data import dataIO, transformData
#io=dataIO() 
#transform = transformData() 

import torch

# 加噪函数（最终版）
import torch

def add_mixed_noise_tensor(img_lq, noise_level, seed=None):
    """
    添加混合噪声（高斯 + 条纹）
    img_lq: torch.Tensor, shape [C, H, W] 或 [B, C, H, W]
    noise_level: 1 或 2
    seed: int, 可选，设置随机种子以复现结果
    """
    if seed is not None:
        torch.manual_seed(seed)

    if img_lq.dim() == 3:
        img_lq = img_lq.unsqueeze(0)  # [1, C, H, W]

    B, C, H, W = img_lq.size()

    # ---------- Gaussian Noise ----------
    if noise_level == 1:
        sigma = 10 / 255.0
    elif noise_level == 2:
        sigma = 15 / 255.0
    else:
        raise ValueError("noise_level must be 1 or 2")
    
    gauss = torch.randn_like(img_lq) * sigma
    img_lq = img_lq + gauss

    # ---------- Stripe Noise ----------
    for b in range(B):
        for c in range(C):

            if noise_level == 1:
                # Level 1: 仅列条纹 ±4
                col_amp = 4 / 255.0
                col_bias = (torch.rand(1, W) * 2 - 1) * col_amp
                stripe = col_bias.expand(H, W)

            elif noise_level == 2:
                # Level 2: 列 ±5%，行 ±1%
                col_amp = 5 / 255.0
                row_amp = 2 / 255.0

                col_bias = (torch.rand(1, W) * 2 - 1) * col_amp
                row_bias = (torch.rand(H, 1) * 2 - 1) * row_amp

                stripe = col_bias.expand(H, W) + row_bias.expand(H, W)

            img_lq[b, c] += stripe

    return img_lq.squeeze(0) if B == 1 else img_lq


import torch


def add_poisson_gaussian_stripe_noise_tensor(img_lq, noise_level=2, seed=None):
    """
    添加混合噪声：Poisson-Gaussian + 条纹
    Args:
        img_lq: torch.Tensor, shape [C,H,W] 或 [B,C,H,W]，值范围 [0,1] 或 [0,255]，建议先归一化到 [0,1]
        noise_level: int, 1 或 2，表示噪声等级
        peak: int, 泊松噪声峰值参数，控制泊松噪声强度
        seed: int, 可选，设置随机种子以复现结果
    Returns:
        torch.Tensor, 同原输入形状，添加噪声后的图像
    """
    if seed is not None:
        torch.manual_seed(seed)

    if img_lq.dim() == 3:
        img_lq = img_lq.unsqueeze(0)  # [1,C,H,W]

    B, C, H, W = img_lq.size()

    # 确保图像在 0-1 范围
    img = img_lq.clone()
    if img.max() > 1.0:
        img = img / 255.0

    # ---------- Poisson-Gaussian Noise ----------
    ########### 这里不管噪声等级，先固定为 peak=30, sigma=10 ###########
    for b in range(B):
        for c in range(C):
            # 模拟泊松噪声
            poisson = torch.poisson(img[b, c] * 40) / 40  
            # 模拟高斯噪声
            sigma = 10 / 255.0 
            gaussian = torch.randn_like(img[b, c]) * sigma
            img[b, c] = img[b, c] + poisson + gaussian

    # ---------- Stripe Noise ----------
    for b in range(B):
        for c in range(C):
            if noise_level == 1:
                # Level 1: 仅列条纹 ±4/255
                col_amp = 4 / 255.0
                col_bias = (torch.rand(1, W) * 2 - 1) * col_amp
                stripe = col_bias.expand(H, W)
            else:
                # Level 2: 列 ±5/255，行 ±2/255
                col_amp = 5 / 255.0
                row_amp = 2 / 255.0
                col_bias = (torch.rand(1, W) * 2 - 1) * col_amp
                row_bias = (torch.rand(H, 1) * 2 - 1) * row_amp
                stripe = col_bias.expand(H, W) + row_bias.expand(H, W)
            img[b, c] += stripe

    # 裁剪到 [0,1]
    img = torch.clamp(img, 0.0, 1.0)

    return img.squeeze(0) if B == 1 else img



#TODO 用于处理成对图像数据集（Real Denoising）
# 训练集，测试集和验证集的数据加载
class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend'] 

        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        # 获取文件夹路径
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        # 根据指定文件夹生成图像路径
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        ############################## TODO 读取图像数据的核心代码 ##############################
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt') # 读取图像字节数据
        try:
            img_gt = imfrombytes(img_bytes, float32=True) #BUG 将字节数据转化为图像
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        ######################################### END #########################################

        ############################## TODO 特定格式医学图像的读取（读取结果是Tensor） ##############################
        # gt_path = self.paths[index]['gt_path']
        # try:
        #     img_gt = io.load(gt_path) # [1, 256, 256]
        #     #print("##############################################", img_gt.shape) #Tensor
        # except:
        #     raise Exception("gt path {} not working".format(gt_path))
        
        # if isinstance(img_gt, torch.Tensor): # 针对.bin解码出的 [1, 256, 256] Tensor
        #    img_gt = img_gt.permute(1, 2, 0).cpu().numpy() # [256, 256, 1] Tensor转换为Numpy
        # else: # 针对.nii解码出的 [512, 512] 的 array
        #    img_gt = np.expand_dims(img_gt, axis=-1) # [256, 256, 1]
        

        # lq_path = self.paths[index]['lq_path']
        # try:
        #     img_lq = io.load(lq_path) # [1, 256, 256] [C,H,W]
        #     #print("##############################################", img_lq.shape)
        # except:
        #     raise Exception("lq path {} not working".format(lq_path))
        
        # if isinstance(img_lq, torch.Tensor):
        #    img_lq = img_lq.permute(1, 2, 0).cpu().numpy()
        # else: 
        #    img_lq = np.expand_dims(img_lq, axis=-1)
        ######################################### END #########################################


        # augmentation for training,下面用到的方法都接收Numpy数组
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding 当前尺度小于gt_size才需要padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        ########################### BUG  BGR to RGB, HWC to CHW, numpy to tensor ###########################
        img_gt, img_lq = img2tensor([img_gt, img_lq],   
                                    bgr2rgb=True,
                                    float32=True)
        

        #TODO normalize（不同模态图像的归一化过程不同）
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        #TODO 用normalize，对应train里要用denormalize ["CT", "MRI", "PET"]
        #img_lq = transform.normalize(img_lq, "CT")
        #img_gt = transform.normalize(img_gt, "CT")
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


################################ TODO 加噪数据处理 ################################ 
class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt
        
        #################################################################
        if self.opt['phase'] == 'train':
            #BUG 加噪关键参数
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            #self.noise_level = opt['noise_level']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
            #self.noise_level_test = opt['noise_level_test']
        #################################################################
        

        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = paired_paths_from_folder(
                [self.gt_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)
            # self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()   # NumPy ndarray


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)

            ############################### 原有加高斯噪声 ##################################
            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise) # PyTorch tensor
            ###########################################################################
            
            ############################### 训练阶段加不同等级的混合噪声 ##################################
            # img_lq = add_mixed_noise_tensor(img_lq, self.noise_level)
            #img_lq = add_poisson_gaussian_stripe_noise_tensor(img_lq, self.noise_level)

        else:     
            ############################### 验证原有加高斯噪声 ##################################      
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()
            
            #img_lq = add_mixed_noise_tensor(img_lq, self.noise_level_test)
            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)
            ###########################################################################

            ############################### 训练阶段加不同等级的混合噪声 ##################################
            #img_gt, img_lq = img2tensor([img_gt, img_lq],bgr2rgb=False,float32=True)
            # img_lq = add_mixed_noise_tensor(img_lq, self.noise_level_test)
            #img_lq = add_poisson_gaussian_stripe_noise_tensor(img_lq, self.noise_level_test)


        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
