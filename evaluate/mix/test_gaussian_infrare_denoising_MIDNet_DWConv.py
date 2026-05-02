import numpy as np
import os
import argparse
from tqdm import tqdm
import sys

import torch.nn as nn
import torch
import torch.nn.functional as F

from pdb import set_trace as stx
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

# from basicsr.models.archs.x_former_arch import Xformer
from basicsr.models.archs.MIDNet_dwconv_arch import INRDSUNet_DWConv
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils



########### 第一步：加噪函数 ###########
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

###TODO 获取预测结果，并储存于指定文件夹
################################ 用于获取可视化结果 ################################

parser = argparse.ArgumentParser(description='Gasussian Infrare Denoising')

parser.add_argument('--input_dir', default='/home/caoxinyu/infrare_data/test', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/home/caoxinyu/Xformer-main/results/MIDNet_DWConv', type=str, help='Directory for results')

########### 第二步：添加输入参数 ###########
parser.add_argument('--level', default='2', type=str, help='Noise Levels, 1 or 2')

args = parser.parse_args()

####### Load model options #######
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

#TODO 模型参数与训练阶段保持不变
#TODO 改动点三
opt_str = r"""
  type: INRDSUNet_DWConv   # 注意：Difformer不是文件名是类名
  inp_channels: 1
  out_channels: 1
  dim: 48
  num_blocks: [4,6,6,8]   # [2,4,4,6]
  num_refinement_blocks: 4
  heads: [1, 2, 4, 8]
  ffn_expansion_factor: 2.66
  bias: False
  LayerNorm_type: BiasFree
  dual_pixel_task: False
"""
opt = yaml.safe_load(opt_str)
network_type = opt.pop('type')
##########################################

########### 第三步：获取噪声级别 ###########
level = np.int_(args.level)

factor = 8 #TODO

datasets = ['DLS-NUC-100', 'ESPOL-FIR', 'Flir', 'IR100', 'IR700_test']

print("Compute results for noise level", level)

# 创建模型
model_restoration = INRDSUNet_DWConv(**opt)    

########### 第四步：加载权重 ###########
if level == 1:
    #TODO 根据log文件自由选取
    weights = '/home/caoxinyu/UNet-based/Xformer-main/experiments/GaussianGrayDenoising_DFormerFPNLevel1/models/net_g_280000.pth'
elif level == 2:
    weights = '/home/caoxinyu/Xformer-main/experiments/GaussianGrayDenoising_MIDNetMixLevel2DWConv/models/net_g_264000.pth'


checkpoint = torch.load(weights)
model_restoration.load_state_dict(checkpoint['params'])

print("===>Testing using weights: ",weights)
print("------------------------------------------------")
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

for dataset in datasets:
    # 输入图像路径：'/home/caoxinyu/UNet-based/infrare_data/test/DLS-NUC-100'
    inp_dir = os.path.join(args.input_dir, dataset) 
    files = natsorted(
                          glob(os.path.join(inp_dir, '*.png')) 
                        + glob(os.path.join(inp_dir, '*.jpg'))
                        + glob(os.path.join(inp_dir, '*.bmp'))
                        )
    # 输出图像路径: '/home/caoxinyu/UNet-based/Xformer-main/results/Difformer/DLS-NUC-100/15'
    ########### 第五步：拼接路径 ###########
    result_dir_tmp = os.path.join(args.result_dir, dataset, str(level)) 
    os.makedirs(result_dir_tmp, exist_ok=True)

    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            # 加载图像 [H, W, C]

            ########### 第六步：添加噪声 ###########
            #####################################################
            img = np.float32(utils.load_gray_img(file_))/255. 

            # 先转换为 torch.Tensor 并调整维度 [C, H, W]
            img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

            # TODO: 改变测试集中随机加噪的噪声种子对于最终测试结果应当是负面影响
            # 使用新的加噪函数（内部会设置 torch 随机种子）
            img = add_poisson_gaussian_stripe_noise_tensor(img, noise_level=level, seed=0)

            # 添加 batch 维度并移至 GPU [1, C, H, W]
            input_ = img.unsqueeze(0).cuda()
            ######################################################

            #TODO 如果图像尺寸不是 8 的倍数，则进行填充（配合窗口大小）
            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
            
            ################################## TODO 注意Oformer返回的数组 ##################################
            restored = model_restoration(input_)
            
            #TODO 去掉填充
            # Unpad images to original dimensions 
            restored = restored[:,:,:h,:w]

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
            utils.save_gray_img(save_file, img_as_ubyte(restored))