import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

#TODO 改动点一
from basicsr.models.archs.Ablate_Difformer_2_arch import Ablate_Difformer_2

from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils
from pdb import set_trace as stx

###TODO 获取预测结果，并储存于指定文件夹
################################ 用于获取可视化结果 ################################

parser = argparse.ArgumentParser(description='Gasussian Infrare Denoising')

parser.add_argument('--input_dir', default='/home/caoxinyu/UNet-based/infrare_data/test', type=str, help='Directory of validation images')

#TODO 改动点二
parser.add_argument('--result_dir', default='/home/caoxinyu/UNet-based/Xformer-main/results/Ablate_2', type=str, help='Directory for results')
parser.add_argument('--sigma', default='15', type=str, help='Sigma values, 15, 25, or 50')

args = parser.parse_args()

####### Load model options #######
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


#TODO 改动点三
opt_str = r"""
  type: Ablate_Difformer_2   # 单输入单输出
  inp_channels: 1
  out_channels: 1
  dim: 48
  num_blocks: [2, 4, 4]
  spatial_num_blocks: [2,4,4,6]
  num_refinement_blocks: 4
  heads: [1, 2, 4, 8]
  window_size: [16,16,16,16]
  ffn_expansion_factor: 2.66
  bias: False
  LayerNorm_type: BiasFree
  dual_pixel_task: False
"""
opt = yaml.safe_load(opt_str)
network_type = opt.pop('type')
##########################################

sigma = np.int_(args.sigma)

factor = 8 #TODO

datasets = ['DLS-NUC-100', 'ESPOL-FIR', 'Flir', 'IR100', 'IR700_test']

print("Compute results for noise level",sigma)

#TODO 改动点四： 创建模型
model_restoration = Ablate_Difformer_2(**opt)    

#TODO 改动点五： 加载权重
if sigma == 15:
    #TODO 根据log文件自由选取
    weights = '/home/caoxinyu/UNet-based/Xformer-main/experiments/GaussianGrayDenoising_SwinIRSigma15/models/net_g_latest.pth'
elif sigma == 25:
    weights = '/home/caoxinyu/UNet-based/Xformer-main/experiments/GaussianGrayDenoising_SwinIRSigma25/models/net_g_latest.pth'
else:
    weights = '/home/caoxinyu/UNet-based/Xformer-main/experiments/Ablate_DFormerSigma50_2/models/net_g_latest.pth'

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
    result_dir_tmp = os.path.join(args.result_dir, dataset, str(sigma)) 
    os.makedirs(result_dir_tmp, exist_ok=True)

    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            # 加载图像，并且归一化
            img = np.float32(utils.load_gray_img(file_))/255. 
            
            #TODO 改变测试集中随机加噪的噪声种子对于最终测试结果应当是负面影响
            np.random.seed(seed=0)  # for reproducibility

            # 向输入干净图像中加噪声
            img += np.random.normal(0, sigma/255., img.shape)

            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()

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
