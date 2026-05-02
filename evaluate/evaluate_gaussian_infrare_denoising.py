import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
import argparse
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures
import utils

###TODO 需要先执行test_gaussian_infrare_denoising.py，得到预测结果之后才能计算指标
################################ 用于计算PSNR与SSIM ################################

# 加载图像计算指标
def proc(filename):
    tar,prd = filename # filename是一个元组
    tar_img = utils.load_gray_img(tar)
    prd_img = utils.load_gray_img(prd)
        
    PSNR = utils.calculate_psnr(tar_img, prd_img)
    SSIM = utils.calculate_ssim(tar_img, prd_img)
    #LPIPS = utils.calculate_lpips(tar_img, prd_img)
    #CLIP_IQA = utils.calculate_clip_iqa(tar_img, prd_img)
    #MUSIQ = utils.calculate_musiq(prd_img)
    return PSNR, SSIM  

parser = argparse.ArgumentParser(description='Gasussian Infrare Denoising')

#TODO 决定加入噪声强度
parser.add_argument('--sigmas', default='15,25,50', type=str, help='Sigma values')

args = parser.parse_args()

sigmas = np.int_(args.sigmas.split(',')) # [15,25,50]

#TODO 列出所有测试集名称
datasets = ['IR700_test', 'DLS-NUC-100', 'IR100', 'ESPOL-FIR', 'Flir']

for dataset in datasets:

    gt_path = os.path.join('/home/caoxinyu/UNet-based/infrare_data/test', dataset)

    gt_list = natsorted(
                          glob(os.path.join(gt_path, '*.png')) 
                        + glob(os.path.join(gt_path, '*.jpg'))
                        + glob(os.path.join(gt_path, '*.bmp'))
                        )

    assert len(gt_list) != 0, "Target files not found"

    for sigma_test in sigmas:
        # 执行test_gaussian_infrare_denoising.py得到的预测结果文件夹
        file_path = os.path.join('/home/caoxinyu/UNet-based/Xformer-main/results/INRDSUNet_INR321', dataset, str(sigma_test))

        path_list = natsorted(
                              glob(os.path.join(file_path, '*.png')) 
                            + glob(os.path.join(file_path, '*.jpg'))
                            + glob(os.path.join(file_path, '*.bmp'))
                            )
                            
        assert len(path_list) != 0, "Predicted files not found"

        psnr, ssim = [], []
        img_files =[(i, j) for i,j in zip(gt_list,path_list)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
                psnr.append(PSNR_SSIM[0])
                ssim.append(PSNR_SSIM[1])

        avg_psnr = sum(psnr)/len(psnr)
        avg_ssim = sum(ssim)/len(ssim)

        # print('For {:s} dataset Noise Level {:d} PSNR: {:f}\n'.format(dataset, sigma_test, avg_psnr))
        print('For {:s} dataset PSNR: {:.2f} SSIM: {:.4f}\n'.format(dataset, avg_psnr, avg_ssim))
