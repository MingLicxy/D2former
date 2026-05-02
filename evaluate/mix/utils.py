## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import cv2
import math
#import lpips
import numpy as np
import torch
import torchvision.transforms as transforms
import clip
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
# def calculate_clip_iqa(img1, img2, border=0):
#     """
#     计算 CLIP-IQA 分数，图像输入为 numpy 数组，范围 [0, 255]，支持灰度和RGB图。
#     越高表示越接近参考图像。
#     """
#     if not img1.shape == img2.shape:
#         raise ValueError("Input images must have the same dimensions.")

#     h, w = img1.shape[:2]
#     img1 = img1[border:h-border, border:w-border]
#     img2 = img2[border:h-border, border:w-border]

#     def to_clip_tensor(img):
#         if img.ndim == 2:  # 灰度图
#             img = np.stack([img] * 3, axis=-1)
#         elif img.ndim == 3 and img.shape[2] == 1:
#             img = np.repeat(img, 3, axis=2)
#         img_pil = Image.fromarray(img.astype(np.uint8))
#         return preprocess_clip(img_pil).unsqueeze(0).to(device)

#     img1_tensor = to_clip_tensor(img1)
#     img2_tensor = to_clip_tensor(img2)

#     with torch.no_grad():
#         feat1 = clip_model.encode_image(img1_tensor).float()
#         feat2 = clip_model.encode_image(img2_tensor).float()
#         feat1 /= feat1.norm(dim=-1, keepdim=True)
#         feat2 /= feat2.norm(dim=-1, keepdim=True)
#         similarity = (feat1 * feat2).sum().item()

#     return similarity


# 初始化 LPIPS 网络（建议在外部只初始化一次）
# lpips_fn = lpips.LPIPS(net='vgg')  # 可选 'vgg' 或 'squeeze'
# def calculate_lpips(img1, img2, border=0):
#     """
#     计算两张图像的 LPIPS 距离，图像输入为 numpy 数组，范围 [0, 255]，支持灰度图和RGB图。
#     """
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')

#     h, w = img1.shape[:2]
#     img1 = img1[border:h-border, border:w-border]
#     img2 = img2[border:h-border, border:w-border]

#     def preprocess(img):
#         img = img.astype(np.float32) / 255.0  # [0,1]

#         # 如果是灰度图：H x W 或 H x W x 1，扩展为 H x W x 3
#         if img.ndim == 2:
#             img = np.stack([img] * 3, axis=-1)
#         elif img.ndim == 3 and img.shape[2] == 1:
#             img = np.repeat(img, 3, axis=2)
#         elif img.ndim == 3 and img.shape[2] != 3:
#             raise ValueError("Only 1-channel or 3-channel images are supported.")

#         img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
#         img = img * 2 - 1  # [0,1] -> [-1,1]
#         return img

#     img1_tensor = preprocess(img1)
#     img2_tensor = preprocess(img2)

#     with torch.no_grad():
#         lpips_dist = lpips_fn(img1_tensor, img2_tensor)

#     return lpips_dist.item()



def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

#TODO 加载灰度图像
def load_gray_img(filepath):
    # 可以读取.png .jpg .bmp格式的红外图像
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)
