import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

# ===============================
#   噪声构造函数
# ===============================

def add_poisson_gaussian_noise(img, peak=30, sigma=10):
    """
    添加 Poisson-Gaussian 噪声
    Args:
        img: 输入图像，float32 类型，范围 0-255
        peak: 用于归一化泊松噪声的峰值参数
        sigma: 高斯噪声标准差
    Returns:
        img_noisy: 添加噪声后的图像
    """
    img_norm = img / 255.0
    poisson_noise = np.random.poisson(np.minimum(img_norm * peak, peak)) / peak
    gaussian_noise = np.random.randn(*img.shape) * (sigma / 255.0)
    img_noisy = img_norm + poisson_noise + gaussian_noise
    img_noisy = np.clip(img_noisy * 255.0, 0, 255).astype(np.float32)
    return img_noisy

def add_row_stripe(img, amplitude):
    """添加行方向条纹"""
    H, W = img.shape
    bias = np.random.uniform(-amplitude, amplitude, H)
    stripe = np.repeat(bias[:, np.newaxis], W, axis=1)
    return img + stripe

def add_col_stripe(img, amplitude):
    """添加列方向条纹"""
    H, W = img.shape
    bias = np.random.uniform(-amplitude, amplitude, W)
    stripe = np.repeat(bias[np.newaxis, :], H, axis=0)
    return img + stripe

# ===============================
#   三种噪声等级
# ===============================

def apply_noise_level1(img):
    """Level 1: 轻度噪声"""
    out = add_poisson_gaussian_noise(img, peak=50, sigma=10)
    out = add_col_stripe(out, amplitude=4)
    return out

def apply_noise_level2(img):
    """Level 2: 中度噪声"""
    out = add_poisson_gaussian_noise(img, peak=50, sigma=10)
    out = add_row_stripe(out, amplitude=2)
    out = add_col_stripe(out, amplitude=5)
    return out

# ===============================
#   单图像处理函数
# ===============================

def process_single_image(input_path, output_dir, levels=[1,2]):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{input_path}")
    
    img = img.astype(np.float32)
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    
    result = {'filename': filename, 'levels': []}
    
    if 1 in levels:
        noisy1 = apply_noise_level1(img)
        noisy1 = np.clip(noisy1, 0, 255).astype(np.uint8)
        output_path1 = os.path.join(output_dir, f"{name}_noisy_level1{ext}")
        cv2.imwrite(output_path1, noisy1)
        result['levels'].append({'level':1, 'path':output_path1})
    
    if 2 in levels:
        noisy2 = apply_noise_level2(img)
        noisy2 = np.clip(noisy2, 0, 255).astype(np.uint8)
        output_path2 = os.path.join(output_dir, f"{name}_noisy_level2{ext}")
        cv2.imwrite(output_path2, noisy2)
        result['levels'].append({'level':2, 'path':output_path2})
    
    return result

# ===============================
#   批量处理函数
# ===============================

def batch_process(input_dir, output_dir, levels=[1,2]):
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm', '.ppm')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
    image_files.sort()
    
    if not image_files:
        print(f"错误：在 {input_dir} 中没有找到有效图像文件")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"找到 {len(image_files)} 张图像待处理")
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print(f"生成噪声等级：{levels}")
    print("-"*60)
    
    success_count = 0
    fail_count = 0
    
    for filename in tqdm(image_files, desc="添加噪声中"):
        input_path = os.path.join(input_dir, filename)
        try:
            result = process_single_image(input_path, output_dir, levels)
            success_count += 1
            level_info = ", ".join([f"Level{l['level']}" for l in result['levels']])
            print(f"✓ {filename} -> {level_info}")
        except Exception as e:
            fail_count += 1
            print(f"✗ {filename} 处理失败：{str(e)}")
    
    print("\n" + "="*60)
    print("处理完成统计：")
    print(f"  成功：{success_count} 张")
    print(f"  失败：{fail_count} 张")
    print(f"  总计：{len(image_files)} 张")
    print(f"噪声图像保存位置：{output_dir}")

# ===============================
#   主函数
# ===============================

def main():
    parser = argparse.ArgumentParser(description='批量添加 Poisson-Gaussian + 条纹噪声')
    parser.add_argument('--input','-i', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--output','-o', type=str, required=True, help='输出噪声图像文件夹路径')
    parser.add_argument('--levels','-l', type=int, nargs='+', default=[1,2], help='噪声等级，可选 1/2')
    parser.add_argument('--seed','-s', type=int, default=None, help='随机种子（可选）')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"已设置随机种子：{args.seed}")
    
    valid_levels = [1,2]
    levels = [l for l in args.levels if l in valid_levels]
    if not levels:
        print(f"错误：无效噪声等级 {args.levels}")
        return
    
    batch_process(args.input, args.output, levels)

if __name__ == '__main__':
    main()

# 使用示例（命令行）：
# python /home/caoxinyu/UNet-based/Xformer-main/utils/add_Mix.py -i /home/caoxinyu/UNet-based/Xformer-main/inputs/DLS-NUC-100 -o /home/caoxinyu/UNet-based/Xformer-main/inputs/DLS-NUC-100_noise -l 1 2 --seed 42

