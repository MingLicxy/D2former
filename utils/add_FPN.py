import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

# ===============================
#   条纹噪声构造函数
# ===============================

def add_gaussian_noise(img, sigma):
    """添加高斯噪声"""
    noise = np.random.randn(*img.shape) * sigma
    return img + noise

def add_row_stripe(img, amplitude):
    """添加行方向条纹"""
    H, W = img.shape
    bias = np.random.uniform(-amplitude, amplitude, H)  # 每行一个偏置
    stripe = np.repeat(bias[:, np.newaxis], W, axis=1)
    return img + stripe

def add_col_stripe(img, amplitude):
    """添加列方向条纹"""
    H, W = img.shape
    bias = np.random.uniform(-amplitude, amplitude, W)  # 每列一个偏置
    stripe = np.repeat(bias[np.newaxis, :], H, axis=0)
    return img + stripe

# ===============================
#   三种噪声等级
# ===============================

def apply_noise_level1(img):
    """Level 1: 轻度噪声"""
    out = add_gaussian_noise(img, sigma=10)
    out = add_col_stripe(out, amplitude=4)
    return out

def apply_noise_level2(img):
    """Level 2: 中度噪声"""
    out = add_gaussian_noise(img, sigma=15)
    out = add_row_stripe(out, amplitude=2)
    out = add_col_stripe(out, amplitude=5)
    return out

# ===============================
#   单图像处理函数
# ===============================

def process_single_image(input_path, output_dir, levels=[1, 2]):
    """
    处理单张图像，生成不同噪声等级的图像
    
    Args:
        input_path: 输入图像路径
        output_dir: 输出文件夹路径
        levels: 需要生成的噪声等级列表，默认 [1, 2]
    
    Returns:
        dict: 处理结果信息
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{input_path}")
    
    img = img.astype(np.float32)
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    
    result = {'filename': filename, 'levels': []}
    
    # Level 1
    if 1 in levels:
        noisy1 = apply_noise_level1(img)
        noisy1 = np.clip(noisy1, 0, 255).astype(np.uint8)
        output_path1 = os.path.join(output_dir, f"{name}_noisy_level1{ext}")
        cv2.imwrite(output_path1, noisy1)
        result['levels'].append({'level': 1, 'path': output_path1})
    
    # Level 2
    if 2 in levels:
        noisy2 = apply_noise_level2(img)
        noisy2 = np.clip(noisy2, 0, 255).astype(np.uint8)
        output_path2 = os.path.join(output_dir, f"{name}_noisy_level2{ext}")
        cv2.imwrite(output_path2, noisy2)
        result['levels'].append({'level': 2, 'path': output_path2})
    
    return result

# ===============================
#   批量处理函数
# ===============================

def batch_process(input_dir, output_dir, levels=[1, 2]):
    """
    批量处理文件夹中的所有图像
    
    Args:
        input_dir: 输入图像文件夹路径
        output_dir: 输出噪声图像文件夹路径
        levels: 需要生成的噪声等级列表，默认 [1, 2]
    """
    # 支持的图像格式
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm', '.ppm')
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(valid_ext)]
    image_files.sort()
    
    if not image_files:
        print(f"错误：在 {input_dir} 中没有找到有效的图像文件")
        print(f"支持的格式：{valid_ext}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"找到 {len(image_files)} 张图像待处理")
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print(f"生成噪声等级：{levels}")
    print("-" * 60)
    
    # 统计信息
    success_count = 0
    fail_count = 0
    
    # 批量处理
    for filename in tqdm(image_files, desc="添加噪声中"):
        input_path = os.path.join(input_dir, filename)
        
        try:
            result = process_single_image(input_path, output_dir, levels)
            success_count += 1
            
            # 打印处理详情
            level_info = ", ".join([f"Level{l['level']}" for l in result['levels']])
            print(f"✓ {filename} -> {level_info}")
            
        except Exception as e:
            fail_count += 1
            print(f"✗ {filename} 处理失败：{str(e)}")
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("处理完成统计：")
    print(f"  成功：{success_count} 张")
    print(f"  失败：{fail_count} 张")
    print(f"  总计：{len(image_files)} 张")
    print(f"\n噪声图像保存位置：{output_dir}")

# ===============================
#   主函数
# ===============================

def main():
    parser = argparse.ArgumentParser(description='批量添加混合噪声（高斯+条纹）')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入图像文件夹路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出噪声图像文件夹路径')
    parser.add_argument('--levels', '-l', type=int, nargs='+', default=[1, 2],
                        help='噪声等级，可选 1 和/或 2，默认 [1, 2]')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='随机种子，用于复现结果（可选）')
    
    args = parser.parse_args()
    
    # 设置随机种子（如果指定）
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"已设置随机种子：{args.seed}")
    
    # 验证噪声等级参数
    valid_levels = [1, 2]
    levels = [l for l in args.levels if l in valid_levels]
    if not levels:
        print(f"错误：无效的噪声等级 {args.levels}，只能选 1 或 2")
        return
    
    # 执行批量处理
    batch_process(args.input, args.output, levels)

# ===============================
#   程序入口
# ===============================

if __name__ == '__main__':
    main()
    
    # 示例调用（命令行）：
    # python add_noise_batch.py -i /path/to/clean/images -o /path/to/noisy/output
    # python add_noise_batch.py -i ./clean -o ./noisy -l 1 2 --seed 42

