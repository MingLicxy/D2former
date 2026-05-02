# 对去噪后图像进行双三次插值上采样，提升可视化结果的清晰度
import os
from PIL import Image

def bicubic_upsample(input_folder, output_folder, target_size):
    """
    对指定路径的图像进行双三次上采样，并保存结果到指定文件夹。

    Args:
        input_folder (str): 输入图像所在的文件夹路径。
        output_folder (str): 上采样后图像保存的文件夹路径。
        target_size (tuple): 目标尺寸 (width, height)。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                with Image.open(input_path) as img:
                    # 双三次上采样
                    upsampled_img = img.resize(target_size, Image.BICUBIC)
                    output_path = os.path.join(output_folder, filename)
                    upsampled_img.save(output_path)
                    print(f"保存文件: {output_path}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

# 使用示例
input_folder = "/path/to/input/folder"  # 替换为输入文件夹路径
output_folder = "/path/to/output/folder"  # 替换为输出文件夹路径
target_size = (512, 512)  # 替换为目标尺寸 (width, height)由输入图像尺寸决定

bicubic_upsample(input_folder, output_folder, target_size)

