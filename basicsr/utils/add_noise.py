import os
import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, std=25):
    """
    对图像添加高斯噪声。
    :param image: 输入图像（灰度图）。
    :param mean: 高斯噪声的均值。
    :param std: 高斯噪声的标准差。
    :return: 加噪后的图像。
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # 将像素值限制在[0, 255]之间
    return noisy_image.astype(np.uint8)

def process_images(input_folder, output_folder, mean=0, std=25):
    """
    对指定文件夹中的灰度PNG图像添加高斯噪声，并保存到输出文件夹。
    :param input_folder: 输入文件夹路径。
    :param output_folder: 输出文件夹路径。
    :param mean: 高斯噪声的均值。
    :param std: 高斯噪声的标准差。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):  # 仅处理 .png 文件
            input_path = os.path.join(input_folder, file_name)

            if os.path.isfile(input_path):
                # 读取灰度图像
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"文件 {file_name} 不是有效的图像，跳过...")
                    continue

                # 添加高斯噪声
                noisy_image = add_gaussian_noise(image, mean, std)

                # 保存加噪图像
                output_path = os.path.join(output_folder, file_name)
                cv2.imwrite(output_path, noisy_image)
                print(f"处理完成: {file_name} -> {output_path}")

# 示例调用
input_folder = "/home/caoxinyu/UNet-based/infrare_data/test/DLS-NUC-100"  # 替换为实际输入文件夹路径
output_folder = "/home/caoxinyu/UNet-based/infrare_data/test/noise"  # 替换为实际输出文件夹路径
mean = 0  # 高斯噪声均值
std = 50  # 高斯噪声标准差

process_images(input_folder, output_folder, mean, std)
