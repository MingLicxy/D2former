import os
import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, std=25):
    """
    向图像添加高斯噪声
    :param image: 输入图像
    :param mean: 高斯噪声的均值
    :param std: 高斯噪声的标准差
    :return: 加噪后的图像
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def process_images(input_folder, output_folder, mean=0, std=25):
    """
    向文件夹中的图像添加高斯噪声并保存到目标文件夹
    :param input_folder: 输入图像文件夹路径
    :param output_folder: 输出图像文件夹路径
    :param mean: 高斯噪声的均值
    :param std: 高斯噪声的标准差
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.bmp')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
            if image is not None:
                noisy_image = add_gaussian_noise(image, mean, std)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, noisy_image)
                print(f"已处理并保存: {output_path}")
            else:
                print(f"无法读取图像: {img_path}")

if __name__ == "__main__":
    input_folder = "/home/caoxinyu/UNet-based/infrare_data/test/IR700_test"  # 替换为输入图像文件夹路径
    output_folder = "/home/caoxinyu/UNet-based/infrare_data/test/IR700_test-noise/50"  # 替换为输出图像文件夹路径
    mean = 0  # 高斯噪声均值
    std = 50  # 高斯噪声标准差 [15, 25, 50]

    process_images(input_folder, output_folder, mean, std)
