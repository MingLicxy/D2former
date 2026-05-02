import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_mae(img1, img2):
    """
    Compute Mean Absolute Error (MAE) between two images.
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return np.mean(np.abs(img1 - img2))


def error_level_pie_chart(
    pred_dir,
    gt_dir,
    save_dir,
    thresholds=(2, 5, 10),
    save_name="error_level_distribution.png"
):
    """
    Compute error levels between restored images and GT images,
    then plot and save a pie chart.
    """

    os.makedirs(save_dir, exist_ok=True)

    level_counts = [0, 0, 0, 0]
    pred_files = sorted(os.listdir(pred_dir))

    for fname in tqdm(pred_files, desc="Processing images"):
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.exists(gt_path):
            continue

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            continue

        mae = compute_mae(pred, gt)

        if mae < thresholds[0]:
            level_counts[0] += 1
        elif mae < thresholds[1]:
            level_counts[1] += 1
        elif mae < thresholds[2]:
            level_counts[2] += 1
        else:
            level_counts[3] += 1

    # ===== Plot pie chart (no external labels) =====
    colors = [
        "#FAF3DD",  # blue
        "#FFA69E",  # green
        "#B8F2E6",  # yellow
        "#AED9E0"   # red
    ]

    explode = (0.05, 0.05, 0.05, 0.05)  # uniform spacing

    plt.figure(figsize=(7, 7))
    plt.pie(
        level_counts,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",          # percentage inside
        pctdistance=0.7,            # control text position
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(color="black", fontsize=12)
    )

    plt.axis("equal")  # keep circle

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nPie chart saved to: {save_path}")
    print("Image counts per level:", level_counts)

    return level_counts


if __name__ == "__main__":

    thresholds = (3, 4, 5)  # MAE thresholds

    error_level_pie_chart(
        pred_dir="/home/caoxinyu/UNet-based/Xformer-main/results/SwinIR/IR700_test/50",
        gt_dir="/home/caoxinyu/UNet-based/infrare_data/test/IR700_test",
        save_dir="/home/caoxinyu/UNet-based/Xformer-main/results",
        thresholds=thresholds,
        save_name="IR700_test_SwinIR_50.png"
    )

    

   