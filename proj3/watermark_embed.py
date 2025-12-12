import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def spread_spectrum_embed(original_img_path, output_img_path="watermarked_img.png",
                          alpha=0.05, n=512, seed=123):

    original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        raise ValueError("无法读取原始图像，请检查文件路径是否正确")

    img = original_img.astype(np.float64) / 255.0
    h, w = img.shape

    dct_img = cv2.dct(np.float32(img))

    np.random.seed(seed)
    watermark = np.random.randn(n)

    dct_flat = dct_img.flatten()
    sorted_idx = np.argsort(np.abs(dct_flat))[::-1]  # 按绝对值降序排序
    embed_idx = sorted_idx[1:n + 1]  # 跳过第1个直流分量，选后续n个系数

    original_coeff = dct_flat[embed_idx].copy()

    dct_flat[embed_idx] = original_coeff + alpha * watermark

    dct_img_wm = dct_flat.reshape((h, w))
    watermarked_img = cv2.idct(dct_img_wm)

    watermarked_img = np.clip(watermarked_img, 0, 1)
    watermarked_img_8bit = (watermarked_img * 255).astype(np.uint8)

    cv2.imwrite(output_img_path, watermarked_img_8bit)
    print(f"水印嵌入完成！含水印图像已保存至：{output_img_path}")

    key_dict = {
        "embed_idx": embed_idx.tolist(),  # 嵌入位置索引（转列表便于存储）
        "original_coeff": original_coeff.tolist(),  # 原始DCT系数
        "alpha": alpha,  # 水印强度
        "watermark": watermark.tolist(),  # 原始水印序列
        "img_shape": (h, w)  # 图像尺寸
    }
    return key_dict,original_img,watermarked_img_8bit

def plot_side_by_side(image_before, image_after, title, filename):
    plt.figure(figsize=(12, 6))

    # # 转换为RGB以正确显示
    # image_before_rgb = cv2.cvtColor(image_before, cv2.COLOR_BGR2RGB)
    # image_after_rgb = cv2.cvtColor(image_after, cv2.COLOR_BGR2RGB)

    # 并排显示
    plt.subplot(1, 2, 1)
    plt.imshow(image_before,cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_after,cmap='gray')
    plt.title("Stego Image")
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # 配置参数
    ORIGINAL_IMG_PATH = "pic_gray.png"
    OUTPUT_IMG_PATH = "1.3/watermarked_out.png"
    ALPHA = 0.05
    N = 512
    SEED = 42

    key,image_before,image_after = spread_spectrum_embed(ORIGINAL_IMG_PATH, OUTPUT_IMG_PATH, ALPHA, N, SEED)
    # 将密钥保存到文本文件
    with open("1.3/watermark_key.txt", "w", encoding="utf-8") as f:
        f.write(str(key))
    print("检测密钥已保存至：watermark_key.txt")

    plot_side_by_side(image_before, image_after,"Original vs Watermarked", "1.3/comparison.png")

