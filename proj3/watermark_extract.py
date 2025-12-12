import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

def spread_spectrum_extract(test_img_path, key_dict):

    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        raise ValueError("无法读取待测图像，请检查文件路径是否正确")

    img = test_img.astype(np.float64) / 255.0
    h, w = img.shape

    if (h, w) != key_dict["img_shape"]:
        raise ValueError(f"待测图像尺寸({h}×{w})与原始图像尺寸{key_dict['img_shape']}不匹配")

    dct_test_img = cv2.dct(np.float32(img))
    dct_test_flat = dct_test_img.flatten()

    embed_idx = np.array(key_dict["embed_idx"])
    wm_dct_coeff = dct_test_flat[embed_idx]

    original_coeff = np.array(key_dict["original_coeff"])
    alpha = key_dict["alpha"]
    extracted_wm = (wm_dct_coeff - original_coeff) / alpha

    original_wm = np.array(key_dict["watermark"])
    numerator = np.sum(original_wm * extracted_wm)
    denominator = np.sqrt(np.sum(original_wm ** 2)) * np.sqrt(np.sum(extracted_wm ** 2))
    similarity = numerator / denominator if denominator != 0 else 0

    print(f"\n水印检测结果：")
    print(f"提取水印与原始水印的相似度：{similarity:.4f}")
    threshold = 0.8
    if similarity > threshold:
        print("检测结论：待测图像中存在目标扩频水印")
    else:
        print("检测结论：待测图像中未检测到目标扩频水印")

    return similarity, extracted_wm

def visualize_dct_embed_positions(original_img_path, key_dict):
    """
    可视化DCT域嵌入位置的分布特征
    """
    # 读取原始图像并进行DCT变换
    original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        raise ValueError("无法读取原始图像，请检查文件路径")
    
    img = original_img.astype(np.float64) / 255.0
    h, w = img.shape
    dct_img = cv2.dct(np.float32(img))
    
    # 从密钥中获取嵌入位置
    embed_idx = np.array(key_dict["embed_idx"])
    n = len(embed_idx)
    
    # 将一维索引转换为二维坐标
    embed_positions = np.unravel_index(embed_idx, (h, w))
    
    # 创建DCT系数绝对值热力图
    
    dct_abs = np.abs(dct_img)
    # 方法1：截断极值（DC分量），聚焦高频区域
    dct_abs_clipped = np.clip(dct_abs, 0, np.percentile(dct_abs, 99))  # 截断99%分位数以上的极值
    dct_abs_normalized = dct_abs_clipped / np.max(dct_abs_clipped)  # 归一化
    
    # 自定义颜色映射（增强低亮度区域的区分度）
    colors = [(0, 0, 0), (0.2, 0.1, 0.4), (0.5, 0.3, 0.8), (0.8, 0.6, 1), (1, 1, 1)]
    cmap = LinearSegmentedColormap.from_list("dct_cmap", colors, N=256)
    
    plt.figure(figsize=(8, 6))
    
    # ========== 子图1：优化后的DCT系数分布 + 嵌入位置 ==========
    
    # 使用更温和的对数变换，或直接使用截断后的系数
    im = plt.imshow(dct_abs_clipped, cmap=cmap, vmin=0, vmax=np.max(dct_abs_clipped))
    # 突出显示嵌入位置（增大点的尺寸和透明度）
    plt.scatter(embed_positions[1], embed_positions[0], c='red', s=5, alpha=0.8, label='嵌入位置', edgecolors='white', linewidths=0.2)
    plt.colorbar(im, label='|DCT系数|（截断99%极值后）')
    plt.title('DCT系数分布与水印嵌入位置', fontsize=12)
    plt.xlabel('水平频率', fontsize=10)
    plt.ylabel('垂直频率', fontsize=10)
    plt.legend(fontsize=9)
    plt.xticks(np.arange(0, w, 50))  # 增加刻度，便于定位
    plt.yticks(np.arange(0, h, 50))
    plt.legend()
    plt.savefig('1.3/dct_embed_positions.png', dpi=300)
    
    # 绘制系数绝对值排序与嵌入位置分布

    plt.figure(figsize=(8, 6))
    dct_flat = dct_img.flatten()
    sorted_abs = np.sort(np.abs(dct_flat))[::-1]  # 按绝对值降序排序
    
    
    plt.plot(range(len(sorted_abs)), sorted_abs, 'b-', linewidth=0.8, label='DCT系数绝对值')
    plt.scatter(range(1, n+1), sorted_abs[1:n+1], c='r', s=10, label='嵌入位置系数')
    plt.axvline(x=0, color='g', linestyle='--', label='直流分量(跳过)')
    plt.yscale('log')
    plt.xlabel('系数排序位置(按绝对值降序)')
    plt.ylabel('系数绝对值(对数尺度)')
    plt.title('DCT系数排序与嵌入位置选择')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1.3/dct_embed_visualization.png', dpi=300)
    print("DCT域嵌入位置可视化结果已保存至：dct_embed_visualization.png")
    


if __name__ == "__main__":
    TEST_IMG_PATH = "1.3/watermarked_out.png"  # 含水印图像路径
    KEY_FILE_PATH = "1.3/watermark_key.txt"  # 密钥文件路径
    ORIGINAL_IMG_PATH = "pic_gray.png"
    try:
        with open(KEY_FILE_PATH, "r", encoding="utf-8") as f:
            key_str = f.read()
            key_dict = eval(key_str)  # 还原字典（实验场景下安全使用）
    except FileNotFoundError:
        raise FileNotFoundError(f"密钥文件不存在，请检查路径：{KEY_FILE_PATH}")

    similarity, extracted_wm = spread_spectrum_extract(TEST_IMG_PATH, key_dict)
    # visualize_dct_embed_positions(ORIGINAL_IMG_PATH, key_dict)