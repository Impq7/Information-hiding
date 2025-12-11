from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from PIL import Image
import pywt
import matplotlib.pyplot as plt

# 基准参数配置（固定值）
BASE_SEED = 42                # 随机数种子
BASE_WAVELET = "db1"          # 小波基函数
BASE_LEVEL = 1                # 小波分解尺度（level对应分解尺度）
BASE_RATIO = 0.8              # d/n修改比例
BASE_ALPHA = 0.5              # 默认水印强度

def load_rgb_image_float01(path: str | Path) -> Tuple[Image.Image, np.ndarray]:
    """读取RGB图像并归一化到[0,1]"""
    path = Path(path)
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return img, arr

def save_rgb_image_float01(arr: np.ndarray, path: str | Path) -> Image.Image:
    """保存归一化的RGB图像"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_clipped = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr_clipped * 255.0).round().astype(np.uint8), mode="RGB")
    img.save(path)
    return img

def pad_to_square(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """将图像填充为正方形（保证SVD分解的矩阵为方阵）"""
    row, col = image.shape
    standard = max(row, col)
    padded = np.zeros((standard, standard), dtype=image.dtype)
    if row <= col:
        padded[:row, :] = image
    else:
        padded[:, :col] = image
    return padded, (row, col)

def wavemarksvd_like(
    data_rgb: np.ndarray,
    alpha: float = BASE_ALPHA,
    seed: int = BASE_SEED,
    wavelet: str = BASE_WAVELET,
    level: int = BASE_LEVEL,
    ratio: float = BASE_RATIO,
) -> Dict[str, np.ndarray]:
    """
    W-SVD水印嵌入核心函数
    :param data_rgb: 归一化的RGB载体图像
    :param alpha: 水印强度（唯一可变参数）
    :param seed: 随机数种子（固定为42）
    :param wavelet: 小波基（固定为db1）
    :param level: 小波分解尺度（固定为1）
    :param ratio: d/n修改比例（固定为0.8）
    :return: 水印图像及相关信息
    """
    data = np.asarray(data_rgb, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("期望输入为 RGB 图像（H×W×3）")
    
    # 提取R通道进行水印嵌入
    datared = data[:, :, 0]
    row, col = datared.shape
    
    # 小波分解
    coeffs_real = pywt.wavedec2(datared, wavelet, level=level)
    CA_real = coeffs_real[0]
    real_CA_shape = CA_real.shape
    
    # 填充为正方形
    new, orig_shape = pad_to_square(datared)
    coeffs = pywt.wavedec2(new, wavelet, level=level)
    CA = coeffs[0]
    d1, d2 = CA.shape
    if d1 != d2:
        raise RuntimeError(f"最深层 CA 非方阵：shape={CA.shape}")
    d = d1
    
    # 低频系数归一化
    CAmin = float(CA.min())
    CAmax = float(CA.max())
    eps = 1e-12
    CA_norm = (CA - CAmin) / (CAmax - CAmin + eps)
    
    # SVD分解
    U, sigma, Vt = np.linalg.svd(CA_norm, full_matrices=True)
    V = Vt.T
    
    # 计算替换列数
    np_cap = int(round(d * ratio))
    np_cap = max(1, min(np_cap, d))
    
    # 生成正交矩阵（固定种子保证可重复）
    rng = np.random.RandomState(seed)
    M_V = rng.rand(d, np_cap) - 0.5
    Q_V, _ = np.linalg.qr(M_V)
    M_U = rng.rand(d, np_cap) - 0.5
    Q_U, _ = np.linalg.qr(M_U)
    
    # 替换U/V矩阵列向量
    U2 = U.copy()
    V2 = V.copy()
    U[:, d - np_cap : d] = Q_U[:, :np_cap]
    V[:, d - np_cap : d] = Q_V[:, :np_cap]
    
    # 生成水印模板
    sigma_rand = rng.rand(d)
    sigma_sorted = np.sort(sigma_rand)[::-1]
    sigma_tilda = alpha * sigma_sorted
    watermark = U @ np.diag(sigma_tilda) @ V.T
    
    # 计算U/V相关系数
    def corr2(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.ravel().astype(np.float64)
        b_flat = b.ravel().astype(np.float64)
        if a_flat.size == 0 or b_flat.size == 0:
            return 0.0
        a_mean = a_flat.mean()
        b_mean = b_flat.mean()
        a_z = a_flat - a_mean
        b_z = b_flat - b_mean
        denom = float(np.linalg.norm(a_z) * np.linalg.norm(b_z))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_z, b_z) / denom)
    
    correlationU = corr2(U, U2)
    correlationV = corr2(V, V2)
    
    # 嵌入水印并还原
    CA_tilda_norm = CA_norm + watermark
    CA_tilda_norm = np.clip(CA_tilda_norm, 0.0, 1.0)
    CA_tilda_real = (CAmax - CAmin) * CA_tilda_norm + CAmin
    
    # 恢复原始尺寸
    h_ll, w_ll = real_CA_shape
    waterCA = CA_tilda_real[:h_ll, :w_ll]
    coeffs_new = [CA_tilda_real] + list(coeffs[1:])
    watermarked_padded = pywt.waverec2(coeffs_new, wavelet)
    r0, c0 = orig_shape
    if r0 <= c0:
        watermarkimage = watermarked_padded[:r0, :]
    else:
        watermarkimage = watermarked_padded[:, :c0]
    
    # 合并RGB通道
    watermarkimagergb = data.copy()
    watermarkimagergb[:, :, 0] = watermarkimage
    
    return {
        "watermarkimagergb": watermarkimagergb,
        "watermarkimage": watermarkimage,
        "correlationU": correlationU,
        "correlationV": correlationV,
        "CA_shape": real_CA_shape,
        "CA_min": CAmin,
        "CA_max": CAmax
    }

def plot_side_by_side(image_before, image_after, alpha, output_dir):
    """生成原始图像与水印图像对比图"""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_before)
    plt.title("原始图像")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_after)
    plt.title(f"水印图像 (α={alpha})")
    plt.axis('off')
    
    plt.suptitle(f"W-SVD水印嵌入效果对比（α={alpha}）", fontsize=14)
    plt.tight_layout()
    save_path = output_dir / f"comparison_alpha_{alpha}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_alpha_variations(
    cover_path: str | Path,
    output_dir: str | Path,
    alpha_list: List[float] = None
):
    """
    生成不同α值的水印图像（其他参数固定）
    :param cover_path: 载体图像路径
    :param output_dir: 输出目录
    :param alpha_list: 待测试的α值列表，默认[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    """
    if alpha_list is None:
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取载体图像
    img_before, cover_rgb = load_rgb_image_float01(cover_path)
    print(f"成功读取载体图像：{cover_path}")
    print(f"基准参数：seed={BASE_SEED}, wavelet={BASE_WAVELET}, level={BASE_LEVEL}, ratio={BASE_RATIO}")
    print(f"待测试α值：{alpha_list}")
    
    # 生成日志文件
    # log_file = output_dir / "alpha_experiment_log.txt"
    # with open(log_file, "w", encoding="utf-8") as f:
    #     f.write("W-SVD水印α参数实验日志\n")
    #     f.write(f"基准参数：seed={BASE_SEED}, wavelet={BASE_WAVELET}, level={BASE_LEVEL}, ratio={BASE_RATIO}\n")
    #     f.write("="*60 + "\n")
    #     f.write(f"{'α值':<8} {'输出文件名':<40} {'corrU':<10} {'corrV':<10} {'CA形状':<15}\n")
    #     f.write("="*60 + "\n")
    
    # 遍历所有α值生成水印图像
    for alpha in alpha_list:
        try:
            # 生成水印图像
            result = wavemarksvd_like(cover_rgb, ratio=alpha)
            watermark_img_rgb = result["watermarkimagergb"]
            
            # 构建输出文件名
            img_filename = f"wsvd_ratio_{alpha}.bmp"
            img_path = output_dir / img_filename
            
            # 保存水印图像
            img_after = save_rgb_image_float01(watermark_img_rgb, img_path)
            
            # # 生成对比图
            # plot_side_by_side(img_before, img_after, alpha, output_dir)
            
            # # 记录日志
            # log_entry = (
            #     f"{alpha:<8} {img_filename:<40} {result['correlationU']:<10.4f} "
            #     f"{result['correlationV']:<10.4f} {str(result['CA_shape']):<15}\n"
            # )
            # with open(log_file, "a", encoding="utf-8") as f:
            #     f.write(log_entry)
            
            print(f"✓ α={alpha}：生成完成 -> {img_path}")
        
        except Exception as e:
            error_msg = f"✗ α={alpha}：生成失败 - {str(e)}\n"
            # with open(log_file, "a", encoding="utf-8") as f:
            #     f.write(error_msg)
            # print(error_msg)
    
    print(f"\n实验完成！所有结果已保存至：{output_dir}")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="W-SVD水印α参数测试工具（固定其他参数）")
    parser.add_argument("--cover", required=True, help="载体图像路径（RGB格式）")
    parser.add_argument("--out_dir", required=True, help="输出目录（自动创建）")
    parser.add_argument("--ratios", nargs="+", type=float, 
                       default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                       help="待测试的α值列表（默认：0.1-1.0 步长0.1）")
    return parser

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # 执行α参数测试
    generate_alpha_variations(
        cover_path=args.cover,
        output_dir=args.out_dir,
        alpha_list=args.ratios
    )