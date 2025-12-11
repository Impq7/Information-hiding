from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from PIL import Image
import pywt
import matplotlib.pyplot as plt

# 基准参数配置
BASE_SEED = 42                # 随机数种子
BASE_WAVELET = "db1"          # 小波基函数
BASE_ALPHA = 0.5              # 水印强度
BASE_RATIO = 0.8              # d/n修改比例

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
    """将图像填充为正方形（保证SVD分解为方阵）"""
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
    level: int,  # 改为可变参数（分解尺度）
    alpha: float = BASE_ALPHA,
    seed: int = BASE_SEED,
    wavelet: str = BASE_WAVELET,
    ratio: float = BASE_RATIO,
) -> Dict[str, np.ndarray]:
    """W-SVD水印嵌入核心函数（分解尺度为可变参数）"""
    data = np.asarray(data_rgb, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("期望输入为 RGB 图像（H×W×3）")
    
    # 提取R通道嵌入水印
    datared = data[:, :, 0]
    row, col = datared.shape
    
    # 小波分解（使用指定的分解尺度）
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
    
    # 生成正交矩阵（固定种子）
    rng = np.random.RandomState(seed)
    M_V = rng.rand(d, np_cap) - 0.5
    Q_V, _ = np.linalg.qr(M_V)
    M_U = rng.rand(d, np_cap) - 0.5
    Q_U, _ = np.linalg.qr(M_U)
    
    # 替换U/V矩阵列向量
    U[:, d - np_cap : d] = Q_U[:, :np_cap]
    V[:, d - np_cap : d] = Q_V[:, :np_cap]
    
    # 生成水印模板
    sigma_rand = rng.rand(d)
    sigma_sorted = np.sort(sigma_rand)[::-1]
    sigma_tilda = alpha * sigma_sorted
    watermark = U @ np.diag(sigma_tilda) @ V.T
    
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
        "watermark_template": watermark,  # 返回水印模板（用于绘制形态图）
        "CA_shape": real_CA_shape
    }

def plot_level_morphology(watermark_templates: Dict[int, np.ndarray], output_dir: Path):
    """绘制不同分解尺度的水印形态图（2行3列布局）"""
    levels = sorted(watermark_templates.keys())  # 按level=1~6排序
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2行3列
    axes = axes.flatten()  # 转为一维数组遍历
    
    for ax, level in zip(axes, levels):
        watermark = watermark_templates[level]
        # 归一化水印模板到[0, 255]（便于显示）
        wm_norm = (watermark - watermark.min()) / (watermark.max() - watermark.min() + 1e-12) * 255
        # 绘制水印形态图
        ax.imshow(wm_norm, cmap="gray", interpolation="none")
        ax.set_title(f"level={level}", fontsize=14, pad=8)
        ax.axis("off")
    
    # 调整布局
    plt.tight_layout(pad=2.0)
    save_path = output_dir / "watermark_morphology_by_level.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 分解尺度水印形态图已保存至：{save_path}")

def generate_level_variations(
    cover_path: str | Path,
    output_dir: str | Path,
    level_list: List[int] = None
):
    """生成不同分解尺度的水印图像+形态图"""
    if level_list is None:
        level_list = [1, 2, 3, 4, 5, 6]  # 目标分解尺度
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取载体图像
    img_before, cover_rgb = load_rgb_image_float01(cover_path)
    print(f"成功读取载体图像：{cover_path}")
    print(f"基准参数：seed={BASE_SEED}, wavelet={BASE_WAVELET}, alpha={BASE_ALPHA}, ratio={BASE_RATIO}")
    print(f"待测试分解尺度：{level_list}")
    
    # 存储各尺度的水印模板
    watermark_templates = {}
    
    # 遍历所有分解尺度生成水印图像
    for level in level_list:
        try:
            # 生成水印图像
            result = wavemarksvd_like(
                data_rgb=cover_rgb,
                level=level  # 传入当前分解尺度
            )
            watermark_img_rgb = result["watermarkimagergb"]
            watermark_templates[level] = result["watermark_template"]  # 记录水印模板
            
            # 保存水印图像
            img_filename = f"wsvd_level_{level}.bmp"
            img_path = output_dir / img_filename
            img_after = save_rgb_image_float01(watermark_img_rgb, img_path)
            
            print(f"✓ 分解尺度={level}：生成完成 -> {img_path}")
        
        except Exception as e:
            print(f"✗ 分解尺度={level}：生成失败 - {str(e)}")
    
    # 绘制水印形态图
    plot_level_morphology(watermark_templates, output_dir)
    
    print(f"\n实验完成！所有结果已保存至：{output_dir}")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="W-SVD水印-分解尺度参数测试工具")
    parser.add_argument("--cover", required=True, help="载体图像路径（RGB格式）")
    parser.add_argument("--out_dir", required=True, help="输出目录（自动创建）")
    parser.add_argument("--levels", nargs="+", type=int, 
                       default=[1,2,3,4,5,6],
                       help="待测试的分解尺度列表（默认：1 2 3 4 5 6）")
    return parser

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # 执行分解尺度参数测试
    generate_level_variations(
        cover_path=args.cover,
        output_dir=args.out_dir,
        level_list=args.levels
    )