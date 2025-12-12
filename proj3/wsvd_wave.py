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
BASE_LEVEL = 1                # 小波分解尺度
BASE_RATIO = 0.8              # d/n修改比例
BASE_ALPHA = 0.5              # 水印强度

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
    wavelet: str,  # 改为可变参数（小波基）
    alpha: float = BASE_ALPHA,
    seed: int = BASE_SEED,
    level: int = BASE_LEVEL,
    ratio: float = BASE_RATIO,
) -> Dict[str, np.ndarray]:
    """W-SVD水印嵌入核心函数（小波基为可变参数）"""
    data = np.asarray(data_rgb, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("期望输入为 RGB 图像（H×W×3）")
    
    # 提取R通道嵌入水印
    datared = data[:, :, 0]
    row, col = datared.shape
    
    # 小波分解（使用传入的小波基）
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
        "watermark_template": watermark,  # 新增：返回水印模板（用于绘制形态图）
        "CA_shape": real_CA_shape
    }

def plot_watermark_morphology(watermark_templates: Dict[str, np.ndarray], output_dir: Path):
    """绘制不同小波基的水印形态图（2行2列布局）"""
    wavelets = list(watermark_templates.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()  # 转为一维数组方便遍历
    
    for ax, wname in zip(axes, wavelets):
        watermark = watermark_templates[wname]
        # 归一化水印模板到[0, 255]（便于显示）
        wm_norm = (watermark - watermark.min()) / (watermark.max() - watermark.min() + 1e-12) * 255
        # 绘制水印形态图
        im = ax.imshow(wm_norm, cmap="gray", interpolation="none")
        ax.set_title(f"wname={wname}", fontsize=12, pad=5)
        ax.axis("off")
    
    # 隐藏多余的子图（若小波基数量不足4个）
    for i in range(len(wavelets), len(axes)):
        axes[i].axis("off")
    
    # 添加颜色条
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="归一化水印值")
    plt.tight_layout()
    save_path = output_dir / "watermark_morphology_by_wavelet.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 水印形态图已保存至：{save_path}")

def generate_wavelet_variations(
    cover_path: str | Path,
    output_dir: str | Path,
    wavelet_list: List[str] = None
):
    """生成不同小波基的水印图像+形态图"""
    if wavelet_list is None:
        wavelet_list = ["db1", "db3", "db6", "haar"]  # 包含haar作为对比
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取载体图像
    img_before, cover_rgb = load_rgb_image_float01(cover_path)
    print(f"成功读取载体图像：{cover_path}")
    print(f"基准参数：seed={BASE_SEED}, level={BASE_LEVEL}, ratio={BASE_RATIO}, alpha={BASE_ALPHA}")
    print(f"待测试小波基：{wavelet_list}")
    
    # 存储各小波基的水印模板（用于绘制形态图）
    watermark_templates = {}
    
    # 遍历所有小波基生成水印图像
    for wname in wavelet_list:
        try:
            # 生成水印图像
            result = wavemarksvd_like(
                data_rgb=cover_rgb,
                wavelet=wname  # 传入当前小波基
            )
            watermark_img_rgb = result["watermarkimagergb"]
            watermark_templates[wname] = result["watermark_template"]  # 记录水印模板
            
            # 保存水印图像
            img_filename = f"wsvd_wavelet_{wname}.bmp"
            img_path = output_dir / img_filename
            img_after = save_rgb_image_float01(watermark_img_rgb, img_path)
            
            print(f"✓ 小波基={wname}：生成完成 -> {img_path}")
        
        except Exception as e:
            print(f"✗ 小波基={wname}：生成失败 - {str(e)}")
    
    # 绘制水印形态图
    plot_watermark_morphology(watermark_templates, output_dir)
    
    print(f"\n实验完成！所有结果已保存至：{output_dir}")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="W-SVD水印-小波基参数测试工具")
    parser.add_argument("--cover", required=True, help="载体图像路径（RGB格式）")
    parser.add_argument("--out_dir", required=True, help="输出目录（自动创建）")
    parser.add_argument("--wavelets", nargs="+", type=str, 
                       default=["db1", "db3", "db6", "haar"],
                       help="待测试的小波基列表（默认：db1 db3 db6 haar）")
    return parser

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # 执行小波基参数测试
    generate_wavelet_variations(
        cover_path=args.cover,
        output_dir=args.out_dir,
        wavelet_list=args.wavelets
    )