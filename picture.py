from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import pywt
import matplotlib.pyplot as plt
import cv2

# 全局配置：解决中文显示和负号问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


def load_rgb_image_float01(path: str | Path) -> Tuple[Image.Image, np.ndarray]:
    path = Path(path)
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return img, arr


def save_rgb_image_float01(arr: np.ndarray, path: str | Path) -> Image.Image:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_clipped = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr_clipped * 255.0).round().astype(np.uint8), mode="RGB")
    img.save(path)
    return img


def pad_to_square(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
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
    alpha: float,
    seed: int,
    wavelet: str,
    level: int,
    ratio: float,
) -> Dict[str, np.ndarray]:
    data = np.asarray(data_rgb, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("wavemarksvd_like 期望输入为 RGB 图像（H×W×3）")

    datared = data[:, :, 0]
    row, col = datared.shape

    coeffs_real = pywt.wavedec2(datared, wavelet, level=level)
    CA_real = coeffs_real[0]
    real_CA_shape = CA_real.shape

    new, orig_shape = pad_to_square(datared)

    coeffs = pywt.wavedec2(new, wavelet, level=level)
    CA = coeffs[0]
    d1, d2 = CA.shape
    if d1 != d2:
        raise RuntimeError(f"最深层 CA 非方阵：shape={CA.shape}")
    d = d1

    CAmin = float(CA.min())
    CAmax = float(CA.max())
    eps = 1e-12
    CA_norm = (CA - CAmin) / (CAmax - CAmin + eps)

    U, sigma, Vt = np.linalg.svd(CA_norm, full_matrices=True)
    V = Vt.T

    np_cap = int(round(d * ratio))
    np_cap = max(1, min(np_cap, d))

    rng = np.random.RandomState(seed)
    M_V = rng.rand(d, np_cap) - 0.5
    Q_V, _ = np.linalg.qr(M_V)
    M_U = rng.rand(d, np_cap) - 0.5
    Q_U, _ = np.linalg.qr(M_U)

    U2 = U.copy()
    V2 = V.copy()

    U[:, d - np_cap : d] = Q_U[:, :np_cap]
    V[:, d - np_cap : d] = Q_V[:, :np_cap]

    sigma_rand = rng.rand(d)
    sigma_sorted = np.sort(sigma_rand)[::-1]
    sigma_tilda = alpha * sigma_sorted

    watermark = U @ np.diag(sigma_tilda) @ V.T

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

    CA_tilda_norm = CA_norm + watermark
    CA_tilda_norm = np.clip(CA_tilda_norm, 0.0, 1.0)

    CA_tilda_real = (CAmax - CAmin) * CA_tilda_norm + CAmin

    h_ll, w_ll = real_CA_shape
    waterCA = CA_tilda_real[:h_ll, :w_ll]

    coeffs_wm = [watermark] + list(coeffs[1:])
    watermark2_full = pywt.waverec2(coeffs_wm, wavelet)
    r0, c0 = orig_shape
    if r0 <= c0:
        watermark2 = watermark2_full[:r0, :]
    else:
        watermark2 = watermark2_full[:, :c0]

    coeffs_new = [CA_tilda_real] + list(coeffs[1:])
    watermarked_padded = pywt.waverec2(coeffs_new, wavelet)
    if r0 <= c0:
        watermarkimage = watermarked_padded[:r0, :]
    else:
        watermarkimage = watermarked_padded[:, :c0]

    watermarkimagergb = data.copy()
    watermarkimagergb[:, :, 0] = watermarkimage

    info: Dict[str, np.ndarray] = {
        "CA": CA,
        "CA_norm": CA_norm,
        "CA_tilda_norm": CA_tilda_norm,
        "CA_tilda_real": CA_tilda_real,
        "watermark": watermark,
        "waterCA": waterCA,
        "watermark2": watermark2,
        "CAmin": np.array([CAmin]),
        "CAmax": np.array([CAmax]),
        "U": U,
        "V": V,
        "sigma_tilda": sigma_tilda,
        "correlationU": np.array([correlationU]),
        "correlationV": np.array([correlationV]),
        "real_CA_shape": np.array(real_CA_shape),
    }

    return {
        "watermarkimagergb": watermarkimagergb,
        "watermarkimage": watermarkimage,
        "waterCA": waterCA,
        "watermark2": watermark2,
        "correlationU": correlationU,
        "correlationV": correlationV,
        "info": info,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_embed = subparsers.add_parser("embed")
    p_embed.add_argument("--cover", required=True)
    p_embed.add_argument("--out", required=True)
    p_embed.add_argument("--alpha", type=float, default=0.5)
    p_embed.add_argument("--seed", type=int, default=1234)
    p_embed.add_argument("--wavelet", type=str, default="db1")
    p_embed.add_argument("--level", type=int, default=1)
    p_embed.add_argument("--ratio", type=float, default=0.8)

    return parser


def plot_watermark_and_images(image_before, image_after, watermark, title, filename):
    """
    绘制组合图：原始图像 + 含水印图像 + 水印形态图 + 水印序列直方图
    """
    # 处理水印形态图：归一化到[0,255]便于灰度显示
    watermark_norm = (watermark - watermark.min()) / (watermark.max() - watermark.min()) * 255
    watermark_norm = watermark_norm.astype(np.uint8)

    # 创建画布（2行2列布局）
    fig = plt.figure(figsize=(15, 10))

    # 子图1：原始图像
    ax1 = plt.subplot(2, 2, 1)
    plt.imshow(image_before)
    plt.title("原始图像", fontsize=12)
    plt.axis('off')

    # 子图2：含水印图像
    ax2 = plt.subplot(2, 2, 2)
    plt.imshow(image_after)
    plt.title("嵌入水印后的图像", fontsize=12)
    plt.axis('off')

    # 子图3：水印形态图（二维矩阵灰度显示）
    ax3 = plt.subplot(2, 2, 3)
    im = plt.imshow(watermark_norm, cmap='gray')
    plt.title("水印序列形态图", fontsize=12)
    plt.axis('off')
    # 添加颜色条（可选，量化水印值）
    plt.colorbar(im, ax=ax3, shrink=0.8, label='归一化水印值')

    # 子图4：水印序列直方图（展示值分布）
    ax4 = plt.subplot(2, 2, 4)
    plt.hist(watermark.ravel(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    plt.title("水印序列值分布直方图", fontsize=12)
    plt.xlabel("水印值")
    plt.ylabel("频数")
    plt.grid(alpha=0.3)

    # 全局标题和布局调整
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "embed":
        img_before, cover_rgb = load_rgb_image_float01(args.cover)
        result = wavemarksvd_like(
            cover_rgb,
            alpha=args.alpha,
            seed=args.seed,
            wavelet=args.wavelet,
            level=args.level,
            ratio=args.ratio,
        )
        watermarkimagergb = result["watermarkimagergb"]
        info = result["info"]
        watermark = info["watermark"]  # 获取水印矩阵
        img_after = save_rgb_image_float01(watermarkimagergb, args.out)
        
        print(f"嵌入完成，已保存含水印 RGB 图像到: {args.out}")
        print(
            f"  最深层 CA 形状: {info['CA'].shape}, "
            f"CA 范围: [{info['CAmin'][0]:.6f}, {info['CAmax'][0]:.6f}], "
            f"U/V 相关系数: "
            f"corrU={info['correlationU'][0]:.4f}, "
            f"corrV={info['correlationV'][0]:.4f}"
        )
        
        # 绘制包含水印序列的组合图
        plot_watermark_and_images(
            image_before=img_before,
            image_after=img_after,
            watermark=watermark,
            title=f"W-SVD水印嵌入效果（α={args.alpha}）",
            filename="1.1/comparison_with_watermark.png"
        )

    else:
        parser.error(f"未知模式: {args.mode}")


if __name__ == "__main__":
    run_args = [
        "embed",  # 模式
        "--cover", "pic.bmp",  # 输入图片路径
        "--out", "1.1/out.bmp",  # 输出图片路径
        "--alpha", "0.5",  # 水印强度
        "--seed", "42",  # 设定密钥种子
        "--wavelet", "db1",
        "--level", "1",
        "--ratio", "0.8"
    ]
    main(run_args)