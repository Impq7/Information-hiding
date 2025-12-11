from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn
import pywt
from matplotlib.ticker import MultipleLocator
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

from wsvd import load_rgb_image_float01, wavemarksvd_like


def normalized_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_flat = x.ravel().astype(np.float64)
    y_flat = y.ravel().astype(np.float64)

    num = float(np.dot(x_flat, y_flat))
    denom = float(np.linalg.norm(x_flat) * np.linalg.norm(y_flat))
    if denom == 0.0:
        return 0.0
    return num / denom


def compute_watermark_templates(
    cover_rgb: np.ndarray,
    test_rgb: np.ndarray,
    alpha: float,
    seed: int,
    wavelet: str,
    level: int,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    cover_r = cover_rgb[:, :, 0]
    test_r = test_rgb[:, :, 0]

    embed_result = wavemarksvd_like(
        cover_rgb,
        alpha=alpha,
        seed=seed,
        wavelet=wavelet,
        level=level,
        ratio=ratio,
    )
    waterCA = embed_result["waterCA"]

    coeffs_test = pywt.wavedec2(test_r, wavelet, level=level)
    CA_test = coeffs_test[0]

    coeffs_real = pywt.wavedec2(cover_r, wavelet, level=level)
    realCA = coeffs_real[0]

    realwatermark = waterCA - realCA
    testwatermark = CA_test - realCA
    return realwatermark, testwatermark


def detect_once(
    cover_path: str | Path,
    test_path: str | Path,
    alpha: float,
    seed: int,
    wavelet: str = "db1",
    level: int = 1,
    ratio: float = 0.8,
)-> Tuple[float, float]:
    _,cover_rgb = load_rgb_image_float01(cover_path)
    _,test_rgb = load_rgb_image_float01(test_path)

    W, Wp = compute_watermark_templates(
        cover_rgb,
        test_rgb,
        alpha=alpha,
        seed=seed,
        wavelet=wavelet,
        level=level,
        ratio=ratio,
    )
    corr_coef = normalized_correlation(W, Wp)

    W_dct = dctn(W, type=2, norm=None)
    Wp_dct = dctn(Wp, type=2, norm=None)

    h, w = W_dct.shape
    d_block = min(32, max(h, w))
    Wb = W_dct[:d_block, :d_block].copy()
    Wpb = Wp_dct[:d_block, :d_block].copy()
    Wb[0, 0] = 0.0
    Wpb[0, 0] = 0.0

    corr_DCTcoef = normalized_correlation(Wb, Wpb)

    print(f"检测 seed={seed} 的结果：")
    print(f"  小波系数相关性 corr_coef     = {corr_coef:.6f}")
    print(f"  DCT 后小波系数相关性 corr_DCT = {corr_DCTcoef:.6f}")
    return corr_coef, corr_DCTcoef


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_detect = subparsers.add_parser("detect")
    p_detect.add_argument("--cover", required=True)
    p_detect.add_argument("--test", required=True)
    p_detect.add_argument("--alpha", type=float, default=0.1)
    p_detect.add_argument("--seed", type=int, default=1)
    p_detect.add_argument("--wavelet", type=str, default="db1")
    p_detect.add_argument("--level", type=int, default=1)
    p_detect.add_argument("--ratio", type=float, default=0.8)

    p_scan = subparsers.add_parser("scan")
    p_scan.add_argument("--cover", required=True)
    p_scan.add_argument("--test", required=True)
    p_scan.add_argument("--alpha", type=float, default=0.1)
    p_scan.add_argument("--wavelet", type=str, default="db1")
    p_scan.add_argument("--level", type=int, default=1)
    p_scan.add_argument("--ratio", type=float, default=0.8)
    p_scan.add_argument("--seed-start", type=int, default=1)
    p_scan.add_argument("--seed-end", type=int, default=20)
    p_scan.add_argument("--out-plot", type=str, default=None)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "detect":

        corr_coef, corr_DCTcoef=detect_once(
            cover_path=args.cover,
            test_path=args.test,
            alpha=args.alpha,
            seed=args.seed,
            wavelet=args.wavelet,
            level=args.level,
            ratio=args.ratio,
        )

    return corr_coef, corr_DCTcoef

    

if __name__ == "__main__":
    coef=[]
    DCTcoef=[]
    for i in range(0,101,20):

        detect_args = [
            "detect",  # 模式
            "--cover", "pic.bmp",  # 输入图片路径
            #"--test", "StirMarkBenchmark_4_0_129/Media/Output/Images/Set1/lenna-waved_JPEG_60.bmp",  # 测试图片路径
            "--test",f"attacked/out_NOISE_{i}.bmp",
            "--alpha", "0.5",  # 水印强度 (0.1 较隐蔽)
            "--seed", "42",  # 设定密钥种子
            "--wavelet", "db1",
            "--level", "1",
            "--ratio", "0.8",
        ]

        corr_coef,corr_dctcoef=main(detect_args)
        coef.append(corr_coef)
        DCTcoef.append(corr_dctcoef)
    x=range(0,101,20)
    plt.grid(axis='y',linestyle='--',color='r',alpha=0.6)
    plt.plot(x,coef,label='空域相关性')
    plt.plot(x,DCTcoef,label='DCT域相关性')
    plt.scatter(x,coef)
    plt.scatter(x,DCTcoef)
    plt.title('加噪声攻击下的水印检测效果')
    plt.xlabel('NOISE强度')
    plt.ylabel('相关性系数')
    
    plt.legend()
    plt.savefig('2.1/noise_detection.png',dpi=300)
    plt.show()

