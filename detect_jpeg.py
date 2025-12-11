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


def scan_seeds(
    cover_path: str | Path,
    test_path: str | Path,
    alpha: float,
    wavelet: str,
    level: int,
    ratio: float,
    seed_start: int,
    seed_end: int,
    out_plot: str | Path | None = None,
) -> Tuple[List[int], np.ndarray]:
    _,cover_rgb = load_rgb_image_float01(cover_path)
    _,test_rgb = load_rgb_image_float01(test_path)

    seeds: List[int] = []
    ds_spatial: List[float] = []
    ds_dct: List[float] = []

    print(
        f"开始种子扫描：seed ∈ [{seed_start}, {seed_end}]，"
        f"alpha={alpha}, wavelet={wavelet}, level={level}, ratio={ratio}"
    )

    for s in range(seed_start, seed_end + 1):
        W, Wp = compute_watermark_templates(
            cover_rgb,
            test_rgb,
            alpha=alpha,
            seed=s,
            wavelet=wavelet,
            level=level,
            ratio=ratio,
        )
        d_spatial = normalized_correlation(W, Wp)

        W_dct = dctn(W, type=2, norm=None)
        Wp_dct = dctn(Wp, type=2, norm=None)

        h, w = W_dct.shape
        d_block = min(32, max(h, w))
        Wb = W_dct[:d_block, :d_block].copy()
        Wpb = Wp_dct[:d_block, :d_block].copy()
        Wb[0, 0] = 0.0
        Wpb[0, 0] = 0.0

        d_dct = normalized_correlation(Wb, Wpb)

        seeds.append(s)
        ds_spatial.append(d_spatial)
        ds_dct.append(d_dct)
        print(f"  seed={s:3d} -> d={d_spatial:.4f}, d^={d_dct:.4f}")

    ds_spatial_arr = np.array(ds_spatial, dtype=np.float64)
    ds_dct_arr = np.array(ds_dct, dtype=np.float64)

    abs_spatial = np.abs(ds_spatial_arr)
    abs_dct = np.abs(ds_dct_arr)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # # ------------------- 空域图（ax1） -------------------
    # ax1.plot(seeds, abs_spatial, marker="o", markersize=2, linewidth=1)
    # spatial_max_idx = np.argmax(abs_spatial)
    # spatial_max_seed = seeds[spatial_max_idx]
    # spatial_max_val = abs_spatial[spatial_max_idx]

    # # 添加标注
    # ax1.annotate(
    #     f"最大值: {spatial_max_val:.4f}\nseed: {spatial_max_seed}",
    #     xy=(spatial_max_seed, spatial_max_val),
    #     xytext=(spatial_max_seed + 5, spatial_max_val - (max(abs_spatial)-min(abs_spatial))*0.1),  # 按数据范围比例偏移
    #     ha="left",
    #     va="bottom",
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")
    # )

    # ax1.set_ylabel("相关性 d")
    # ax1.set_title("小波系数相关性分析（空域）")
    # ax1.grid(True, alpha=0.3)

    # # 自动计算空域数据范围，并强制设置刻度步长密度（保证图中高度紧凑）
    # spatial_min, spatial_max = min(abs_spatial), max(abs_spatial)
    # spatial_range = spatial_max - spatial_min
    # # 固定“数据范围/图高”的比例（让刻度在图中高度一致）
    # target_range_ratio = 0.1  # 控制紧凑度，值越小刻度越密集（图中高度越一致）
    # # 计算步长：让图中显示约5-8个刻度
    # spatial_step = spatial_range * target_range_ratio
    # # 设置刻度
    # ax1.set_ylim(spatial_min - spatial_range*0.05, spatial_max + spatial_range*0.05)  # 留边距
    # ax1.yaxis.set_major_locator(MultipleLocator(base=spatial_step))


    # # ------------------- DCT域图（ax2） -------------------
    # ax2.plot(seeds, abs_dct, marker="o", markersize=2, linewidth=1)
    # dct_max_idx = np.argmax(abs_dct)
    # dct_max_seed = seeds[dct_max_idx]
    # dct_max_val = abs_dct[dct_max_idx]

    # # 添加标注
    # ax2.annotate(
    #     f"最大值: {dct_max_val:.4f}\nseed: {dct_max_seed}",
    #     xy=(dct_max_seed, dct_max_val),
    #     xytext=(dct_max_seed + 5, dct_max_val - (max(abs_dct)-min(abs_dct))*0.1),
    #     ha="left",
    #     va="bottom",
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")
    # )

    # ax2.set_xlabel("种子")
    # ax2.set_ylabel("相关性 d^")
    # ax2.set_title("DCT 变换后小波系数相关性分析")
    # ax2.grid(True, alpha=0.3)

    # # 与空域图统一“数据范围/图高”的比例，保证刻度高度一致
    # dct_min, dct_max = min(abs_dct), max(abs_dct)
    # dct_range = dct_max - dct_min
    # dct_step = dct_range * target_range_ratio  # 复用空域的紧凑度比例
    # ax2.set_ylim(dct_min - dct_range*0.05, dct_max + dct_range*0.05)
    # ax2.yaxis.set_major_locator(MultipleLocator(base=dct_step))


    # # ------------------- 图整体设置 -------------------
    # fig.suptitle("“种子-相关性值”SC 图（空域与 DCT 域）")
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # if out_plot is not None:
    #     out_plot = Path(out_plot)
    #     out_plot.parent.mkdir(parents=True, exist_ok=True)
    #     fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    #     print(f"SC 曲线已保存到: {out_plot}")
    # else:
    #     plt.show()

    # plt.close(fig)
    return seeds, ds_spatial_arr

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


# if __name__ == "__main__":
    
#     for j in [0.1,0.9]:
#         coef=[]
#         DCTcoef=[]
#         for i in range(10,101,10):

#             detect_args = [
#                 "detect",  # 模式
#                 "--cover", "pic.bmp",  # 输入图片路径
#                 #"--test", "StirMarkBenchmark_4_0_129/Media/Output/Images/Set1/lenna-waved_JPEG_60.bmp",  # 测试图片路径
#                 "--test",f"alpha/alpha_{j}_JPEG_{i}.bmp",
#                 "--alpha", f"{j}",  # 水印强度 (0.1 较隐蔽)
#                 "--seed", "42",  # 设定密钥种子
#                 "--wavelet", "db1",
#                 "--level", "1",
#                 "--ratio", "0.8",
#             ]

#             corr_coef,corr_dctcoef=main(detect_args)
#             coef.append(corr_coef)
#             DCTcoef.append(corr_dctcoef)
#         x=range(10,101,10)
#         plt.figure()  # 初始化新画布，清空上一轮数据
#         plt.plot(x,coef,label='空域相关性')
#         plt.plot(x,DCTcoef,label='DCT域相关性')
#         plt.scatter(x,coef)
#         plt.scatter(x,DCTcoef)
#         plt.grid(axis='y',linestyle='--',color='r',alpha=0.6)
#         plt.title(f'α={j} JPEG攻击下的水印检测效果')
#         plt.xlabel('JPEG质量因子')
#         plt.ylabel('相关性系数')
        
#         plt.legend()
#         plt.savefig(f'analyze/jpeg_detection_alpha_{j}.png',dpi=300)
#         plt.close()  # 关闭画布，释放资源

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 定义要测试的α值列表
    alpha_list = ["db6", "haar"]
    # 初始化字典存储不同α的相关性数据
    data = {
        "db6": {"coef": [], "DCTcoef": []},
        "haar": {"coef": [], "DCTcoef": []}
    }
    
    # 第一步：遍历所有α和JPEG质量因子，收集数据
    for j in alpha_list:
        coef = []
        DCTcoef = []
        for i in range(10, 101, 10):
            detect_args = [
                "detect",  # 模式
                "--cover", "pic.bmp",  # 输入图片路径
                "--test", f"wavelet/wavelet_{j}_JPEG_{i}.bmp",
                "--alpha", "0.5",  # 水印强度
                "--seed", "42",  # 密钥种子
                "--wavelet", f"{j}",
                "--level", "1",
                "--ratio", "0.8",
            ]
            # 调用检测函数获取相关性系数
            corr_coef, corr_dctcoef = main(detect_args)
            coef.append(corr_coef)
            DCTcoef.append(corr_dctcoef)
        # 保存当前α的所有数据
        data[j]["coef"] = coef
        data[j]["DCTcoef"] = DCTcoef
    
    # 第二步：创建1行2列的子图，绘制所有数据
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1行2列，画布宽12高5
    x = range(10, 101, 10)  # JPEG质量因子x轴
    
    # 绘制第一个子图（α=0.1）
    ax1.plot(x, data["db6"]["coef"], label='空域相关性', marker='o', markersize=4)
    ax1.plot(x, data["db6"]["DCTcoef"], label='DCT域相关性', marker='o', markersize=4)
    ax1.grid(axis='y', linestyle='--', color='r', alpha=0.6)
    ax1.set_title(f'db6 JPEG攻击下的水印检测效果')
    ax1.set_xlabel('JPEG质量因子')
    ax1.set_ylabel('相关性系数')
    ax1.legend()
    
    # 绘制第二个子图（α=0.9）
    ax2.plot(x, data["haar"]["coef"], label='空域相关性', marker='o', markersize=4)
    ax2.plot(x, data["haar"]["DCTcoef"], label='DCT域相关性', marker='o', markersize=4)
    ax2.grid(axis='y', linestyle='--', color='r', alpha=0.6)
    ax2.set_title(f'haar JPEG攻击下的水印检测效果')
    ax2.set_xlabel('JPEG质量因子')
    ax2.set_ylabel('相关性系数')
    ax2.legend()
    
    # 调整子图间距，避免标题/标签重叠
    plt.tight_layout()
    # 保存合并后的图
    plt.savefig(f'analyze/jpeg_detection_wavelet.png', dpi=300, bbox_inches='tight')
    # 显示图片（可选）
    plt.show()
    plt.close(fig)
    
