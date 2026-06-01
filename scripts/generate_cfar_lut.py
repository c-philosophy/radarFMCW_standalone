#!/usr/bin/env python
"""CFAR alpha 查找表（LUT）生成脚本。

通过蒙特卡洛仿真预计算 4 种 CFAR 变体在不同 (guard_cells, reference_cells)
参数下的 alpha 值，保存为 .npz 文件供运行时快速查表。

用法:
    python scripts/generate_cfar_lut.py
    python scripts/generate_cfar_lut.py --trials 2000000
    python scripts/generate_cfar_lut.py --output custom_lut.npz
    python scripts/generate_cfar_lut.py --guards 1,2,3 --refs 4,8,12

依赖:
    numpy (必需), fmcw 包在 PYTHONPATH 中
"""

import argparse
import os
import sys
import time
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fmcw.detection.cfar_alpha import AlphaLUT


def main():
    parser = argparse.ArgumentParser(
        description="生成 CFAR alpha 蒙特卡洛查找表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/generate_cfar_lut.py
  python scripts/generate_cfar_lut.py --trials 2000000 --seed 123
  python scripts/generate_cfar_lut.py --guards 1,2,3,4 --refs 4,6,8,10,12,16
        """,
    )
    parser.add_argument(
        "--trials", type=int, default=1_000_000,
        help="每次 MC 仿真的试验次数（默认 1,000,000）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）"
    )
    parser.add_argument(
        "--guards", type=str, default="1,2,3,4",
        help="guard_cells 列表，逗号分隔（默认 1,2,3,4）"
    )
    parser.add_argument(
        "--refs", type=str, default="4,6,8,10,12,16",
        help="reference_cells 列表，逗号分隔（默认 4,6,8,10,12,16）"
    )
    parser.add_argument(
        "--pfas", type=str, default=None,
        help="Pfa 网格，逗号分隔（默认 25 个 log-spaced 点 1e-6→1e-1）"
    )
    parser.add_argument(
        "--ranks", type=str, default="0.5,0.6,0.75,0.8,0.9",
        help="OS-CFAR rank_ratio 列表（默认 0.5,0.6,0.75,0.8,0.9）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出文件路径（默认 fmcw/data/cfar_alpha_lut.npz）"
    )

    args = parser.parse_args()

    # 解析参数
    guards = [int(x.strip()) for x in args.guards.split(",")]
    refs = [int(x.strip()) for x in args.refs.split(",")]
    rank_ratios = [float(x.strip()) for x in args.ranks.split(",")]

    if args.pfas:
        pfa_grid = np.array([float(x.strip()) for x in args.pfas.split(",")])
    else:
        pfa_grid = np.logspace(-6, -1, 25)

    # 输出路径
    if args.output:
        save_path = args.output
    else:
        save_path = os.path.join(
            os.path.dirname(__file__), "..", "fmcw", "data", "cfar_alpha_lut.npz"
        )
    save_path = os.path.abspath(save_path)

    # 估计耗时
    n_combos = len(guards) * len(refs)
    n_mc_runs = n_combos * (3 + len(rank_ratios))  # CA, GO, SO + OS variants
    est_seconds = n_mc_runs * (args.trials / 1_000_000) * 0.15  # 粗略估计
    print(f"参数网格: guards={guards}, refs={refs}")
    print(f"Pfa 网格: {len(pfa_grid)} points, {pfa_grid[0]:.1e} → {pfa_grid[-1]:.1e}")
    print(f"OS rank_ratios: {rank_ratios}")
    print(f"MC 试验次数: {args.trials:,} per run")
    print(f"总 MC 运行数: {n_mc_runs}")
    print(f"预计耗时: ~{est_seconds:.0f} 秒")
    print(f"输出文件: {save_path}")
    print()

    t_start = time.perf_counter()

    AlphaLUT.generate(
        guards=guards,
        refs=refs,
        pfa_grid=pfa_grid,
        rank_ratios=rank_ratios,
        n_trials=args.trials,
        seed=args.seed,
        save_path=save_path,
    )

    t_elapsed = time.perf_counter() - t_start
    file_size_kb = os.path.getsize(save_path) / 1024
    print(f"\n完成! 耗时 {t_elapsed:.1f} 秒, 文件大小 {file_size_kb:.1f} KB")


if __name__ == "__main__":
    main()
