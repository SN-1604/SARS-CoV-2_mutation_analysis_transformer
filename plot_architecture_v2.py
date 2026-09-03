# -*- coding: utf-8 -*-
"""plot_architecture_v2.py — v2 口径两个模型的网络结构示意图
风格贴合 Vaswani et al. (2017) Transformer 结构图: 粗边彩色层块 + 灰底
堆叠大框 + 粗连线与实心大箭头 + 残差绕线 + ⊕/Add 汇合 + 裸文字输入输出。
输出 (300 dpi, 覆盖同名旧文件):
  figure_monthly_revised/supplementary/figS9_repr_architecture.png
      预训练表征模型 (repr13_best.pt, v4_supcon: Pre-LN Transformer 编码器)
  figure_monthly_revised/supplementary/figS10_forecast_architecture_v2.png
      月度熵预测模型 v2 (双成员 resv2 集成 + 逐国仿射校准)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 8,
})

OUT = "figure_monthly_revised/supplementary"
os.makedirs(OUT, exist_ok=True)

PINK = "#F6DADD"      # 输入投影 (对应参考图 Embedding)
YELLOW = "#EDF3C7"    # LayerNorm / Add
ORANGE = "#F9DFB5"    # Attention
BLUE = "#D3E7F1"      # FFN
PURPLE = "#DBDBE8"    # Linear / 池化 / 平均
GREEN = "#D6E8D4"     # 输出 (对应参考图 Softmax)
GRAY = "#EDEDED"      # 拼接
BOXGRAY = "#F4F4F4"   # 大框
WIRE = "#222222"
LW, BLW, MS = 1.7, 1.5, 14      # 线宽 / 块边宽 / 箭头大小


def canvas(w_in, h_in, ymax, title, xmax=100):
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    ax.text(2, ymax - 1.6, title, ha="left", va="top", fontsize=8,
            color="#1a1a1a")
    return fig, ax


def block(ax, x, y, w, h, text, fc, fs=7.5, dashed=False, bold=False,
          ec="#222222"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.1",
                       fc=fc, ec=ec, lw=BLW, linestyle="--" if dashed else "-",
                       zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#1a1a1a", zorder=3, linespacing=1.4,
            weight="bold" if bold else "normal")


def plus(ax, x, y, r=2.0):
    ax.add_patch(Circle((x, y), r, fc="white", ec=WIRE, lw=BLW, zorder=3))
    ax.text(x, y, "+", ha="center", va="center", fontsize=10, color=WIRE,
            zorder=4, weight="bold")


def posenc_icon(ax, x, y, r=3.2):
    ax.add_patch(Circle((x, y), r, fc="white", ec=WIRE, lw=BLW, zorder=3))
    t = np.linspace(-0.75 * r, 0.75 * r, 60)
    ax.plot(x + t, y + 0.55 * r * np.sin(t / r * 2.2 * np.pi), color=WIRE,
            lw=1.3, zorder=4)


def wire(ax, xs, ys):
    """折线 (广播单值), 末段实心箭头"""
    if len(xs) == 1 and len(ys) > 1:
        xs = xs * len(ys)
    if len(ys) == 1 and len(xs) > 1:
        ys = ys * len(xs)
    ax.plot(xs, ys, color=WIRE, lw=LW, zorder=1.5, solid_capstyle="round")
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle="-|>", color=WIRE, lw=LW,
                                mutation_scale=MS), zorder=1.6)


# ---------------------------------------------------------------- figS9 预训练表征模型
def figS9():
    fig, ax = canvas(6.6, 8.3, 128,
                     "预训练表征模型结构：13 周 × 38 变量 → 3 层 Pre-LN Transformer 编码器 → 64 维表征")
    # ---- 底部输入 (裸文字 + Embedding 位粉块)
    ax.text(50, 2.4, "输入：13 周 × 38 变量（log1p + z 归一化）",
            ha="center", va="center", fontsize=7.5, color="#1a1a1a")
    block(ax, 32, 5, 36, 6, "线性投影 38→96", PINK)
    wire(ax, [50], [11, 13.4])
    plus(ax, 50, 15.6)
    posenc_icon(ax, 65.5, 15.6)
    ax.text(71, 15.6, "可学习位置\n编码（13×96）", ha="left", va="center",
            fontsize=6.5, color="#1a1a1a", linespacing=1.4)
    wire(ax, [62.3, 52.1], [15.6, 15.6])
    # ---- 编码器层大框 ×3
    ax.add_patch(FancyBboxPatch((26, 21), 48, 72.5,
                                boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=BOXGRAY, ec="#777777", lw=1.2, zorder=1))
    ax.text(77, 57, "3×", ha="left", va="center", fontsize=10,
            color="#1a1a1a")
    ax.text(72, 23, "Pre-LN · Dropout 0.1", ha="right", va="center",
            fontsize=6, color="#666666")
    block(ax, 33, 24.5, 34, 4.5, "LayerNorm", YELLOW, fs=7)
    block(ax, 33, 31.5, 34, 10, "Multi-Head\nSelf-Attention（4 头）", ORANGE)
    block(ax, 33, 44, 34, 4.5, "Add", YELLOW, fs=7)
    block(ax, 33, 51, 34, 4.5, "LayerNorm", YELLOW, fs=7)
    block(ax, 33, 58, 34, 10, "Feed Forward\n（384 维 · GELU）", BLUE)
    block(ax, 33, 70.5, 34, 4.5, "Add", YELLOW, fs=7)
    # 主干
    wire(ax, [50], [17.7, 24.5])
    wire(ax, [50], [29, 31.5])
    wire(ax, [50], [41.5, 44])
    wire(ax, [50], [48.5, 51])
    wire(ax, [50], [55.5, 58])
    wire(ax, [50], [68, 70.5])
    wire(ax, [50], [75, 96.5])
    # 残差绕线 (参照原图: 左绕自注意力, 右绕前馈)
    wire(ax, [47.9, 29.5, 29.5, 32.6], [15.6, 15.6, 46.2, 46.2])
    wire(ax, [52, 70.5, 70.5, 67.4], [46.2, 46.2, 72.7, 72.7])
    # ---- 顶部输出头
    block(ax, 32, 96.5, 36, 5.5, "时间维均值池化", PURPLE)
    wire(ax, [50], [102, 104.5])
    block(ax, 32, 104.5, 36, 5.5, "Linear 96→64 + LayerNorm", PURPLE, fs=7)
    wire(ax, [50], [110, 112.5])
    block(ax, 32, 112.5, 36, 5.5, "64 维周窗表征", GREEN, bold=True)
    ax.text(50, 121.5, "月度表征（按窗结束月聚合平均）",
            ha="center", va="center", fontsize=7.5, color="#1a1a1a")
    wire(ax, [50], [118, 119.7])
    # 训练头 (仅训练, 虚线)
    block(ax, 73, 99.5, 25, 11.5,
          "仅训练：\nLinear 64→141\nSoftmax 国别分类（CE）\n+ SupCon（τ=0.1）",
          "white", fs=6.5, dashed=True, ec="#888888")
    wire(ax, [68.4, 72.6], [107.2, 105])
    fig.savefig(os.path.join(OUT, "figS9_repr_architecture.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- figS10 熵预测模型 v2
def figS10():
    fig, ax = canvas(7.6, 7.5, 113,
                     "月度熵预测模型结构：双成员持续性残差网络集成 + 逐国仿射校准",
                     xmax=118)
    # ---- 底部输入 (裸文字)
    ax.text(59, 2.6, "输入：K 个月 × 67 维特征\n= 64 维月度表征 + 熵史 z + 观测掩码 + 熵差分",
            ha="center", va="center", fontsize=7.5, color="#1a1a1a",
            linespacing=1.5)
    for cx, name in ((30, "成员 A：K=5 · H=256"),
                     (86, "成员 B：K=3 · H=128")):
        # 成员大框 (实线灰边, 同参考图 N× 框样式)
        ax.add_patch(FancyBboxPatch((cx - 27, 7), 54, 61,
                                    boxstyle="round,pad=0.4,rounding_size=1.6",
                                    fc=BOXGRAY, ec="#777777", lw=1.2,
                                    zorder=0.5))
        ax.text(cx - 25, 65.6, name, ha="left", va="center", fontsize=7.5,
                color="#1a1a1a", weight="bold")
        # 表征支路 (左, 窄块为直通线让位)
        wire(ax, [cx - 12], [5.6, 11])
        block(ax, cx - 21, 11, 21, 5.5, "Linear 64→H\nGELU", PURPLE, fs=6.5)
        wire(ax, [cx - 12], [16.5, 19])
        block(ax, cx - 21, 19, 21, 5.5, "Linear H→32\n展平 K×32", PURPLE,
              fs=6.5)
        wire(ax, [cx - 12], [24.5, 27.5])
        # 熵特征直通 (右)
        wire(ax, [cx + 6], [5.6, 27.5])
        ax.text(cx + 8.5, 20, "熵史 z\n掩码\n差分", ha="left", va="center",
                fontsize=6, color="#555555", linespacing=1.35)
        # 拼接
        block(ax, cx - 21, 27.5, 36, 5, "拼接 → K×35 维", GRAY)
        wire(ax, [cx - 4], [32.5, 35])
        # 融合 MLP
        block(ax, cx - 21, 35, 34, 6.5, "Linear K·35→H\nGELU · Dropout 0.1",
              PURPLE, fs=7)
        wire(ax, [cx - 4], [41.5, 44])
        block(ax, cx - 21, 44, 34, 6.5, "Linear H→1\n（输出层零初始化）",
              PURPLE, fs=7)
        wire(ax, [cx - 4], [50.5, 53.5])
        # Add 汇合: 校正项 + 持续性基线 (残差绕线自右侧进入)
        block(ax, cx - 21, 53.5, 34, 4.5, "Add", YELLOW, fs=7)
        wire(ax, [cx + 6, cx + 24, cx + 24, cx + 13.4], [8, 8, 55.7, 55.7])
        ax.text(cx + 20.5, 40, "持续性基线（最近有效月熵）", ha="center",
                va="center", fontsize=6, color="#555555", rotation=90)
        # 出塔
        wire(ax, [cx - 4], [58, 68])
    # ---- 集成与校准
    wire(ax, [26, 46, 52.5], [68, 73, 75.5])
    wire(ax, [82, 72, 65.5], [68, 73, 75.5])
    block(ax, 42, 75.5, 34, 5.5, "两成员简单平均", PURPLE)
    wire(ax, [59], [81, 83.5])
    block(ax, 42, 83.5, 34, 5.5, "逆变换 expm1(z·σ+μ)", PURPLE, fs=7)
    wire(ax, [59], [89, 91.5])
    block(ax, 37, 91.5, 44, 6,
          "逐国仿射校准 ŷ′=a+b·ŷ\n（λ=1.0，<5 训练点回退恒等）", PURPLE, fs=6.5)
    wire(ax, [59], [97.5, 100])
    block(ax, 42, 100, 34, 5.5, "月熵预测值 ŷ′", GREEN, bold=True)
    ax.text(59, 108, "输出：目标月核苷酸序列香农熵",
            ha="center", va="center", fontsize=7.5, color="#1a1a1a")
    wire(ax, [59], [105.5, 106.8])
    fig.savefig(os.path.join(OUT, "figS10_forecast_architecture_v2.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    figS9()
    figS10()
    print("DONE ->", OUT)
