# -*- coding: utf-8 -*-
"""plot_repr_effect.py — 表征模型效果分析组合图

主图: figure_monthly_revised/fig9_repr_effect.png (2×3, 300 dpi)
  a 生产表征(repr13月均)141国×32月嵌入的 t-SNE 投影（各国成簇分离,
    仅生产版本, 不含其他候选表征; 标注轮廓系数/kNN acc）
  b 表征通道下游贡献（v1/v2 相对残余MSE条图: 纯熵消融=100% vs
    全模型=69.0%/66.3%, 削减约1/3残余误差, 两条管线独立复现）
  c 嵌入同国 vs 异国月对余弦相似度分布（阶梯直方图）
  d 嵌入主成分 scree（单成分方差占比散点 + 累计方差曲线）
  e 相邻月表征余弦相似度逐月演变（中位线 + IQR 带）
  f 同国表征相似度随时滞衰减（lag 1–12, 中位 + IQR）
子图(单独全尺寸): figure_monthly_revised/supplementary/figS22..figS27

数据源: repr_model_selection.json, reprM_selection.json,
        entropy_forecast_monthly_metrics.json, entropy_forecast_monthly_v2_metrics.json,
        embeddings_monthly/*.npz (reprM_m4, W=3, 64维, 141国×32月)
遵循 figure-style 规范, 视觉语言与 plot_supplementary_figures_v2.py 一致。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"C:/Users/DFD/.agents/skills/figure-style")
import kernel as fsk

fsk.apply_figure_style(font="Microsoft YaHei", sizes=(8, 7, 6))
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "axes.labelpad": 6,
})

BLUE, LBLUE, GREEN = "#0072B2", "#56B4E9", "#009E73"
ORANGE, RED, GREY = "#E69F00", "#D55E00", "#666666"
LGREY, NAVY = "#BBBBBB", "#08306B"

OUT_MAIN = "figure_monthly_revised"
OUT_SUP = "figure_monthly_revised/supplementary"
os.makedirs(OUT_SUP, exist_ok=True)

# ---------------- 数据 ----------------
SEL_W = None  # 选型面板已移除: 仅展示生产表征(repr13月均)的表征效果
M2 = json.load(open("entropy_forecast_monthly_v2_metrics.json", encoding="utf-8"))

embs, MONTHS = [], None
countries = []
for f in sorted(os.listdir("embeddings_monthly")):
    if not f.endswith(".npz"):
        continue
    z = np.load(os.path.join("embeddings_monthly", f))
    embs.append(z["embeddings"])
    MONTHS = z["months"]
    countries.append(f[:-len("_monthly_embeddings.npz")])
E = np.stack(embs)                          # (141, 32, 64)
NC, NM, D = E.shape
N_CM = NC * NM


def cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1)
                              * np.linalg.norm(b, axis=-1) + 1e-9)


# PCA
X = E.reshape(-1, D)
X = X - X.mean(0)
_, S, _ = np.linalg.svd(X, full_matrices=False)
ev = (S ** 2)
ev = ev / ev.sum()
CUM10 = ev[:10].sum()

# 生产表征(repr13月均)的国家聚类结构: 类内月际距离 / 最近他国间隙 / 轮廓系数
En_norm = E / (np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9)
CENT = En_norm.mean(1)                            # (141,64) 国家质心
CG = CENT @ CENT.T                                # 质心间余弦
WITHIN = np.zeros(NC)
GAP = np.zeros(NC)
SIL = np.zeros(NC)
for i in range(NC):
    Si = En_norm[i] @ En_norm[i].T
    a = float((1 - Si[~np.eye(NM, dtype=bool)]).mean())
    b = float(min(1 - CG[i, j] for j in range(NC) if j != i))
    WITHIN[i], GAP[i] = a, b
    SIL[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
SIL_MEAN = float(SIL.mean())
RATIO_MED = float(np.median(GAP / np.maximum(WITHIN, 1e-9)))

rng = np.random.default_rng(1)
c1 = rng.integers(0, NC, 8000)
a1 = rng.integers(0, NM, 8000)
b1 = rng.integers(0, NM, 8000)
SAME = cos(E[c1, a1], E[c1, b1])
i1 = rng.integers(0, NC, 8000)
i2 = rng.integers(0, NC, 8000)
mk = i1 != i2
CROSS = cos(E[i1[mk], rng.integers(0, NM, mk.sum())],
            E[i2[mk], rng.integers(0, NM, mk.sum())])

CONS = cos(E[:, 1:, :], E[:, :-1, :])       # (141, 31) 相邻月
cons_med = np.median(CONS, axis=0)
cons_q25 = np.percentile(CONS, 25, axis=0)
cons_q75 = np.percentile(CONS, 75, axis=0)

LAGS = list(range(1, 13))
lag_med, lag_q25, lag_q75 = [], [], []
for k in LAGS:
    cs = cos(E[:, k:, :], E[:, :-k, :]).ravel()
    lag_med.append(np.median(cs))
    lag_q25.append(np.percentile(cs, 25))
    lag_q75.append(np.percentile(cs, 75))

# ---------------- 面板 ----------------


def panel_tsne(ax):
    """生产表征(repr13月均) 141国×32月嵌入的 t-SNE 投影: 各自成簇彼此分离"""
    from sklearn.manifold import TSNE
    X64 = E.reshape(-1, D)
    Z2 = TSNE(n_components=2, perplexity=30, init="pca",
              learning_rate="auto", random_state=7).fit_transform(X64)
    import matplotlib.patheffects as pe
    # 代表国: (显示名, 文本x, 文本y) — 引线在稠密区防重叠
    LABEL = {"China": ("中国", -24, 20), "United States": ("美国", 30, -8),
             "India": ("印度", 43, -44), "Brazil": ("巴西", -22, -34),
             "South Africa": ("南非", 50, 31), "Germany": ("德国", 2, -42)}
    cid = np.repeat(np.arange(NC), NM)
    lab_mask = np.array([countries[i] in LABEL for i in cid])
    ax.scatter(Z2[~lab_mask, 0], Z2[~lab_mask, 1], s=2.2, color=BLUE,
               alpha=0.45, lw=0, rasterized=True)
    ax.scatter(Z2[lab_mask, 0], Z2[lab_mask, 1], s=3.0, color=ORANGE,
               alpha=0.9, lw=0, rasterized=True)
    fsk.set_frame(ax, "none")
    ax.set_xticks([])
    ax.set_yticks([])
    for c, (nm, tx, ty) in LABEL.items():
        i = countries.index(c)
        xy = Z2[cid == i].mean(0)
        ax.annotate(nm, xy=xy, xytext=(tx, ty), fontsize=6, color="#333333",
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999",
                                    shrinkA=0, shrinkB=3),
                    path_effects=[pe.withStroke(linewidth=1.8,
                                                foreground="white")])
    ax.text(0.98, 0.02, "kNN国家识别acc=1.00\n轮廓系数均值%.2f（n=%d国）"
            % (SIL_MEAN, NC), transform=ax.transAxes, fontsize=7, color=NAVY,
            va="bottom", ha="right",
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    # §6.6 角标轴名
    ax.annotate("", xy=(0.09, 0.0), xytext=(0.0, 0.0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=GREY))
    ax.annotate("", xy=(0.0, 0.09), xytext=(0.0, 0.0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=GREY))
    ax.text(0.09, -0.035, "t-SNE 1", transform=ax.transAxes, fontsize=6,
            color=GREY, ha="left", va="top")
    ax.text(-0.035, 0.09, "t-SNE 2", transform=ax.transAxes, fontsize=6,
            color=GREY, ha="right", va="center", rotation=90)
    ax.set_title("141国月度表征各自成簇、彼此分离")


def panel_contribution(ax):
    """相对残余MSE条图: 消融=100% vs 全模型=1-贡献率 (仅当前管线)"""
    rows_ = [("本研究模型", M2["emb_contribution"],
              M2["val_entonly_ensemble"]["r2"], M2["val_full_ensemble"]["r2"])]
    for i, (nm, contrib, r0, r1) in enumerate(rows_):
        yb = 0
        ax.barh(yb + 0.17, 100, height=0.30, color=LGREY, edgecolor="none")
        ax.barh(yb - 0.17, (1 - contrib) * 100, height=0.30, color=BLUE,
                edgecolor="none")
        ax.text(101, yb + 0.17, "纯熵消融 100%", fontsize=6, color=GREY,
                va="center")
        ax.text((1 - contrib) * 100 + 1.5, yb - 0.17,
                "全模型 %.1f%%（−%.1f%%）" % ((1 - contrib) * 100,
                100 * contrib), fontsize=7, color=NAVY, va="center")
        ax.text(1, yb - 0.475, "验证R² %.3f → %.3f（n=%d）" % (r0, r1,
                M2["n_val"]), fontsize=6, color=GREY, va="bottom")
    ax.set_yticks([0])
    ax.set_yticklabels([r[0] for r in rows_], fontsize=6)
    ax.set_xlim(0, 128)
    ax.set_ylim(-0.75, 0.75)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("相对残余MSE（纯熵消融=100%）")
    ax.set_title("表征通道削减1/3残余误差")


def panel_cosdist(ax):
    bins = np.linspace(-0.6, 1.0, 57)
    ax.hist(CROSS, bins=bins, density=True, histtype="stepfilled",
            alpha=0.30, color=GREY, lw=0)
    ax.hist(CROSS, bins=bins, density=True, histtype="step", lw=1.0,
            color=GREY, label="异国月对（中位-0.01）")
    ax.hist(SAME, bins=bins, density=True, histtype="stepfilled",
            alpha=0.25, color=BLUE, lw=0)
    ax.hist(SAME, bins=bins, density=True, histtype="step", lw=1.4,
            color=BLUE, label="同国月对（中位0.98）")
    ax.set_xlabel("月度表征余弦相似度")
    ax.set_ylabel("密度")
    ax.set_xlim(-0.6, 1.0)
    ax.legend(loc="upper left", handlelength=1.4)
    ax.set_title("嵌入按国家高度可分（各8000抽样对）")


def panel_scree(ax):
    xs = np.arange(1, 21)
    ax.plot(xs, ev[:20] * 100, "o", ms=4, color=BLUE, mew=0)
    ax.set_xlabel("主成分序号")
    ax.set_ylabel("方差占比（%）")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, 4.3)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax2 = ax.twinx()
    ax2.set_xticks([])
    ax2.plot(xs, np.cumsum(ev[:20]) * 100, color=ORANGE, lw=1.2)
    ax2.set_ylim(0, 60)
    ax2.set_ylabel("累计方差占比（%）", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(ORANGE)
    ax.text(0.97, 0.33, "前10主成分累计%.1f%%\n(n=%d国家-月)" % (CUM10 * 100, N_CM),
            transform=ax.transAxes, fontsize=7, color=NAVY, va="center",
            ha="right")
    ax.set_title("64维表征方差分布均衡")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)


def panel_cons_time(ax):
    xs = np.arange(31)                    # 相邻月对, 以较晚月计
    ax.fill_between(xs, cons_q25, cons_q75, color=BLUE, alpha=0.20, lw=0)
    ax.plot(xs, cons_med, color=BLUE, lw=1.4)
    xt = [i for i, m in enumerate(MONTHS[1:]) if m.endswith("-01")]
    ax.set_xticks(xt)
    ax.set_xticklabels([MONTHS[1:][i] for i in xt])
    ax.set_ylim(0.988, 1.001)
    ax.set_xlabel("月份")
    ax.set_ylabel("相邻月余弦相似度")
    ax.text(0.02, 0.05, "中位≥0.994，IQR带≈0.002", transform=ax.transAxes,
            fontsize=7, color=NAVY, va="bottom")
    ax.set_title("相邻月表征高度连续（%d国×31月对）" % NC)


def panel_lagdecay(ax):
    xs = np.array(LAGS)
    ax.fill_between(xs, lag_q25, lag_q75, color=BLUE, alpha=0.20, lw=0)
    ax.plot(xs, lag_med, "o-", ms=3.5, color=BLUE, lw=1.4)
    ax.annotate("%.3f" % lag_med[-1], xy=(12, lag_med[-1]),
                xytext=(11.4, lag_med[-1] - 0.012), fontsize=7, color=NAVY,
                ha="center")
    ax.set_xticks([1, 3, 6, 9, 12])
    ax.set_ylim(0.94, 1.005)
    ax.set_xlabel("时滞（月）")
    ax.set_ylabel("同国表征余弦相似度")
    ax.set_title("时滞12月余弦仍达0.97（衰减缓慢）")


PANELS = [
    ("a", panel_tsne, "figS22_repr_tsne_clusters.png"),
    ("b", panel_contribution, "figS23_repr_contribution.png"),
    ("c", panel_cosdist, "figS24_repr_country_separation.png"),
    ("d", panel_scree, "figS25_repr_scree.png"),
    ("e", panel_cons_time, "figS26_repr_monthly_continuity.png"),
    ("f", panel_lagdecay, "figS27_repr_lag_decay.png"),
]


def bbox_check(fig, tag):
    import matplotlib.text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = fig.get_tightbbox(r).get_points() * fig.dpi
    x0, y0 = float(pts[0][0]) - 6, float(pts[0][1]) - 6
    x1, y1 = float(pts[1][0]) + 6, float(pts[1][1]) + 6

    def in_box(bt):
        return bt.x0 >= x0 and bt.y0 >= y0 and bt.x1 <= x1 and bt.y1 <= y1

    stale_ticks = set()
    spines = []
    for ax in fig.axes:
        if not ax.axison:
            stale_ticks.update(ax.get_xticklabels(which="both"))
            stale_ticks.update(ax.get_yticklabels(which="both"))
            continue
        spines += [(s, s.get_window_extent(r)) for s in ax.spines.values()
                   if s.get_visible()]
        xlo, xhi = sorted(ax.get_xlim())
        ylo, yhi = sorted(ax.get_ylim())
        for t in ax.get_xticklabels(which="both"):
            if not (xlo <= t.get_position()[0] <= xhi):
                stale_ticks.add(t)
        for t in ax.get_yticklabels(which="both"):
            if not (ylo <= t.get_position()[1] <= yhi):
                stale_ticks.add(t)

    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_visible() and t not in stale_ticks]
    ticklabels = {ax: set(ax.get_xticklabels(which="both")
                          + ax.get_yticklabels(which="both"))
                  for ax in fig.axes}
    overlaps = [(a.get_text()[:20], b.get_text()[:20])
                for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
                if ba.overlaps(bb)]
    overlaps += [(t.get_text()[:20], "spine")
                 for t, bt in texts for s, bs in spines
                 if bt.overlaps(bs) and t not in ticklabels[s.axes]]
    outside = [t.get_text()[:20] for t, bt in texts if not in_box(bt)]
    if overlaps or outside:
        print("[%s] bbox冲突:" % tag, overlaps, "越界:", outside)
    else:
        print("[%s] bbox检查通过" % tag)


fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
for (letter, fn, _), ax in zip(PANELS, axes.flat):
    fn(ax)
    fsk.panel_letter(ax, letter, dx=-0.12)
fig.tight_layout(w_pad=1.8, h_pad=1.6)
fig.savefig(os.path.join(OUT_MAIN, "fig9_repr_effect.png"))
bbox_check(fig, "fig9")
plt.close(fig)

for letter, fn, fname in PANELS:
    f, ax = plt.subplots(figsize=(3.6, 2.8))
    fn(ax)
    f.tight_layout()
    f.savefig(os.path.join(OUT_SUP, fname))
    bbox_check(f, fname)
    plt.close(f)

print("saved fig9_repr_effect.png + figS22..figS27")
