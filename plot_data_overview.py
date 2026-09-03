# -*- coding: utf-8 -*-
"""plot_data_overview.py — 数据层面可视化子图（数据分布 / 数据质量 / 数据清洗）

组合图: figure_monthly_revised/fig8_data_overview.png 由 compose_fig8.py 拼接
        (figure-composer, 输入为下方的 figS14..figS19 调整版子图)。
子图(单独全尺寸, 300 dpi): figure_monthly_revised/supplementary/figS14..figS21
  figS14 全球月度序列数时间分布面积图（2020-01–2022-12, 峰值与训练/验证分界）
  figS15 国家序列量 rank-size 对数-对数散点（长尾, 前5国标注, 前15国份额带）
  figS16 数据集筛选横向条图漏斗（219→140→124→111→106 国）
  figS17 国家×月熵观测覆盖热图 + 逐月观测国家比例边际曲线（n=219 国, 增强版）
  figS18 每国熵观测月数 ECDF 曲线（≥15 月建模门槛, 阴影达标区）
  figS19 分年份国家-月序列数小提琴图（含中位数与 <10 条质量线）
  figS20 熵值平滑前后分布对比 + 美国示例轨迹（数据处理补充）
  figS21 全球地图各国熵观测月数分布（choropleth, 小岛屿/属地为圆点）

数据源: seqcounts_monthly.csv, entropy_monthly/, entropy_monthly_smoothed/,
        owid_weekly_panel.pkl, entropy_forecast_monthly_v2_metrics.json,
        val_pred_monthly_v2.npz, ne_110m_admin_0_countries.geojson
遵循 figure-style 规范, 视觉语言与 plot_supplementary_figures_v2.py 一致。
"""
import csv
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

sys.path.insert(0, r"C:/Users/DFD/.agents/skills/figure-style")
import kernel as fsk

fsk.apply_figure_style(font="Microsoft YaHei", sizes=(8, 7, 6))
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "axes.labelpad": 6,
})

BLUE, LBLUE, GREEN = "#0072B2", "#56B4E9", "#009E73"
ORANGE, RED, PURPLE, GREY = "#E69F00", "#D55E00", "#CC79A7", "#666666"
NAVY = "#08306B"
RAMP5 = ["#084594", "#2171B5", "#4292C6", "#6BAED6", "#9ECAE1"]

OUT_MAIN = "figure_monthly_revised"
OUT_SUP = "figure_monthly_revised/supplementary"
os.makedirs(OUT_SUP, exist_ok=True)

MONTHS36 = ["%04d-%02d" % (y, m) for y in (2020, 2021, 2022) for m in range(1, 13)]

# ---------------- 数据汇总 ----------------
rows = list(csv.DictReader(open("seqcounts_monthly.csv", encoding="utf-8")))
countries = sorted({r["country"] for r in rows})
NC = len(countries)
month_tot = {m: 0 for m in MONTHS36}
pre2020_n = 0
country_tot = {}
nseq = np.zeros(len(rows), dtype=int)
nseq_year = {"2020": [], "2021": [], "2022": []}
for i, r in enumerate(rows):
    v = int(r["n_seq"])
    nseq[i] = v
    country_tot[r["country"]] = country_tot.get(r["country"], 0) + v
    if r["month"] in month_tot:
        month_tot[r["month"]] += v
        nseq_year[r["month"][:4]].append(v)
    else:
        pre2020_n += v                       # 2013-07 与 2019-12 共 5 条
TOT = int(nseq.sum())

panel = pickle.load(open("owid_weekly_panel.pkl", "rb"))
n_owid = len(panel)
ent_c = [f[:-len("_monthly_entropy.npy")]
         for f in os.listdir("entropy_monthly") if f.endswith(".npy")]
obs = {}
raw_vals, sm_vals = [], []
for c in ent_c:
    d = np.load(os.path.join("entropy_monthly", c + "_monthly_entropy.npy"),
                allow_pickle=True).item()
    obs[c] = d
    raw_vals += list(d.values())
    s = np.load(os.path.join("entropy_monthly_smoothed",
                             c + "_monthly_entropy.npy"), allow_pickle=True).item()
    sm_vals += list(s.values())
raw_vals = np.array(raw_vals)
sm_vals = np.array(sm_vals)
inter = set(panel) & set(ent_c)
n_inter = len(inter)
n_ge15 = sum(1 for c in inter if len(obs[c]) >= 15)

MET = json.load(open("entropy_forecast_monthly_v2_metrics.json", encoding="utf-8"))
NTR, NV = MET["n_train"], MET["n_val"]
N_EVAL = MET["val_final"]["n_countries_eval"]
VP = np.load("val_pred_monthly_v2.npz", allow_pickle=True)
N_VALC = len(set(VP["countries"].tolist()))

n_obs = np.array([len(obs[c]) for c in countries])
cover = np.zeros((NC, 36), dtype=bool)
for i, c in enumerate(countries):
    for j, m in enumerate(MONTHS36):
        cover[i, j] = m in obs[c]
order = np.argsort(-n_obs)
cover = cover[order]
cover_frac_month = cover.mean(axis=0)      # 逐月有观测国家比例

ranked = np.array(sorted(country_tot.values(), reverse=True))
share15 = ranked[:15].sum() / TOT
rank_of = {c: i + 1 for i, c in
           enumerate(sorted(country_tot, key=lambda k: -country_tot[k]))}
TOP5 = sorted(country_tot, key=lambda k: -country_tot[k])[:5]
CN5 = {"United States": "美国", "United Kingdom": "英国", "Germany": "德国",
       "Denmark": "丹麦", "France": "法国"}

# ---------------- 面板绘制函数 ----------------


def panel_timeline(ax):
    vals = np.array([month_tot[m] for m in MONTHS36], dtype=float)
    x = np.arange(36)
    ax.fill_between(x, vals, color=BLUE, alpha=0.30, lw=0)
    ax.plot(x, vals, color=BLUE, lw=1.3)
    ax.axvline(31.5, color=RED, lw=1.0, ls=(0, (4, 3)))
    ipk = int(vals.argmax())
    ax.annotate("%.1fk" % (vals[ipk] / 1e3), xy=(ipk, vals[ipk]),
                xytext=(ipk + 1.2, vals[ipk] * 1.05), ha="left", va="center",
                fontsize=7, color=NAVY)
    xt = [0, 12, 24]
    ax.set_xticks(xt)
    ax.set_xticklabels([MONTHS36[i] for i in xt])
    ax.set_ylabel("月度去重后序列数")
    ax.set_yticks([0, 25000, 50000, 75000, 100000])
    ax.set_yticklabels(["0", "25k", "50k", "75k", "100k"])
    ax.text(0.02, 0.97, "共1.44M条\n（早期5条未绘出）",
            transform=ax.transAxes, fontsize=7, va="top", color=GREY)
    ax.text(0.02, 0.70, "红虚线=训练/验证分界",
            transform=ax.transAxes, fontsize=7, va="top", color=RED)
    ax.set_title("月度序列量于2021年末达峰")
    ax.margins(x=0.01)
    ax.set_ylim(0, vals.max() * 1.14)


def panel_ranksize(ax):
    ranks = np.arange(1, NC + 1)
    ax.axvspan(0.8, 15.5, color=BLUE, alpha=0.08, lw=0)
    ax.plot(ranks, ranked, "o", ms=3.0, color=BLUE, mew=0, alpha=0.75)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(8, 4.2e3, "前15国\n76.7%", fontsize=7, color=NAVY,
            ha="center", va="center")
    offsets = {"United States": (2.6, 5.2e5), "United Kingdom": (11, 2.0e5)}
    for c in TOP5[:2]:
        rk = rank_of[c]
        v = country_tot[c]
        lab = "%s %dk" % (CN5.get(c, c), round(v / 1e3, -1))
        tx, ty = offsets[c]
        ax.annotate(lab, xy=(rk, v), xytext=(tx, ty), fontsize=6, color="#333333",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999",
                                    shrinkA=0, shrinkB=1.5))
    ax.set_xlim(0.8, 300)
    ax.set_ylim(0.5, 8e5)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1", "10", "100"])
    ax.set_yticks([1, 100, 1e4, 1e5])
    ax.set_yticklabels(["1", "100", "10k", "100k"])
    ax.set_xlabel("国家序列量排名（对数刻度）")
    ax.set_ylabel("去重后序列数（对数刻度）")
    ax.set_title("国家序列量呈长尾分布")


def panel_funnel(ax):
    counts = [NC, n_inter, n_ge15, N_VALC, N_EVAL]
    stages = ["spike序列国家/地区",
              "与OWID面板(%d国)取交集" % n_owid,
              "熵观测≥15月(建模样本3262个)",
              "验证期(2022-08起)有样本(477个)",
              "纳入逐国R²评估"]
    w = [c / counts[0] for c in counts] + [counts[-1] / counts[0]]
    ax.axis("off")
    for i in range(5):
        y0, y1 = 4.35 - i, 3.65 - i
        wt, wb = w[i], w[i + 1]
        ax.add_patch(Polygon([(-wt / 2, y0), (wt / 2, y0),
                              (wb / 2, y1), (-wb / 2, y1)],
                             closed=True, facecolor=RAMP5[i],
                             edgecolor="white", lw=0.8))
        tcol = "white" if i < 3 else NAVY
        ax.text(0, (y0 + y1) / 2, str(counts[i]), ha="center", va="center",
                fontsize=8, color=tcol, fontweight="bold")
        ax.text(-0.60, (y0 + y1) / 2, stages[i], ha="right", va="center",
                fontsize=6, color="#333333")
    ax.set_xlim(-1.75, 0.6)
    ax.set_ylim(-0.35, 4.7)
    ax.set_title("219国序列经筛选形成%d国建模集" % n_ge15, loc="left")


def panel_funnel_bar(ax):
    """figS16 用: 横向条图版数据集筛选漏斗"""
    stages = ["spike序列国家/地区",
              "与OWID面板\n(%d国)取交集" % n_owid,
              "熵观测≥15月\n(建模样本3262个)",
              "验证期(2022-08起)\n有样本(477个)",
              "纳入逐国\nR²评估"]
    counts = [NC, n_inter, n_ge15, N_VALC, N_EVAL]
    y = np.arange(len(stages))[::-1]
    ax.barh(y, counts, color=RAMP5, height=0.62, edgecolor="none")
    for yi, v in zip(y, counts):
        ax.text(v + 4, yi, str(v), va="center", fontsize=8, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=6)
    ax.set_xlabel("国家数")
    ax.set_xlim(0, NC * 1.18)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_title("219国序列经筛选形成%d国建模集" % n_ge15)
    ax.margins(y=0.04)


def panel_coverage(ax):
    cmap = LinearSegmentedColormap.from_list("cov", ["#F0F0F0", BLUE])
    ax.imshow(cover.astype(float), aspect="auto", cmap=cmap,
              interpolation="nearest")
    xt = [0, 12, 24]
    ax.set_xticks(xt)
    ax.set_xticklabels([MONTHS36[i] for i in xt])
    ax.set_yticks([])
    ax.set_xlabel("月份")
    ax.set_ylabel("国家（按观测月数降序）")
    ax.set_title("国家×月熵观测覆盖（n=%d国）" % NC)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)


def panel_ecdf(ax):
    xs = np.arange(1, 39)
    ecdf = np.array([(n_obs <= k).mean() for k in xs])
    ax.plot(xs, ecdf, color=BLUE, lw=1.6)
    ax.fill_between(xs, ecdf, 1, where=xs >= 15, color=BLUE, alpha=0.12, lw=0)
    ax.axvline(15, color=RED, lw=1.0, ls=(0, (4, 3)))
    f15 = (n_obs <= 15).mean()
    ax.plot([15], [f15], "o", ms=4, color=RED, mew=0, zorder=5)
    ax.text(16.2, 0.42, "≥15月：%d国（%.1f%%）" % ((n_obs >= 15).sum(),
            100 * (n_obs >= 15).mean()), fontsize=7, color=NAVY, va="center")
    ax.text(15.6, 0.10, "建模门槛15月", fontsize=7, color=RED,
            ha="left", va="bottom")
    ax.set_xlim(1, 38)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 10, 15, 20, 30, 38])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("每国熵观测月数")
    ax.set_ylabel("国家累积比例")
    ax.set_title("七成国家熵观测达15月门槛")


def panel_violin(ax):
    data = [np.log10(np.array(nseq_year[y])) for y in ("2020", "2021", "2022")]
    parts = ax.violinplot(data, positions=[0, 1, 2], widths=0.62,
                          showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], [LBLUE, BLUE, "#084594"]):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
        pc.set_edgecolor("none")
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.2)
    ax.axhline(1.0, color=RED, lw=1.0, ls=(0, (4, 3)))
    ax.text(2.54, 1.0, "<10条", fontsize=7, color=RED, ha="left",
            va="center", clip_on=False)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["2020\n(n=%d)" % len(nseq_year["2020"]),
                        "2021\n(n=%d)" % len(nseq_year["2021"]),
                        "2022\n(n=%d)" % len(nseq_year["2022"])], fontsize=6)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["1", "10", "100", "1k", "10k"])
    ax.set_ylabel("国家-月去重后序列数")
    ax.set_xlim(-0.5, 2.5)
    ax.set_title("2020年国家-月序列数偏低（中位7条）")


PANELS = [
    ("a", panel_timeline, "figS14_data_timeline.png"),
    ("b", panel_ranksize, "figS15_data_rank_size.png"),
    ("c", panel_funnel, "figS16_data_funnel.png"),
    ("d", panel_coverage, "figS17_entropy_coverage_heatmap.png"),
    ("e", panel_ecdf, "figS18_entropy_observed_ecdf.png"),
    ("f", panel_violin, "figS19_seqcount_violin.png"),
]


def bbox_check(fig, tag):
    import matplotlib.text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # get_tightbbox 返回英寸坐标(带 1/dpi 变换), 换算到显示像素再判断包含
    pts = fig.get_tightbbox(r).get_points() * fig.dpi
    x0, y0 = float(pts[0][0]) - 6, float(pts[0][1]) - 6
    x1, y1 = float(pts[1][0]) + 6, float(pts[1][1]) + 6

    def in_box(bt):
        return bt.x0 >= x0 and bt.y0 >= y0 and bt.x1 <= x1 and bt.y1 <= y1

    # 视野外的刻度标签不会被绘制(其bbox为陈旧值), 不参与检查;
    # axison=False 的坐标轴(漏斗图)的刻度与脊线同样不绘制
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


# ---------------- 主组合图 ----------------
# 组合图改由 compose_fig8.py 负责: 以下方保存的子图(调整版 standalone PNG)
# 经 figure-composer 拼接为 figure_monthly_revised/fig8_data_overview.png。
# 本脚本只生成子图; 重新生成子图后请运行 compose_fig8.py 重排组合图。

# ---------------- 子图（单独全尺寸） ----------------
for letter, fn, fname in PANELS:
    if letter == "d":
        continue                       # figS17 用增强版（见下）
    if letter == "c":
        fn = panel_funnel_bar          # figS16 用横向条图版漏斗
    f, ax = plt.subplots(figsize=(3.6, 2.8))
    fn(ax)
    f.tight_layout()
    f.savefig(os.path.join(OUT_SUP, fname))
    bbox_check(f, fname)
    plt.close(f)

# figS17 增强版: 逐月观测国家比例边际曲线 + 覆盖热图
f, (axt, axh) = plt.subplots(2, 1, figsize=(4.2, 3.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 3.4],
                                              hspace=0.08))
xs = np.arange(36)
axt.fill_between(xs, cover_frac_month, color=GREEN, alpha=0.25, lw=0)
axt.plot(xs, cover_frac_month, color=GREEN, lw=1.2)
axt.set_ylim(0, 1.0)
axt.set_yticks([0, 0.5, 1.0])
axt.set_yticklabels(["0", "50%", "100%"])
axt.set_ylabel("有观测\n国家比例", fontsize=6)
axt.set_title("国家×月熵观测覆盖与逐月观测国家比例（n=%d国）" % NC)
cmap = LinearSegmentedColormap.from_list("cov", ["#F0F0F0", BLUE])
axh.imshow(cover.astype(float), aspect="auto", cmap=cmap,
           interpolation="nearest")
xt = [0, 12, 24]
axh.set_xticks(xt)
axh.set_xticklabels([MONTHS36[i] for i in xt])
axh.set_yticks([])
axh.set_xlabel("月份")
axh.set_ylabel("国家（按观测月数降序）")
for s in axh.spines.values():
    s.set_visible(True)
    s.set_linewidth(0.6)
f.tight_layout()
f.savefig(os.path.join(OUT_SUP, "figS17_entropy_coverage_heatmap.png"))
bbox_check(f, "figS17")
plt.close(f)

# ---------------- figS20 熵平滑前后对比 ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7))
bins = np.linspace(0, 120, 49)
ax1.hist(raw_vals, bins=bins, density=True, histtype="stepfilled",
         alpha=0.35, color=GREY, lw=0)
ax1.hist(raw_vals, bins=bins, density=True, histtype="step", lw=1.0,
         color=GREY, label="原始熵")
ax1.hist(sm_vals, bins=bins, density=True, histtype="stepfilled",
         alpha=0.25, color=BLUE, lw=0)
ax1.hist(sm_vals, bins=bins, density=True, histtype="step", lw=1.4,
         color=BLUE, label="5点滑动平滑熵")
ax1.set_xlabel("月度熵（比特/月）")
ax1.set_ylabel("密度")
ax1.set_xlim(0, 120)
ax1.set_ylim(0, ax1.get_ylim()[1] * 1.08)
ax1.text(118, ax1.get_ylim()[1] * 0.5,
         "x轴截断于120\n(p99：原始83.9，平滑66.1)", fontsize=7, color=GREY,
         ha="right", va="center")
ax1.legend(loc="upper right", handlelength=1.4)
ax1.set_title("平滑压缩熵值极端波动（n=%d国家-月）" % len(raw_vals))
fsk.panel_letter(ax1, "a")

us_m = sorted(obs["United States"])
us_raw = [obs["United States"][m] for m in us_m]
us_sm_d = np.load("entropy_monthly_smoothed/United States_monthly_entropy.npy",
                  allow_pickle=True).item()
us_sm = [us_sm_d[m] for m in us_m]
xs = np.arange(len(us_m))
ax2.plot(xs, us_raw, "o", ms=2.5, lw=0, color=GREY, alpha=0.8, label="原始熵")
ax2.plot(xs, us_sm, "-", lw=1.4, color=BLUE, label="平滑熵")
xt = [i for i, m in enumerate(us_m) if m.endswith("-01")]
ax2.set_xticks(xt)
ax2.set_xticklabels([us_m[i] for i in xt])
ax2.set_xlabel("月份")
ax2.set_ylabel("月度熵（比特/月）")
ax2.set_title("美国月度熵：原始与平滑轨迹")
ax2.legend(loc="upper left", handlelength=1.4)
fsk.panel_letter(ax2, "b")
fig.tight_layout(w_pad=1.8)
fig.savefig(os.path.join(OUT_SUP, "figS20_entropy_smoothing.png"))
bbox_check(fig, "figS20")
plt.close(fig)

# ---------------- figS21 全球数据覆盖地图 ----------------
# Natural Earth 110m 国家边界(本地缓存 ne_110m_admin_0_countries.geojson);
# 该分辨率不含的 51 个小岛屿/属地在对应经纬度以同色标圆点表示。
GJ_ALIAS = {  # 本项目国家名 -> geojson 中的名称(ADMIN/NAME 等)
    "Cote d'Ivoire": "Ivory Coast",
    "Democratic Republic of Congo": "Democratic Republic of the Congo",
    "Eswatini": "eSwatini",
    "Timor": "East Timor",
}
GJ_POINTS = {  # 小岛屿/属地经纬度(近似质心)
    "American Samoa": (-170.7, -14.3), "Andorra": (1.6, 42.5),
    "Anguilla": (-63.07, 18.22), "Antigua and Barbuda": (-61.8, 17.1),
    "Aruba": (-70.0, 12.5), "Bahrain": (50.55, 26.07),
    "Barbados": (-59.55, 13.19), "Bermuda": (-64.77, 32.31),
    "Bonaire": (-68.27, 12.15), "British Virgin Islands": (-64.62, 18.42),
    "Cabo Verde": (-23.6, 15.1), "Canary Islands": (-15.6, 28.3),
    "Cayman Islands": (-81.25, 19.31), "Comoros": (43.87, -11.88),
    "Cook Islands": (-159.78, -21.23), "Crimea": (34.1, 45.3),
    "Curacao": (-68.93, 12.17), "Dominica": (-61.36, 15.42),
    "Faroe Islands": (-6.9, 61.9), "French Guiana": (-53.1, 3.9),
    "French Polynesia": (-149.4, -17.7), "Gibraltar": (-5.35, 36.13),
    "Grenada": (-61.68, 12.12), "Guadeloupe": (-61.58, 16.25),
    "Guam": (144.79, 13.44), "Kiribati": (-157.4, 1.87),
    "Liechtenstein": (9.55, 47.16), "Maldives": (73.5, 3.2),
    "Malta": (14.4, 35.9), "Marshall Islands": (171.2, 7.1),
    "Martinique": (-61.0, 14.65), "Mauritius": (57.55, -20.25),
    "Mayotte": (45.15, -12.8), "Micronesia": (158.2, 6.9),
    "Monaco": (7.42, 43.74), "Montserrat": (-62.19, 16.74),
    "Niue": (-169.87, -19.05), "Northern Mariana Islands": (145.7, 15.2),
    "Palau": (134.6, 7.5), "Reunion": (55.54, -21.11),
    "Saint Barthelemy": (-62.83, 17.9), "Saint Kitts and Nevis": (-62.7, 17.3),
    "Saint Lucia": (-60.97, 13.91), "Saint Martin": (-63.05, 18.08),
    "Saint Vincent and the Grenadines": (-61.2, 13.25), "Samoa": (-172.1, -13.76),
    "Sao Tome and Principe": (6.6, 0.34), "Seychelles": (55.5, -4.65),
    "Tonga": (-175.2, -21.2),
    "Singapore": (103.85, 1.29), "Sint Eustatius": (-62.97, 17.49),
    "Sint Maarten": (-63.06, 18.04), "Turks and Caicos Islands": (-71.8, 21.8),
    "U.S. Virgin Islands": (-64.9, 18.34), "Wallis and Futuna": (-178.1, -14.3),
}

gj = json.load(open("ne_110m_admin_0_countries.geojson", encoding="utf-8"))
gj_idx = {}
for feat in gj["features"]:
    p = feat["properties"]
    for k in ("NAME", "ADMIN", "NAME_LONG", "BRK_NAME"):
        v = p.get(k)
        if v:
            gj_idx.setdefault(v, feat)

obs_of = dict(zip(countries, n_obs))     # 国家 -> 熵观测月数

figm, axm = plt.subplots(figsize=(7.6, 3.9))
axm.axis("off")
cmap = plt.get_cmap("YlGnBu")
VMAX = 38
MISSING = "#E0E0E0"


def feat_rings(feat):
    g = feat["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for poly in polys:
        yield poly[0]                      # 外环(忽略内环, 飞地后画覆盖)


def feat_size(feat):
    m = 0.0
    for ring in feat_rings(feat):
        a = np.asarray(ring)
        m = max(m, (a[:, 0].max() - a[:, 0].min())
                * (a[:, 1].max() - a[:, 1].min()))
    return m


# 大面先画、小面后画, 使飞地(如莱索托)覆盖母国填充
drawn, n_poly = [], 0
for c in countries:
    feat = gj_idx.get(c) or gj_idx.get(GJ_ALIAS.get(c, ""))
    if feat is not None:
        drawn.append((feat_size(feat), obs_of[c], feat))
for _, v, feat in sorted(drawn, key=lambda t: -t[0]):
    col = cmap(v / VMAX)
    for ring in feat_rings(feat):
        a = np.asarray(ring)
        axm.add_patch(Polygon(a, closed=True, facecolor=col,
                              edgecolor="white", lw=0.3))
        n_poly += 1
# 无数据国家灰底
for feat in gj["features"]:
    nm = feat["properties"].get("ADMIN", "")
    if nm == "Antarctica":
        continue
    if all(feat is not f for _, _, f in drawn):
        for ring in feat_rings(feat):
            a = np.asarray(ring)
            axm.add_patch(Polygon(a, closed=True, facecolor=MISSING,
                                  edgecolor="white", lw=0.3))
n_pt = 0
for c, (lon, lat) in GJ_POINTS.items():
    if c in obs_of:
        axm.plot([lon], [lat], "o", ms=3.2, mew=0.5, mec="#555555",
                 mfc=cmap(obs_of[c] / VMAX), clip_on=False)
        n_pt += 1
axm.set_xlim(-180, 180)
axm.set_ylim(-60, 90)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, VMAX))
cb = figm.colorbar(sm, ax=axm, fraction=0.025, pad=0.01, aspect=28)
cb.set_label("熵观测月数", fontsize=7)
cb.set_ticks([0, 10, 20, 30, 38])
cb.ax.tick_params(labelsize=6)
cb.outline.set_visible(False)
axm.set_title("全球数据覆盖：各国熵观测月数（%d国，灰色=无数据，小岛屿/属地为圆点）"
              % NC, loc="left")
figm.tight_layout()
figm.savefig(os.path.join(OUT_SUP, "figS21_world_data_coverage_map.png"))
bbox_check(figm, "figS21")
plt.close(figm)
print("figS21: 多边形国家 %d, 圆点属地 %d" % (len(drawn), n_pt))

print("saved figS14..figS21 (组合图请运行 compose_fig8.py)")
