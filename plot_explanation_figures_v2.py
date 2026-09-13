# -*- coding: utf-8 -*-
"""plot_explanation_figures_v2.py — 事后解释结果可视化 (figure-style 规范) -> figure_monthly_revised/
规范要点: CVD 安全调色板(禁红绿对立)、颜色跨图一致(threading)、句式要点标题、
字体三级阶梯(8/7/6)、lollipop 单观测类别、线端直接标注、发散色语义零点居中、
每面板注明 n 与固定条件、保存后 §9.1 重叠检查 + §9.2 目检。
数据: factor_effects_monthly_v2.json / factor_effects_vaccine_era_v2.json /
      policy_combination_results_v2.json / policy_recommendations_v2.csv
"""
import csv, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 8,                # 基准: 标题/轴标签/系列标识
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,          # 图例/标注
    "xtick.labelsize": 6,          # 刻度
    "ytick.labelsize": 6,
    "axes.linewidth": 0.6,
    "axes.labelpad": 4,
})

OUT = "figure_monthly_revised"
os.makedirs(OUT, exist_ok=True)

FE = json.load(open("factor_effects_monthly_v2.json", encoding="utf-8"))
VA = json.load(open("factor_effects_vaccine_era_v2.json", encoding="utf-8"))
PC = json.load(open("policy_combination_results_v2.json", encoding="utf-8"))
RECS = list(csv.DictReader(open("policy_recommendations_v2.csv", encoding="utf-8")))
AUX = np.load("policy_response_val_pred.npz")
AUX_NAMES = json.load(open("aux_model_names.json"))

CN = {"vacc_coverage": "疫苗覆盖率", "vacc_speed": "疫苗接种速度",
      "stringency": "管控严格度", "cases": "每日净增病例数", "deaths": "死亡数",
      "reproduction_rate": "传播率Rt*"}
RT_NOTE = "* 传播率Rt为领先病例水平 2–3 周的估计值\n（OWID/TrackingR, Kalman 滤波）"
COL = {"vacc_coverage": "#0072B2", "vacc_speed": "#56B4E9",
       "stringency": "#009E73", "cases": "#E69F00", "deaths": "#D55E00",
       "reproduction_rate": "#CC79A7"}
GOOD, BAD = "#0072B2", "#D55E00"          # 遏制=蓝, 加速=朱 (CVD 安全)
deltas = np.array(FE["deltas"], dtype=float)
NS = FE.get("raw") and 477
GROUPS_A = ["vacc_coverage", "vacc_speed", "stringency"]
GROUPS_B = ["cases", "deaths", "reproduction_rate"]


def curve(g):
    raw = FE["raw"][g]
    return np.array([raw[f"{d:.1f}"]["mean_rel"] for d in deltas]) * 100


def end_labels_staggered(ax, series, frac=0.08):
    """线端直接标注, 垂直方向防重叠 (series: [(yvals, text, color)]);
    按结束 y 排序后, 标签间距至少 frac×y 轴量程 (数据单位, 确定性)"""
    ax.figure.canvas.draw()
    y0, y1 = ax.get_ylim()
    step = (y1 - y0) * frac
    items = sorted([[float(s[0][-1]), s[1], s[2]] for s in series],
                   key=lambda x: x[0])
    last = -1e18
    for it in items:
        it[0] = max(it[0], last + step)
        last = it[0]
    # 上推后可能越出 y 轴上限 -> 扩展上限容纳最高标签
    if items and items[-1][0] > y1:
        ax.set_ylim(y0, items[-1][0] + step * 0.6)
    for yv, text, color in items:
        ax.annotate(text, (deltas[-1] * 100, yv),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=7, color=color, va="center")


def bbox_check(fig, fname):
    """§9.1: 非刻度文本互相不重叠、不压 spine、不出画布 (刻度间相邻属正常布局)"""
    import matplotlib as mpl
    r = fig.canvas.get_renderer()
    ticklbl = set()
    for ax in fig.axes:
        ticklbl.update(ax.get_xticklabels(which="both"))
        ticklbl.update(ax.get_yticklabels(which="both"))
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t not in ticklbl]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes
              for s in ax.spines.values() if s.get_visible()]
    bad = []
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if ba.overlaps(bb):
                bad.append((a.get_text()[:20], b.get_text()[:20]))
        for s, bs in spines:
            if ba.overlaps(bs):
                bad.append((a.get_text()[:20], "spine"))
    bad = [b for b in bad if b[0] != b[1]]
    if bad:
        print(f"  [bbox-warn] {fname}: {bad[:4]}")
    else:
        print(f"  [bbox-ok] {fname}")


def save(fig, fname):
    fig.savefig(f"{OUT}/{fname}", dpi=200)
    bbox_check(fig, fname)
    plt.close(fig)


# ---------------- fig0 辅助响应模型验证 ----------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
for ax, key, ttl in zip(axes, ["cases", "deaths"],
                        ["感染病例比例模型", "死亡比例模型"]):
    p, t = AUX[f"{key}_pred"], AUX[f"{key}_true"]
    r2 = 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    ax.plot(t, p, ".", ms=2, color="#0072B2", alpha=0.35, rasterized=True)
    lim = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax.plot(lim, lim, color="#D55E00", lw=1.0, ls="--")
    ax.set_xlabel("实际 log1p(每百万比例)")
    ax.set_ylabel("预测 log1p(每百万比例)")
    ax.set_title(f"{ttl}（{AUX_NAMES[key]}）  验证段 R²={r2:.2f}", loc="left")
    ax.grid(alpha=0.25)
    ax.margins(0.04)
fig.suptitle("辅助模型：政策→病例/死亡比例响应（验证段 2022-08 起, 逐国逐周）",
             x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.90])
save(fig, "fig0_aux_model_validation.png")

# ---------------- fig1 剂量-响应 ----------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
_series_store = []
for ax, groups, ttl in zip(axes, [GROUPS_A, GROUPS_B],
                           ["政策与防控因素", "疫情流行强度因素"]):
    series = []
    for g in groups:
        yv = curve(g)
        ax.plot(deltas * 100, yv, "o-", ms=2.5, lw=1.4, color=COL[g])
        series.append((yv, CN[g], COL[g]))
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.axvline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("因素比例扰动幅度 δ（%）")
    ax.set_title(ttl, loc="left")
    ax.grid(alpha=0.25)
    ax.margins(0.04)
    ax.set_xlim(-118, 155)
    _series_store.append((ax, series))
for ax, series in _series_store:      # 限值稳定后再排布线端标签
    end_labels_staggered(ax, series)
axes[0].set_ylabel("预测序列熵平均相对变化（%）")
axes[1].text(0.98, 0.03, RT_NOTE, transform=axes[1].transAxes,
             ha="right", va="bottom", fontsize=7, color="#333333")
fig.suptitle("疫苗覆盖率、接种速度与管控严格度遏制预测变异速度，"
             "死亡与传播率因素反向推高（n=477 验证样本）", x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.90])
save(fig, "fig1_factor_dose_response.png")

# ---------------- fig2 因素排序 lollipop ----------------
rows = FE["ranking"]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
names = [CN[r["factor"]] for r in rows]
vals = np.array([r["headline_per_10pct"] for r in rows]) * 100
cis = np.array([[r["ci95_lo"], r["ci95_hi"]] for r in rows]) * 100
ypos = np.arange(len(rows))
colors = [GOOD if v < 0 else BAD for v in vals]
ax.hlines(ypos, 0, vals, color="gray", lw=0.8)
for v, yp, c, (lo, hi) in zip(vals, ypos, colors, cis):
    ax.plot([lo, hi], [yp, yp], color=c, lw=1.4, alpha=0.55,
            solid_capstyle="round")                     # 95% bootstrap CI
    ax.plot(v, yp, "o", color=c, ms=5)
    txt = f"{v:+.3f}%" if 0 < abs(v) < 0.005 else f"{v:+.2f}%"
    ax.text(v, yp + 0.24, txt, va="center",
            ha="left" if v >= 0 else "right", fontsize=7)
ax.set_yticks(ypos)
ax.set_yticklabels(names, fontsize=8)
ax.set_ylim(-0.55, len(rows) - 1 + 0.75)     # 顶部为数值标签留位
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("每 +10% 扰动的预测序列熵平均相对变化（%）")
ax.set_title("疫苗接种速度是最强的变异遏制因素（每 +10% 扰动, n=477）", loc="left")
ax.text(0.99, 0.03, RT_NOTE + "\n圆点横线 = 均值 95% bootstrap CI（2000 次重抽样）",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        color="#333333")
ax.margins(0.04)
ax.set_xlim(cis.min() - 0.03, cis.max() + 0.03)
ax.grid(alpha=0.25, axis="x")
fig.tight_layout()
save(fig, "fig2_factor_headline_lollipop.png")

# ---------------- fig3 疫苗时代剂量-响应 ----------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
series3 = []
for g in ["vacc_coverage", "vacc_speed"]:
    c = VA["curves"][g]
    yv = np.array([c[f"{d:.1f}"] for d in deltas]) * 100
    ax.plot(deltas * 100, yv, "o-", ms=2.5, lw=1.4, color=COL[g])
    series3.append((yv, CN[g], COL[g]))
end_labels_staggered(ax, series3)
ax.axhline(0, color="gray", lw=0.6, ls="--")
ax.axvline(0, color="gray", lw=0.6, ls="--")
ax.set_xlabel("扰动幅度 δ（%）")
ax.set_ylabel("预测序列熵平均相对变化（%）")
ax.set_title(f"疫苗时代样本 (n={VA['n_subset']}): 剂量-响应形态与全时段一致", loc="left")
ax.grid(alpha=0.25)
ax.margins(0.04)
ax.set_xlim(-118, 150)
fig.tight_layout()
save(fig, "fig3_vaccine_era_dose_response.png")

# ---------------- fig4/fig5 响应面与协同面 ----------------
surf = PC["surface_stringency_x_coverage"]
cov, strg = surf["coverage"], surf["stringency"]
for data0, ttl, fname, clab in [
        (np.array(surf["reduction"]) * 100, "响应面: 预测熵降幅由覆盖率主导，管控收紧小幅增效",
         "fig4_response_surface.png", "预测熵降幅（%）"),
        (np.array(surf["synergy"]) * 100, "协同面: 覆盖率×管控协同度整体微弱",
         "fig5_synergy_surface.png", "协同度（%, 正=协同增效）")]:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    vmax = np.abs(data0).max()
    im = ax.imshow(data0, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(strg)))
    ax.set_xticklabels([f"{s:g}" for s in strg])
    ax.set_yticks(range(len(cov)))
    ax.set_yticklabels([f"{c:g}" for c in cov])
    ax.set_xlabel("管控严格度缩放倍数")
    ax.set_ylabel("疫苗覆盖率缩放倍数")
    for i in range(len(cov)):
        for j in range(len(strg)):
            ax.text(j, i, f"{data0[i, j]:+.2f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if abs(data0[i, j]) > 0.6 * vmax else "black")
    ax.set_title(f"{ttl}（固定接种速度=1.0, n=477）", loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(clab, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.text(0.99, 1.02, "降幅越大越好 →", transform=ax.transAxes,
            ha="right", fontsize=7, color="#333333")
    fig.tight_layout()
    save(fig, fname)

# ---------------- fig6 Top 组合 + 变动因素数分类 ----------------
ca = PC["factor_unchanged_analysis"]
# 左图: 所有组合按全局获胜占比严格降序 Top10 (类内最优组合见右图)
allc = [(tuple(c["combo"]), c["share"]) for c in PC["all_combo_shares"]]
shown = allc[:10]
CATCOL = {1: "#56B4E9", 2: "#0072B2", 3: "#003f7f"}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6),
                         gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
labels = [f"覆盖×{c[0]:g}\n速度×{c[1]:g}\n管控×{c[2]:g}" for c, _ in shown]
shares = [s * 100 for _, s in shown]
bcolors = [CATCOL[sum(1 for v in c if v != 1.0)] for c, _ in shown]
ax.bar(range(len(shown)), shares, color=bcolors, alpha=0.95, width=0.7)
for i, s in enumerate(shares):
    ax.text(i, max(s + 0.3, 0.8), f"{s:.0f}%" if s >= 10 else f"{s:.1f}%",
            ha="center", fontsize=7)
ax.set_ylim(0, max(shares) * 1.15)
ax.set_xticks(range(len(shown)))
ax.set_xticklabels(labels, fontsize=6)
ax.set_ylabel("作为样本最优组合的占比（%）")
ax.set_title("最优组合按全局获胜占比降序（Top10; n=477 验证样本）", loc="left")
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=CATCOL[k], label=f"{k} 因素变动")
                   for k in (1, 2, 3)], fontsize=7, frameon=False, loc="upper right")
ax.grid(alpha=0.25, axis="y")
ax.margins(0.04)

ax = axes[1]
cats = ["1_factor_changed", "2_factor_changed", "3_factor_changed"]
cat_names = ["仅单因素变动", "双因素变动", "三因素全变动"]
reds = [ca[k]["mean_reduction"] * 100 for k in cats]
best_labels = [f"覆盖×{ca[k]['best_combo'][0]:g} 速度×{ca[k]['best_combo'][1]:g} "
               f"管控×{ca[k]['best_combo'][2]:g}" for k in cats]
bars = ax.bar(range(3), reds, color=["#56B4E9", "#0072B2", "#003f7f"], width=0.6)
for i, v in enumerate(reds):
    ax.text(i, v + 0.06, f"{v:.1f}%", ha="center", fontsize=7)
ax.set_ylim(0, max(reds) * 1.2)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{n}\n（最优: {bl}）" for n, bl in zip(cat_names, best_labels)],
                   fontsize=6)
ax.set_ylabel("最优组合的平均预期熵降幅（%）")
ax.set_title("按变动因素数分类的最优组合效果", loc="left")
ax.grid(alpha=0.25, axis="y")
ax.margins(0.04)
fig.tight_layout()
save(fig, "fig6_top_combos.png")

# ---------------- fig7 分国降幅 ----------------
reds = np.array([float(r["mean_reduction"]) for r in RECS]) * 100
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6),
                         gridspec_kw={"width_ratios": [1, 1.25]})
axes[0].hist(reds, bins=25, color=GOOD, alpha=0.85, edgecolor="white", lw=0.4)
axes[0].axvline(reds.mean(), color=BAD, ls="--", lw=1.0)
axes[0].text(reds.mean() + 0.3, axes[0].get_ylim()[1] * 0.92,
             f"均值 {reds.mean():.1f}%", fontsize=7, color=BAD)
axes[0].set_xlabel("最优组合下的预期熵降幅（%）")
axes[0].set_ylabel("国家数")
axes[0].set_title(f"分国预期熵降幅分布（n={len(RECS)} 国）", loc="left")
top = sorted(RECS, key=lambda r: float(r["mean_reduction"]))[-15:]
axes[1].barh([r["country"] for r in top],
             [float(r["mean_reduction"]) * 100 for r in top], color="#56B4E9")
axes[1].set_xlabel("预期熵降幅（%）")
axes[1].set_title("预期降幅 Top15 国家", loc="left")
axes[1].tick_params(axis="y", labelsize=6)
for ax in axes:
    ax.grid(alpha=0.25)
    ax.margins(0.04)
fig.tight_layout()
save(fig, "fig7_country_reduction.png")

print("figures saved ->", OUT)
