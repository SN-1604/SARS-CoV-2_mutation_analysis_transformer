# -*- coding: utf-8 -*-
"""policy_combination_search.py — 疫苗接种 × 政策 组合寻优 (反事实网格搜索 + 交互分析)
在验证集 477 个 (国家, 目标月) 样本上, 对三杆组合网格批量前向:
  vacc_coverage × {0.5,0.75,1.0,1.5,2.0,3.0}   (不截断)
  vacc_speed    × {0.5,1,1.5,2,3,5}            (只限制变化程度: 每日覆盖率增加值 ≤1/百人;
                                                分国降档: 名义档不可达的国家降至其最高可达档
                                                —— 可达 = 钳制后累计剂次比 ≥0.9×名义;
                                                ×5 仅 36/141 国可达; 速度经加性累计联动覆盖率)
  stringency    × {0.5,0.75,1.0,1.25,1.5,2.0}  (增量钳制: 值域[0,100];
                                                观测值<15 的国家-周不允许下调)
反事实构造与 factor_sensitivity.build_counterfactual_panel 完全一致。
目标: 预测熵最小 (遏制变异速度 -> 趋同进化)。
输出: policy_combination_results_v2.json (含 speed_reachability), policy_recommendations_v2.csv
v2 适配: 熵预测模型为 entropy_forecast_monthly_v2_best.pt (未校准集成,
forecast_v4_adapter); 反事实构造与 factor_sensitivity_v2 完全一致。
"""
import datetime as dt
import json, pickle
from collections import Counter

import numpy as np
import torch

from owid_repr13 import make_model
from forecast_v4_adapter import V4Forecaster
from train_entropy_forecast_monthly import MONTHS, MIDX
from factor_sensitivity_v2 import (GROUPS, build_counterfactual_panel,
                                   clamp_speed_increase)

PANEL_PKL = "owid_weekly_panel.pkl"
REPR_CKPT = "repr13_best.pt"
TRAIN_END = "2022-07"

COV = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
SPD = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
STR = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
REACH_TAU = 0.9          # 可达性容差: 钳制后累计剂次比 ≥ 0.9×名义档 判可达


def month_of(d):
    return f"{d.year:04d}-{d.month:02d}"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPR_CKPT, map_location="cpu", weights_only=False)
    cols = [str(c) for c in ck["cols"]]
    mu = ck["mu"].astype(np.float64)
    sd = ck["sd"].astype(np.float64)
    log_mask = ck["log_mask"].copy()
    W = int(ck["W"])
    countries_repr = sorted(str(c) for c in ck["countries"])
    cfg = {k: v for k, v in ck["cfg"].items() if k != "epochs"}
    repr_model = make_model(len(countries_repr), **cfg).to(device)
    repr_model.load_state_dict(ck["state_dict"])
    repr_model.eval()

    fc = V4Forecaster(ckpt="entropy_forecast_monthly_v2_best.pt", device=device)
    K = fc.K_max
    pr_models = pickle.load(open("policy_response_models.pkl", "rb"))["models"]

    panel = pickle.load(open(PANEL_PKL, "rb"))
    raw = np.stack([np.asarray(panel[c][cols], dtype=np.float64)
                    for c in countries_repr])
    w0 = dt.date(2020, 3, 2)
    week_month = [month_of(w0 + dt.timedelta(weeks=i)) for i in range(148)]
    month_of_year = sorted(set(week_month))
    midx_panel = {m: i for i, m in enumerate(month_of_year)}
    gidx = {g: [cols.index(c) for c in gcols] for g, gcols in GROUPS.items()}

    y, cs, ts, countries = fc.y, fc.cs, fc.ts, fc.countries
    val = fc.val_idx
    use_ci = np.array([countries_repr.index(countries[cs[i]]) for i in val])
    NC = len(countries_repr)

    # ---- 分国速度档位可达性 (双重余量钳制下, 累计剂次比 ≥ 0.9×名义档 判可达) ----
    POP = raw[:, 0, cols.index("population")]
    jf = cols.index("new_people_vaccinated_smoothed_per_hundred")
    cum_obs = np.cumsum(raw[:, :, jf], axis=1)[:, -1]
    reach_grid = [g for g in SPD if g > 1.0]
    reachable = {}
    for g in reach_grid:
        sc = np.ones(len(cols))
        sc[gidx["vacc_speed"]] = g
        Mg = clamp_speed_increase(raw * sc[None, None, :], raw, sc, cols, POP)
        dr = np.cumsum(Mg[:, :, jf], axis=1)[:, -1] / np.maximum(cum_obs, 1e-9)
        dr[cum_obs <= 1e-9] = 1.0                     # 无接种记录国家: 不可提速
        reachable[g] = dr >= REACH_TAU * g
        print(f"[可达性] x{g}: {int(reachable[g].sum())}/{NC} 国可达", flush=True)

    def applied_speed(g):
        """分国降档: 返回 (NC,) 各国在名义档 g 下实际执行的速度倍数
        (不超过 g 的最高可达档; 均不可达则 1.0; 减速/不变档原样)。"""
        if g <= 1.0:
            return np.full(NC, g)
        out = np.ones(NC)
        for cand in reach_grid:
            if cand <= g:
                out = np.where(reachable[cand], cand, out)
        return out

    top_reach = np.ones(NC)                           # 各国最高可达档 (存档用)
    for g in sorted(reach_grid, reverse=True):
        top_reach = np.where((top_reach == 1.0) & reachable[g], g, top_reach)

    def embed_all(scale, spd_scale_by_country=None):
        # 与 factor_sensitivity 共用同一反事实构造 (增量限制上限 + 速度→覆盖率联动)
        M = build_counterfactual_panel(raw, scale, cols, pr_models,
                                       countries_repr, spd_scale_by_country)
        M[:, :, log_mask] = np.log1p(M[:, :, log_mask])
        Z = ((M - mu) / sd).astype(np.float32)
        X = np.stack([Z[:, t - W + 1: t + 1, :]
                      for t in range(W - 1, 148)], axis=1).reshape(NC * 136, W, 38)
        E = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                E.append(repr_model(torch.from_numpy(X[i:i + 8192])
                                    .to(device))["emb"].cpu().numpy())
        E = np.concatenate(E).reshape(NC, 136, 64)
        Em = np.zeros((NC, len(month_of_year), 64), np.float32)
        Cm = np.zeros((NC, len(month_of_year), 1), np.float32)
        for t in range(W - 1, 148):
            mi = midx_panel[week_month[t]]
            Em[:, mi] += E[:, t - W + 1]
            Cm[:, mi] += 1
        Em /= np.maximum(Cm, 1)
        out = np.zeros((len(val), K, 64), np.float32)
        for n, si in enumerate(val):
            i = MIDX[ts[si]]
            jj = 0
            for j in range(max(0, i - K), i):
                m = MONTHS[j]
                if m in midx_panel:
                    out[n, jj] = Em[use_ci[n], midx_panel[m]]
                jj += 1
        return out

    def predict(Em_samples):
        return fc.predict_val(Em_samples)

    # ---- 基线 ----
    y0 = predict(embed_all(np.ones(38)))
    r2 = 1 - ((y0 - y[val]) ** 2).sum() / ((y[val] - y[val].mean()) ** 2).sum()
    print(f"anchor R2 = {r2:.4f}", flush=True)

    # ---- 组合网格搜索 ----
    combos = [(c, s, q) for c in COV for s in SPD for q in STR]
    P = np.zeros((len(combos), len(val)), np.float64)
    for ci, (cv, sp, st) in enumerate(combos):
        scale = np.ones(38)
        scale[gidx["vacc_coverage"]] = cv
        scale[gidx["stringency"]] = st
        P[ci] = predict(embed_all(scale, applied_speed(sp)))
        if (ci + 1) % 30 == 0:
            print(f"combo {ci+1}/{len(combos)}", flush=True)

    def pick_best(Psub, combos_sub):
        """逐样本 argmin; 精确并列 (约束使降档/禁下调成为空操作, 预测逐位相等)
        时优先总变化幅度最小 (sum |log 倍数|) 的组合——不把无变化误标为干预。"""
        pmin = Psub.min(0)
        pref = np.array([abs(np.log(cb[0])) + abs(np.log(cb[1]))
                         + abs(np.log(cb[2])) for cb in combos_sub])
        best_idx = np.empty(Psub.shape[1], dtype=int)
        for n in range(Psub.shape[1]):
            tied = np.nonzero(Psub[:, n] <= pmin[n] + 1e-12)[0]
            best_idx[n] = tied[np.argmin(pref[tied])]
        return best_idx

    best = pick_best(P, combos)
    best_pred = P[best, np.arange(len(val))]
    base_pred = P[combos.index((1.0, 1.0, 1.0))]
    reduction = (base_pred - best_pred) / np.maximum(base_pred, 1e-6)

    # ---- 检验与显式分析: 含"因素不变"(=1.0)档位的组合 ----
    n_chg = np.array([sum(1 for v in cb if v != 1.0) for cb in combos])
    print(f"[检验] 网格共 {len(combos)} 组合; 含不变因素的组合: "
          f"{int((n_chg <= 2).sum())} 个 (单因素变动 {int((n_chg == 1).sum())}, "
          f"双因素 {int((n_chg == 2).sum())}, 三因素 {int((n_chg == 3).sum())}, "
          f"全不变 {int((n_chg == 0).sum())})", flush=True)
    cat_analysis = {}
    for k in (1, 2, 3):
        sel = np.nonzero(n_chg == k)[0]
        Pk = P[sel]
        bk = pick_best(Pk, [combos[i] for i in sel])
        redk = (base_pred - Pk[bk, np.arange(len(val))]) / np.maximum(base_pred, 1e-6)
        cnt = Counter(tuple(combos[sel[i]]) for i in bk)
        bc, bcnt = cnt.most_common(1)[0]
        cat_analysis[f"{k}_factor_changed"] = dict(
            best_combo=list(bc), share=round(bcnt / len(val), 4),
            mean_reduction=float(redk.mean()),
            median_reduction=float(np.median(redk)))
        print(f"[{k}因素变动] 最优组合 {bc} 占比 {bcnt/len(val):.1%} "
              f"平均熵降幅 {redk.mean()*100:.2f}%", flush=True)

    # ---- 交互分析: stringency × coverage (speed=1) ----
    surf = np.zeros((len(COV), len(STR)))
    for ic, cv in enumerate(COV):
        for iq, st in enumerate(STR):
            surf[ic, iq] = P[combos.index((cv, 1.0, st))].mean()
    eff = (base_pred.mean() - surf) / base_pred.mean()      # 熵降幅 (正=好)
    syn = np.zeros_like(surf)
    for ic in range(len(COV)):
        for iq in range(len(STR)):
            syn[ic, iq] = eff[ic, iq] - (eff[ic, STR.index(1.0)]
                                         + eff[COV.index(1.0), iq])
    # ---- 逐样本最优组合统计 ----
    best_combo_count = Counter(tuple(combos[b]) for b in best)
    # 逐国建议: 该国各样本最优组合的众数 + 平均预期降幅
    recs = []
    for ci, c in enumerate(countries):
        m = np.nonzero(cs[val] == ci)[0]
        if len(m) == 0:
            continue
        bc = Counter(tuple(combos[b]) for b in best[m]).most_common(1)[0][0]
        recs.append(dict(country=c, n_samples=int(len(m)),
                         best_coverage=bc[0], best_speed=bc[1], best_stringency=bc[2],
                         best_speed_applied=float(applied_speed(bc[1])[
                             countries_repr.index(c)]),
                         mean_reduction=float(reduction[m].mean()),
                         baseline_pred=float(base_pred[m].mean()),
                         best_pred=float(best_pred[m].mean())))
    recs.sort(key=lambda r: -r["mean_reduction"])

    out = dict(
        anchor_r2=float(r2), n_samples=int(len(val)),
        grid=dict(coverage=COV, speed=SPD, stringency=STR),
        speed_reachability=dict(
            criterion="clamped cumulative dose ratio >= %.2f x nominal "
                      "(new_people_vaccinated_smoothed_per_hundred)" % REACH_TAU,
            n_reachable={str(g): int(reachable[g].sum()) for g in reach_grid},
            highest_reachable={c: float(t)
                               for c, t in zip(countries_repr, top_reach)}),
        tie_break="exact prediction ties (no-op levers after constraints) "
                  "resolved toward smallest total change (sum |log scale|)",
        global_best_combo=[list(x) for x in
                           [best_combo_count.most_common(1)[0][0]]][0],
        best_combo_share=best_combo_count.most_common(1)[0][1] / len(val),
        top_combos=[dict(combo=list(k), share=round(v / len(val), 4))
                    for k, v in best_combo_count.most_common(10)],
        all_combo_shares=[dict(combo=list(k), share=round(v / len(val), 6))
                          for k, v in best_combo_count.most_common()],
        mean_reduction_best=float(reduction.mean()),
        median_reduction_best=float(np.median(reduction)),
        factor_unchanged_analysis=cat_analysis,
        surface_stringency_x_coverage=dict(
            coverage=COV, stringency=STR,
            reduction=eff.round(5).tolist(), synergy=syn.round(5).tolist()),
    )
    json.dump(out, open("policy_combination_results_v2.json", "w",
              encoding="utf-8"), ensure_ascii=False, indent=1)
    import csv
    with open("policy_recommendations_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(json.dumps({k: v for k, v in out.items()
                      if k != "surface_stringency_x_coverage"},
                     ensure_ascii=False, indent=1))
    print("\n=== stringency×coverage 熵降幅响应面 (speed=1) ===")
    hdr = "cov\\str  " + "".join(f"{q:>8.2f}" for q in STR)
    print(hdr)
    for ic, cv in enumerate(COV):
        print(f"{cv:<8.2f}" + "".join(f"{eff[ic, iq]*100:>+8.3f}" for iq in range(len(STR))))
    print("\n=== 协同度 (正=协同增效) ===")
    for ic, cv in enumerate(COV):
        print(f"{cv:<8.2f}" + "".join(f"{syn[ic, iq]*100:>+8.3f}" for iq in range(len(STR))))


if __name__ == "__main__":
    main()
