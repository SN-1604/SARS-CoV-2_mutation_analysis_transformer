# -*- coding: utf-8 -*-
"""compute_country_scenarios_v2.py — 代表国家情景计算 (v2 口径, 供 figS11/figS12)
六国: 美国/英国/中国/西班牙/墨西哥/土耳其
      (世界主要国家, 未校准 R²≥0.69, 众数组合降幅均≥4.4%)。
口径:
  - 基线与情景: 未校准集成 (forecast_v4_adapter, 与解释管线一致);
    组合扰动含分国速度降档 (applied_speed, 与 policy_combination_search_v2 一致)。
  - 剂量-响应: DELTAS 20 档全局倍数 + 约束钳制, 不降档
    (与 factor_sensitivity_v2 / figS5 完全一致; 分国=全样本预测取该国子集均值)。
  - 国家策略线采用该国验证期"众数组合" (policy_recommendations_v2.csv),
    与 csv 的 mean_reduction (逐样本最优口径) 不同, 面板以重算降幅标注。
自检:
  1) anchor R² 与 policy_combination_results_v2.json 一致 (容差 1e-4);
  2) 三因素 20 档全样本 mean_rel 与 factor_effects_monthly_v2.json 逐位一致
     (容差 1e-6) —— 保证本脚本扰动链路与主分析完全同构。
产出: country_scenarios_v2.npz
运行: G:/Anaconda3/envs/esm/python.exe compute_country_scenarios_v2.py
"""
import csv
import datetime as dt
import json, pickle

import numpy as np
import torch

from owid_repr13 import make_model
from forecast_v4_adapter import V4Forecaster
from train_entropy_forecast_monthly import MONTHS, MIDX
from factor_sensitivity_v2 import (DELTAS, GROUPS, build_counterfactual_panel,
                                   clamp_speed_increase)
from policy_combination_search_v2 import REACH_TAU, SPD

PANEL_PKL = "owid_weekly_panel.pkl"
REPR_CKPT = "repr13_best.pt"
OUT_NPZ = "country_scenarios_v2.npz"

COUNTRIES6 = ["United States", "United Kingdom", "China",
              "Spain", "Mexico", "Turkey"]
CN_NAME = {"United States": "美国", "United Kingdom": "英国", "China": "中国",
           "Spain": "西班牙", "Mexico": "墨西哥", "Turkey": "土耳其"}
DOSE_GROUPS = ["vacc_coverage", "vacc_speed", "stringency"]


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

    # ---- 分国速度档位可达性 (与 policy_combination_search_v2 相同) ----
    POP = raw[:, 0, cols.index("population")]
    jf = cols.index("new_people_vaccinated_smoothed_per_hundred")
    cum_obs = np.cumsum(raw[:, :, jf], axis=1)[:, -1]
    reach_grid = [g for g in SPD if g > 1.0]
    reachable = {}
    for g in reach_grid:
        sc = np.ones(38)
        sc[gidx["vacc_speed"]] = g
        Mg = clamp_speed_increase(raw * sc[None, None, :], raw, sc, cols, POP)
        dr = np.cumsum(Mg[:, :, jf], axis=1)[:, -1] / np.maximum(cum_obs, 1e-9)
        dr[cum_obs <= 1e-9] = 1.0
        reachable[g] = dr >= REACH_TAU * g

    def applied_speed(g):
        if g <= 1.0:
            return np.full(NC, g)
        out = np.ones(NC)
        for cand in reach_grid:
            if cand <= g:
                out = np.where(reachable[cand], cand, out)
        return out

    def embed_all(scale, spd_scale_by_country=None):
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

    # ---- 基线 + 自检 1 (anchor) ----
    y0 = predict(embed_all(np.ones(38)))
    r2 = 1 - ((y0 - y[val]) ** 2).sum() / ((y[val] - y[val].mean()) ** 2).sum()
    ref_anchor = json.load(open("policy_combination_results_v2.json",
                                encoding="utf-8"))["anchor_r2"]
    assert abs(r2 - ref_anchor) < 1e-4, f"anchor 不一致: {r2} vs {ref_anchor}"
    print(f"[自检1] anchor R2 = {r2:.4f} == {ref_anchor}  OK", flush=True)

    # ---- 自检 2: 三因素 20 档全样本 mean_rel 与 factor_effects json 一致 ----
    fe_ref = json.load(open("factor_effects_monthly_v2.json", encoding="utf-8"))
    dose_full = {g: {} for g in DOSE_GROUPS}       # 全样本 mean_rel (自检)
    dose_c6 = {g: np.zeros((len(DELTAS), 6)) for g in DOSE_GROUPS}  # 六国均值
    mask_c6 = np.array([countries[cs[i]] in COUNTRIES6 for i in range(len(y))])
    for g in DOSE_GROUPS:
        for di, d in enumerate(DELTAS):
            scale = np.ones(38)
            for col in GROUPS[g]:
                scale[cols.index(col)] = 1.0 + d
            yp = predict(embed_all(scale))
            rel = (yp - y0) / np.maximum(y0, 1e-6)
            dose_full[g][d] = float(rel.mean())
            refv = fe_ref["raw"][g][f"{d:.1f}"]["mean_rel"]
            assert abs(dose_full[g][d] - refv) < 1e-6, \
                f"{g} d={d}: {dose_full[g][d]} vs {refv}"
            for k, c in enumerate(COUNTRIES6):
                m = np.array([countries[cs[val[i]]] == c
                              for i in range(len(val))])
                dose_c6[g][di, k] = float(rel[m].mean())
        print(f"[自检2] {g} 20 档与 factor_effects_monthly_v2.json 一致  OK",
              flush=True)

    # ---- 各国众数组合情景 ----
    recs = {r["country"]: r for r in
            csv.DictReader(open("policy_recommendations_v2.csv",
                                encoding="utf-8"))}
    combos6 = {}
    for c in COUNTRIES6:
        r = recs[c]
        combos6[c] = (float(r["best_coverage"]), float(r["best_speed"]),
                      float(r["best_stringency"]))
    uniq = sorted(set(combos6.values()))
    P_combo = {}
    for cb in uniq:
        scale = np.ones(38)
        scale[gidx["vacc_coverage"]] = cb[0]
        scale[gidx["stringency"]] = cb[2]
        P_combo[cb] = predict(embed_all(scale, applied_speed(cb[1])))
        print(f"combo {cb} done", flush=True)

    # ---- 组装逐国数据 ----
    out = dict(deltas=np.array(DELTAS), countries6=np.array(COUNTRIES6),
               cn_names=np.array([CN_NAME[c] for c in COUNTRIES6]))
    for g in DOSE_GROUPS:
        out[f"dose_{g}"] = dose_c6[g]

    # ---- 三因素 实际 vs 众数组合反事实 周频轨迹 (figS13 用) ----
    LEV = {"coverage": "total_vaccinations_per_hundred",
           "speed": "new_people_vaccinated_smoothed_per_hundred",
           "stringency": "stringency_index"}
    out["weeks"] = np.array([(w0 + dt.timedelta(weeks=i)).isoformat()
                             for i in range(148)])
    M_cf_cache = {}
    for cb in uniq:
        scale = np.ones(38)
        scale[gidx["vacc_coverage"]] = cb[0]
        scale[gidx["stringency"]] = cb[2]
        M_cf_cache[cb] = build_counterfactual_panel(
            raw, scale, cols, pr_models, countries_repr, applied_speed(cb[1]))
    for k, c in enumerate(COUNTRIES6):
        ci = countries_repr.index(c)
        cb = combos6[c]
        for lk, col in LEV.items():
            j = cols.index(col)
            out[f"actual_{k}_{lk}"] = raw[ci, :, j]
            out[f"cf_{k}_{lk}"] = M_cf_cache[cb][ci, :, j]
        cf_str, cf_spd, cf_cov = (out[f"cf_{k}_stringency"],
                                  out[f"cf_{k}_speed"], out[f"cf_{k}_coverage"])
        assert -1e-9 <= cf_str.min() and cf_str.max() <= 100 + 1e-9
        assert cf_spd.min() >= -1e-9
        assert np.diff(cf_cov).min() >= -1e-6, f"{c} 反事实覆盖率非单调"
    print("lever trajectories exported", flush=True)

    meta = []
    for k, c in enumerate(COUNTRIES6):
        m = np.array([countries[cs[val[i]]] == c for i in range(len(val))])
        vv = val[m]
        order = np.argsort(ts[vv])
        vv = vv[order]
        cb = combos6[c]
        yb, ybest = y0[m][order], P_combo[cb][m][order]
        red = float(((yb - ybest) / np.maximum(yb, 1e-6)).mean())
        assert red >= -1e-9, f"{c} 众数组合不降熵? {red}"
        out[f"months_{k}"] = np.array(ts[vv])
        out[f"yobs_{k}"] = y[vv]
        out[f"ybase_{k}"] = yb
        out[f"ybest_{k}"] = ybest
        meta.append(dict(country=c, cn=CN_NAME[c], combo=list(cb),
                         speed_applied=float(applied_speed(cb[1])[
                             countries_repr.index(c)]),
                         reduction=red))
        print(f"{CN_NAME[c]}({c}): 组合{cb} 月均降幅 {red*100:.2f}%",
              flush=True)
    json.dump(meta, open("country_scenarios_v2_meta.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(OUT_NPZ, **out)
    print("DONE ->", OUT_NPZ, flush=True)


if __name__ == "__main__":
    main()
