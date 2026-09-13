# -*- coding: utf-8 -*-
"""factor_sensitivity_v2.py — 月熵预测模型的事后解释: 控制变量比例扰动敏感性
链路: 周面板扰动 -> log1p/z -> repr13 周窗表征 -> 按窗结束月平均 -> 跨配置集成预测
v2 适配: 熵预测模型为 entropy_forecast_monthly_v2_best.pt (跨配置成员集成,
全局归一化); 解释统一使用未校准集成预测 (forecast_v4_adapter, members 结构通用)。
扰动: δ ∈ ±10%..±100% (10% 梯度), 全时间轴持续性缩放 (目标月输入窗只含 ≤ 最后输入月
      的数据, 故与"截至最后输入月"等价)。
约束 (2024 修订):
  - stringency: 控制变化幅度使扰动后值 ∈[0,100]; 观测值 <15 的国家-周不允许下调
    (已处最低残余管控水平, 保持观测值, 仍可上调);
  - vacc_speed: 每日覆盖率增加值 ≤ 1/百人 (=1e4/百万=0.01×人口/日), 只限制速度的
    变化程度 (增量不超过上限余量, 观测已超上限的周增量为 0 保持观测历史, 不做数值截断);
    速度扰动时覆盖率按加性累计联动 (日净增值 ≡ 钳制后速度, apply_speed_to_coverage);
  - tests (检测量) 已移出分析; 静态背景列不扰动。
因素分组: cases 组为每日净增病例数 (new_cases* 流量列, 不含累计列)。
输出: factor_effects_monthly_v2.csv / .json + factor_effects_vaccine_era_v2.json
锚点: δ=0 复现验证 R²≈0.832 (未校准)。
"""
import datetime as dt
import json, pickle

import numpy as np
import torch

from owid_repr13 import make_model
from forecast_v4_adapter import V4Forecaster
from train_entropy_forecast_monthly import MONTHS, MIDX
from policy_response_models import counterfactual_panel

PANEL_PKL = "owid_weekly_panel.pkl"
REPR_CKPT = "repr13_best.pt"
TRAIN_END = "2022-07"

GROUPS = {
    "cases": ["new_cases", "new_cases_smoothed",
              "new_cases_per_million", "new_cases_smoothed_per_million"],  # 每日净增病例数 (流量列)
    "deaths": ["total_deaths", "new_deaths", "new_deaths_smoothed",
               "total_deaths_per_million", "new_deaths_per_million",
               "new_deaths_smoothed_per_million"],
    "reproduction_rate": ["reproduction_rate"],
    "vacc_coverage": ["total_vaccinations", "people_vaccinated",
                      "people_fully_vaccinated", "total_vaccinations_per_hundred",
                      "people_vaccinated_per_hundred",
                      "people_fully_vaccinated_per_hundred"],
    "vacc_speed": ["new_vaccinations_smoothed",
                   "new_vaccinations_smoothed_per_million",
                   "new_people_vaccinated_smoothed",
                   "new_people_vaccinated_smoothed_per_hundred"],
    "stringency": ["stringency_index"],
}
STATIC = ["population_density", "median_age", "aged_65_older", "aged_70_older",
          "gdp_per_capita", "cardiovasc_death_rate", "diabetes_prevalence",
          "life_expectancy", "human_development_index", "population",
          "total_tests", "total_tests_per_thousand",
          "new_tests_smoothed", "new_tests_smoothed_per_thousand"]  # tests 移出
DELTAS = [round(-1.0 + 0.1 * i, 1) for i in range(10)] + \
         [round(0.1 * i, 1) for i in range(1, 11)]
POLICY_FLAT = [c for g in ("vacc_coverage", "vacc_speed", "stringency")
               for c in GROUPS[g]]
FACTOR_NOTES = {"reproduction_rate":
                "传播率Rt为领先病例水平2-3周的估计值 (OWID reproduction_rate; "
                "Arroyo-Marioli et al. 2021, TrackingR/Kalman滤波)"}


def apply_speed_to_coverage(raw, M, cols, cov_lever=1.0):
    """覆盖率随接种速度变化 (加性联动, 锚定观测轨迹防积分漂移):
      覆盖率列 = cov_lever × (观测覆盖率 + 7×cumsum(速度_扰动 − 速度_观测))
    (周频面板: 速度列为日速率, 相邻周样本相隔 7 天, 累计需 ×7)
    覆盖率日净增值 = cov_lever × 扰动后速度 — 构造上与增量钳制后的速度列一致:
    只限制速度的变化程度 (不截断数值), 覆盖率杠杆不变 (cov_lever=1) 时
    每日覆盖率净增值 ≤ 1/百人; 观测本已超上限的周增量为 0 (保持观测历史)。
    clip ≥0 防降速情景出现负覆盖率。M 为已缩放并增量钳制后的面板 (就地改写)。"""
    def _apply(obs_col, spd_col, unit=1.0):
        oc = cols.index(obs_col)
        sc = cols.index(spd_col)
        # 周频面板: 相邻周样本相隔 7 天, 日速率的累计需 ×7 才是覆盖率变化
        delta = np.cumsum((M[:, :, sc] - raw[:, :, sc]) * unit * 7.0, axis=1)
        M[:, :, oc] = np.clip(cov_lever * (raw[:, :, oc] + delta), 0, None)
    # 覆盖率列随"新增接种人数/日"累计
    _apply("people_vaccinated", "new_people_vaccinated_smoothed")
    _apply("people_vaccinated_per_hundred", "new_people_vaccinated_smoothed_per_hundred")
    _apply("people_fully_vaccinated", "new_people_vaccinated_smoothed")
    _apply("people_fully_vaccinated_per_hundred",
           "new_people_vaccinated_smoothed_per_hundred")
    # 总剂次列随"剂次/日"累计 (per-million -> per-hundred 需 ×1e-4)
    _apply("total_vaccinations", "new_vaccinations_smoothed")
    _apply("total_vaccinations_per_hundred",
           "new_vaccinations_smoothed_per_million", 1e-4)
    return M


def clamp_speed_increase(M, raw, scale, cols, POP, spd_scale_by_country=None):
    """vacc_speed 上限: 每日覆盖率增加值 ≤ 1/百人 (=1e4/百万 = 0.01×人口/日)。
    只限制速度的变化程度, 不做数值截断: 提速时增量同时不超过
      (a) 速度列自身距上限的余量, (b) 联动覆盖率列距上限的余量
          (按覆盖率轨迹观测日净增值 = 周差分/7 计),
    使扰动后速度值与覆盖率日净增值均不超上限; 观测本已超上限的周增量为 0
    (保持观测历史); 降速时不施加上限。M 为已缩放面板 (就地改写)。
    spd_scale_by_country: 可选 (NC,) 分国速度倍数 (组合搜索的分国降档),
    给出时取代 scale 中速度列的全局倍数, 逐国判定是否限制增量。"""
    # (速度列, 上限, [联动覆盖率列], 覆盖率增量换算到速度列单位的倍率)
    pairs = [("new_people_vaccinated_smoothed_per_hundred", 1.0,
              ["people_vaccinated_per_hundred",
               "people_fully_vaccinated_per_hundred"], 1.0),
             ("new_people_vaccinated_smoothed", None,
              ["people_vaccinated", "people_fully_vaccinated"], 1.0),
             ("new_vaccinations_smoothed_per_million", 1e4,
              ["total_vaccinations_per_hundred"], 1e4),
             ("new_vaccinations_smoothed", None,
              ["total_vaccinations"], 1.0)]
    for col, cap, stocks, stk_unit in pairs:
        j = cols.index(col)
        if spd_scale_by_country is None:
            if scale[j] <= 1.0:
                continue                              # 下调不施加上限
            up = None                                 # 整列提速
        else:
            sv = np.asarray(spd_scale_by_country, dtype=np.float64)
            if (sv <= 1.0).all():
                continue
            up = (sv > 1.0)[:, None]                  # 逐国判定
        obs = raw[:, :, j]
        capv = (0.01 * POP)[:, None] if cap is None else cap
        room = np.maximum(capv - obs, 0.0)            # (a) 速度列余量
        for scol in stocks:                           # (b) 覆盖率轨迹余量
            stk = raw[:, :, cols.index(scol)]
            inc = np.concatenate([np.zeros((stk.shape[0], 1)),
                                  np.diff(stk, axis=1) / 7.0], axis=1)
            room = np.minimum(room, np.maximum(capv - inc * stk_unit, 0.0))
        incr = M[:, :, j] - obs
        limited = obs + np.minimum(incr, room)
        M[:, :, j] = limited if up is None else np.where(up, limited, M[:, :, j])
    return M


def build_counterfactual_panel(raw, scale, cols, pr_models, countries_repr,
                               spd_scale_by_country=None):
    """政策杠杆反事实面板 (factor_sensitivity 与 policy_combination_search 共用):
    比例缩放 -> stringency 增量钳制 (值域[0,100]; 观测值 <15 的国家-周不允许
    下调——已处最低残余管控水平, 降幅下界取 0, 仍可上调) -> 速度增量限制 (≤1%/日,
    可由 spd_scale_by_country 指定 (NC,) 分国速度倍数, 用于不可达高档降档)
    -> 速度→覆盖率联动 -> 辅助模型合成病例/死亡反事实列。
    scale: (38,) 各列缩放倍数; 全 1 且 spd_scale_by_country 为 None 时返回 raw 副本。"""
    M = raw * scale[None, None, :]
    si = cols.index("stringency_index")
    if scale[si] != 1.0:
        obs = raw[:, :, si]
        lo = np.where(obs < 15.0, 0.0, -obs)      # 观测<15 的周不允许下调
        chg = np.clip(M[:, :, si] - obs, lo, 100.0 - obs)
        M[:, :, si] = obs + chg
    spd_idx = [cols.index(c) for c in GROUPS["vacc_speed"]]
    if spd_scale_by_country is not None:
        sv = np.asarray(spd_scale_by_country, dtype=np.float64)
        M[:, :, spd_idx] = raw[:, :, spd_idx] * sv[:, None, None]
    POP = raw[:, 0, cols.index("population")]
    M = clamp_speed_increase(M, raw, scale, cols, POP, spd_scale_by_country)
    if spd_scale_by_country is None:
        spd_changed = any(scale[j] != 1.0 for j in spd_idx)
    else:
        spd_changed = bool((np.asarray(spd_scale_by_country) != 1.0).any())
    if spd_changed:
        cov_lever = scale[cols.index("people_vaccinated_per_hundred")]
        M = apply_speed_to_coverage(raw, M, cols, cov_lever)
    if spd_changed or any(scale[cols.index(c)] != 1.0 for c in POLICY_FLAT):
        M = counterfactual_panel(raw, M, cols, pr_models, countries_repr)
    return M


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
                    for c in countries_repr])           # (141,148,38)
    w0 = dt.date(2020, 3, 2)
    week_month = [month_of(w0 + dt.timedelta(weeks=i)) for i in range(148)]
    month_of_year = sorted(set(week_month))              # 35 个自然月
    midx_panel = {m: i for i, m in enumerate(month_of_year)}

    y, cs, ts, countries = fc.y, fc.cs, fc.ts, fc.countries
    val = fc.val_idx
    print(f"val samples: {len(val)}", flush=True)
    use_ci = np.array([countries_repr.index(countries[cs[i]]) for i in val])

    def embed_all(scale):
        """scale: (38,) 或 None; 返回 (len(val), K, 64) 各验证样本输入月表征"""
        if scale is None:
            M = raw.copy()
        else:
            M = build_counterfactual_panel(raw, scale, cols, pr_models,
                                           countries_repr)
        M[:, :, log_mask] = np.log1p(M[:, :, log_mask])
        Z = ((M - mu) / sd).astype(np.float32)
        NC = len(countries_repr)
        X = np.stack([Z[:, t - W + 1: t + 1, :]
                      for t in range(W - 1, 148)], axis=1)      # (NC,136,13,38)
        X = X.reshape(NC * 136, W, 38)
        E = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                E.append(repr_model(torch.from_numpy(X[i:i + 4096])
                                    .to(device))["emb"].cpu().numpy())
        E = np.concatenate(E).reshape(NC, 136, 64)              # (NC,136,64)
        # 周窗 -> 月平均
        Em = np.zeros((NC, len(month_of_year), 64), np.float32)
        Cm = np.zeros((NC, len(month_of_year), 1), np.float32)
        for t in range(W - 1, 148):
            mi = midx_panel[week_month[t]]
            Em[:, mi] += E[:, t - W + 1]
            Cm[:, mi] += 1
        Em = Em / np.maximum(Cm, 1)
        # 取每个验证样本的 K 个输入月
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
        """Em_samples: (N,K_max,64); 返回预测熵 (N,), v4 未校准集成"""
        return fc.predict_val(Em_samples)

    # ---- 锚点 ----
    y0 = predict(embed_all(None))
    r2 = 1 - ((y0 - y[val]) ** 2).sum() / ((y[val] - y[val].mean()) ** 2).sum()
    print(f"anchor R2 = {r2:.4f} (expect v2 uncal ensemble ~0.832)", flush=True)

    # ---- 扰动扫描 ----
    results = {g: {} for g in GROUPS}
    rel_p10 = {}                                   # +10% 的逐样本向量 (bootstrap 用)
    for g, gcols in GROUPS.items():
        for d in DELTAS:
            scale = np.ones(38)
            for col in gcols:
                scale[cols.index(col)] = 1.0 + d
            yp = predict(embed_all(scale))
            rel = (yp - y0) / np.maximum(y0, 1e-6)
            results[g][d] = dict(
                mean_rel=float(rel.mean()),
                mean_abs=float((yp - y0).mean()),
                sign_share=float((np.sign(yp - y0) == np.sign(d)).mean()))
            if d == 0.1:
                rel_p10[g] = rel
        print(f"{g} done", flush=True)

    # ---- headline 的 bootstrap 95% CI (对 477 个样本的均值重抽样, 固定种子) ----
    rng = np.random.default_rng(2024)
    ci95 = {}
    for g, rel in rel_p10.items():
        bs = rng.integers(0, len(rel), size=(2000, len(rel)))
        q = np.quantile(rel[bs].mean(axis=1), [0.025, 0.975])
        ci95[g] = (float(q[0]), float(q[1]))
        print(f"ci95 {g:<20} [{q[0]:+.5f}, {q[1]:+.5f}]", flush=True)

    rows = []
    for g in GROUPS:
        r = results[g]
        rows.append(dict(
            factor=g,
            headline_per_10pct=r[0.1]["mean_rel"],
            rel_chg_at_plus10=r[0.1]["mean_rel"],
            rel_chg_at_minus10=r[-0.1]["mean_rel"],
            rel_chg_at_plus100=r[1.0]["mean_rel"],
            rel_chg_at_minus100=r[-1.0]["mean_rel"],
            sign_share_plus10=r[0.1]["sign_share"],
            ci95_lo=ci95[g][0], ci95_hi=ci95[g][1],
            note=FACTOR_NOTES.get(g, ""),
            curve={str(d): r[d]["mean_rel"] for d in DELTAS}))
    rows.sort(key=lambda r: r["headline_per_10pct"])
    json.dump(dict(anchor_r2=float(r2), deltas=DELTAS, groups=GROUPS,
                   static_excluded=STATIC, ranking=rows, raw=results),
              open("factor_effects_monthly_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    import csv
    with open("factor_effects_monthly_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["factor", "headline_per_10pct",
                                          "rel_chg_at_plus10", "rel_chg_at_minus10",
                                          "rel_chg_at_plus100", "rel_chg_at_minus100",
                                          "sign_share_plus10",
                                          "ci95_lo", "ci95_hi", "note"])
        w.writeheader()
        w.writerows([{k: v for k, v in r.items() if k != "curve"} for r in rows])
    print("\n=== 因素排序 (每 +10% 扰动的预测熵平均相对变化; 负=遏制变异) ===")
    for r in rows:
        print(f"{r['factor']:<20} {r['headline_per_10pct']:+.4f}  "
              f"(+100%: {r['rel_chg_at_plus100']:+.3f}, "
              f"-100%: {r['rel_chg_at_minus100']:+.3f}, "
              f"符号一致率 {r['sign_share_plus10']:.2f})")

    # ---- 疫苗时代子样本 (最后输入月覆盖率 > 0) ----
    vp_col = cols.index("people_vaccinated_per_hundred")
    sub = np.zeros(len(val), bool)
    for n, si in enumerate(val):
        i = MIDX[ts[si]]
        m_last = MONTHS[i - 1]
        wks = [t for t in range(148) if week_month[t] == m_last]
        sub[n] = raw[use_ci[n], wks[-1], vp_col] > 0
    print(f"\n疫苗时代子样本: {sub.sum()}/{len(val)}")
    va_res = {}
    for g in ["vacc_coverage", "vacc_speed"]:
        for d in DELTAS:
            scale = np.ones(38)
            for col in GROUPS[g]:
                scale[cols.index(col)] = 1.0 + d
            yp = predict(embed_all(scale))
            rel = (yp[sub] - y0[sub]) / np.maximum(y0[sub], 1e-6)
            va_res.setdefault(g, {})[str(d)] = float(rel.mean())
    json.dump(dict(n_subset=int(sub.sum()), curves=va_res),
              open("factor_effects_vaccine_era_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("疫苗时代子样本曲线已存 factor_effects_vaccine_era_v2.json")


if __name__ == "__main__":
    main()
