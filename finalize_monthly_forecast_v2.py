# -*- coding: utf-8 -*-
"""finalize_monthly_forecast_v2.py — 定稿月熵预测模型 (不覆盖 v1/v2/v3 产物)
读取 best_config 描述 (单配置或跨配置), 缺种子运行自动补训, 打包成员集成权重。
后处理: 逐国仿射静态校准 (训练段拟合 y≈a+b·pred, 向恒等映射岭收缩,
descriptor 的 cal_lambda 控制强度; 另支持严格因果的滚动在线校准 roll_lam/roll_win);
评估口径: 全量 106 国 + 剔除月均序列<10 国后的过滤口径 (--counts_file 提供计数,
descriptor 的 exclude_mean_seq_below 控制阈值, 默认 10)。
产物 ({prefix}_*):
  {prefix}_best.pt / {prefix}_entonly_best.pt / {prefix}_metrics.json /
  {prefix}_per_country.csv
验收: 过滤口径逐国 R2>=0.8 国家数 >= ceil(0.5*n_eval); 表征贡献 >=0.30;
      pooled R2 >=0.80 (后两者均为过滤口径未校准集成)
运行: G:/Anaconda3/envs/esm/python.exe finalize_monthly_forecast_v2.py
        (默认 --best_cfg forecast_v2_best_config.json --runs monthly_forecast_runs_v2
         --prefix entropy_forecast_monthly_v2)
"""
import argparse, json, os

import numpy as np
import torch

import train_entropy_forecast_v2 as T
import search_forecast_v2 as S

BEST_CFG = "forecast_v2_best_config.json"
V1 = dict(pooled_r2=0.8223, per_country_median=0.4012, n_ge_08=19,
          n_eval=106, emb_contribution=0.31)
V2 = dict(pooled_r2=0.832, per_country_median=0.4479, n_ge_08=25,
          n_eval=106, emb_contribution=0.337)
ENS_SEEDS = [7, 42, 123]


def build_members(desc):
    if desc.get("type") == "cross":
        members = [(c, s, ENS_SEEDS)
                   for c, s in zip(desc["cfgs"], desc["stages"])]
    else:
        members = [(desc["cfg"], desc["stage"], desc["seeds"])]
    for cfg, stage, seeds in members:
        for s in seeds:
            for mode in ("full", "ent_only"):
                S.run_once(stage, cfg, mode, s)
    return members


def ensemble_predict(members, mode):
    ps, D = [], None
    for cfg, stage, seeds in members:
        for s in seeds:
            p, D = S.predict_run(S.cfg_name(stage, cfg, mode, s))
            ps.append(p)
    return np.mean(np.stack(ps), axis=0), D


def member_specs(members, mode):
    specs = []
    for cfg, stage, seeds in members:
        states, norm, F, ent_dir = [], None, None, None
        for s in seeds:
            ck = torch.load(os.path.join(
                T.RUNS, f"{S.cfg_name(stage, cfg, mode, s)}.pt"),
                map_location="cpu", weights_only=False)
            states.append(ck["state_dict"])
            norm, F = ck["norm"], ck["F"]
            ent_dir = ck.get("ent_dir", "entropy_monthly_smoothed")
        specs.append(dict(model=cfg["model"], K=cfg["K"],
                          hidden=cfg.get("hidden", 128),
                          layers=cfg.get("layers", 1),
                          dropout=cfg.get("dropout", 0.1),
                          delta=bool(cfg.get("delta")),
                          seq_weight=cfg.get("seq_weight", "none"),
                          target_norm=cfg["target_norm"], F=F, norm=norm,
                          ent_dir=ent_dir,
                          ensemble_state_dicts=states, seeds=list(seeds)))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_cfg", default=BEST_CFG)
    ap.add_argument("--runs", default=T.RUNS)
    ap.add_argument("--prefix", default="entropy_forecast_monthly_v2")
    ap.add_argument("--ent_dir", default="entropy_monthly_smoothed")
    ap.add_argument("--log", default=None)
    ap.add_argument("--counts_file", default=None)
    a = ap.parse_args()
    T.RUNS = a.runs
    S.ENT_DIR = a.ent_dir
    if a.log:
        S.LOG = a.log
    if a.counts_file:
        S.COUNTS_FILE = a.counts_file
        S.COUNTS = T.load_counts(a.counts_file)
    desc = json.load(open(a.best_cfg, encoding="utf-8"))
    lam = desc.get("cal_lambda", 10.0)
    roll_lam = desc.get("roll_lam", S.ROLL["lam"])
    roll_win = desc.get("roll_win", S.ROLL["win"])
    thr = desc.get("exclude_mean_seq_below", 10.0)
    counts = S.COUNTS
    print("finalize:", json.dumps(desc, ensure_ascii=False)[:300])

    members = build_members(desc)
    pf, D = ensemble_predict(members, "full")
    pe, _ = ensemble_predict(members, "ent_only")
    y, cs, ts, countries = D["y"], D["cs"], D["ts"], D["countries"]
    tr, va = D["tr"], D["va"]

    # 口径1: 未校准集成 (贡献核算协议)
    m_full = T.summarize(pf[va], y[va], cs[va])
    m_ent = T.summarize(pe[va], y[va], cs[va])
    contrib = (m_ent["mse"] - m_full["mse"]) / m_ent["mse"]

    # 口径2: 滚动在线校准 (严格因果)
    pf_r = T.rolling_calibrate(pf, y, cs, ts, countries,
                               lam=roll_lam, win=roll_win)
    pe_r = T.rolling_calibrate(pe, y, cs, ts, countries,
                               lam=roll_lam, win=roll_win)
    m_full_r = T.summarize(pf_r[va], y[va], cs[va])
    m_ent_r = T.summarize(pe_r[va], y[va], cs[va])
    contrib_r = (m_ent_r["mse"] - m_full_r["mse"]) / m_ent_r["mse"]
    # 参考: 旧静态校准
    cal = T.fit_calibration(pf[tr], y[tr], cs[tr], countries, lam)
    pf_c = T.apply_calibration(pf[va], cs[va], countries, cal)
    m_full_c = T.summarize(pf_c, y[va], cs[va])

    # 过滤口径 (剔除月均序列 < thr 的国家)
    f80 = None
    if counts is not None:
        keep = T.filtered_country_set(D, counts, thr=thr)
        f_full = T.summarize_on(pf[va], y[va], cs[va], keep)
        f_ent = T.summarize_on(pe[va], y[va], cs[va], keep)
        f_full_r = T.summarize_on(pf_r[va], y[va], cs[va], keep)
        f_ent_r = T.summarize_on(pe_r[va], y[va], cs[va], keep)
        contrib_f = (f_ent["mse"] - f_full["mse"]) / f_ent["mse"]
        contrib_fr = (f_ent_r["mse"] - f_full_r["mse"]) / f_ent_r["mse"]
        f80 = dict(n_keep=len(keep), full=f_full, ent=f_ent,
                   full_roll=f_full_r, ent_roll=f_ent_r,
                   contrib=round(float(contrib_f), 4),
                   contrib_roll=round(float(contrib_fr), 4))

    # 验收: 主口径 = 过滤口径(若有 counts), 否则全量
    if f80:
        pm, pem, pcontrib = f80["full_roll"], f80["full"], f80["contrib"]
    else:
        pm, pem, pcontrib = m_full_r, m_full, contrib
    tgt = int(np.ceil(0.5 * pm["n_countries_eval"]))
    accept = dict(
        scope=("filtered_mean_seq>=%g" % thr) if f80 else "all",
        per_country_ge08=f"{pm['n_countries_r2_ge_08']}/"
                         f"{pm['n_countries_eval']}",
        per_country_target=f">={tgt}",
        per_country_pass=bool(pm["n_countries_r2_ge_08"] >= tgt),
        emb_contribution=round(float(pcontrib), 4),
        emb_contribution_pass=bool(pcontrib >= 0.30),
        pooled_r2=pem["r2"], pooled_r2_pass=bool(pem["r2"] >= 0.80))

    torch.save(dict(members=member_specs(members, "full"), mode="full",
                    rolling=dict(lam=roll_lam, win=roll_win),
                    exclude_mean_seq_below=thr, calibration=None,
                    cal_lambda=lam, countries=countries, descriptor=desc,
                    ent_dir=a.ent_dir,
                    note="full 集成: 成员x种子均值(原始尺度)+滚动在线逐国仿射"
                         "校准(严格因果); 无国家身份输入特征"),
               f"{a.prefix}_best.pt")
    torch.save(dict(members=member_specs(members, "ent_only"),
                    mode="ent_only", rolling=dict(lam=roll_lam, win=roll_win),
                    calibration=None, countries=countries, descriptor=desc,
                    ent_dir=a.ent_dir, note="ent_only 消融集成"),
               f"{a.prefix}_entonly_best.pt")

    mean_seq = {}
    if counts is not None:
        for ci, c in enumerate(countries):
            ns = [counts.get(c, {}).get(m, 0)
                  for m in ts[va][cs[va] == ci]]
            mean_seq[ci] = float(np.mean(ns)) if ns else 0.0
    rows = []
    r2u = {c_: r_ for c_, _, r_ in T.country_r2_list(pf[va], y[va], cs[va])}
    for ci, n, r2c in T.country_r2_list(pf_r[va], y[va], cs[va]):
        m = cs[va] == ci
        excluded = bool(counts is not None and mean_seq.get(ci, 0) < thr)
        rows.append(dict(country=countries[ci], n=n,
                         r2=round(r2c, 4) if r2c is not None else None,
                         r2_uncalibrated=(round(r2u[ci], 4)
                                          if r2u.get(ci) is not None else None),
                         mae=round(float(np.abs(pf_r[va][m] - y[va][m]).mean()), 4),
                         mean_n_seq=round(mean_seq.get(ci, 0.0), 1),
                         excluded_eval=excluded))
    rows.sort(key=lambda r: (r["r2"] is None, -(r["r2"] or 0)))
    import csv
    with open(f"{a.prefix}_per_country.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["country", "n", "r2",
                                          "r2_uncalibrated", "mae",
                                          "mean_n_seq", "excluded_eval"])
        w.writeheader(); w.writerows(rows)

    summary = dict(final_model="v_ensemble", descriptor=desc,
                   members=[dict(model=c["model"], K=c["K"],
                                 target_norm=c["target_norm"],
                                 delta=bool(c.get("delta")),
                                 hidden=c.get("hidden", 128),
                                 dropout=c.get("dropout", 0.1),
                                 seq_weight=c.get("seq_weight", "none"))
                            for c, _, _ in members],
                   split="train: target month <= 2022-07; val: >= 2022-08",
                   n_train=int(tr.sum()), n_val=int(va.sum()),
                   countries=len(countries),
                   rolling=dict(lam=roll_lam, win=roll_win),
                   exclude_mean_seq_below=thr,
                   val_full_ensemble=m_full, val_full_roll=m_full_r,
                   val_full_static_cal=m_full_c,
                   val_entonly_ensemble=m_ent, val_entonly_roll=m_ent_r,
                   emb_contribution=round(float(contrib), 4),
                   emb_contribution_roll=round(float(contrib_r), 4),
                   emb_contribution_rule="(MSE_entonly_ens - MSE_full_ens) / "
                                         "MSE_entonly_ens (未校准)",
                   filtered_eval=f80,
                   acceptance=accept, v1_reference=V1, v2_reference=V2)
    json.dump(summary, open(f"{a.prefix}_metrics.json", "w",
              encoding="utf-8"), ensure_ascii=False, indent=1)

    print(json.dumps(accept, ensure_ascii=False, indent=1))
    print(f"v1 对照(106国): {V1['n_ge_08']}/{V1['n_eval']} "
          f"v2 对照(106国): {V2['n_ge_08']}/{V2['n_eval']}")
    print(f"本次: pooled R2={pem['r2']} 逐国中位="
          f"{pm['per_country_r2_median']} R2>=0.8 国家 "
          f"{pm['n_countries_r2_ge_08']}/{pm['n_countries_eval']} "
          f"(目标>={tgt}) 贡献={pcontrib:.4f}")
    print(f"saved {a.prefix}_best.pt / _entonly_best.pt / "
          "_metrics.json / _per_country.csv")


if __name__ == "__main__":
    main()
