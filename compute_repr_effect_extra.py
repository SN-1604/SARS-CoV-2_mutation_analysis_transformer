# -*- coding: utf-8 -*-
"""compute_repr_effect_extra.py — 为 fig9 a/b 补充表征优势的量化数据

输出 repr_effect_extra.json:
  1) probe_raw: 原始月度特征(无表征, 38维×K=3=114维) Ridge 探针, 与 reprM_m4
     同一样本集(键取自 embeddings_monthly + entropy_monthly_smoothed),
     协议同 repr_model_selection.py (RidgeCV, log1p z 归一化);
     同时复算 probe_emb 作锚点校验(期望 r2≈0.0841, n_train=2706, n_val=488)。
  2) per_country_full / per_country_ent: v2 全模型与纯熵消融集成的逐国
     未校准 R² (V4Forecaster 推理; 消融为表征通道置零);
     锚点校验: 全模型 pooled R²≈0.8320, 消融≈0.7466; 全模型逐国值与
     entropy_forecast_monthly_v2_per_country.csv 的 r2_uncalibrated 一致。
"""
import json
import os

import numpy as np
import pandas as pd

from repr_model_selection import build_probe_samples, ridge_probe, load_embeddings, load_entropy
from owid_reprM import MONTHS, TRAIN_END_IDX

OUT = "repr_effect_extra.json"

# ---------------- 1) 原始特征基线探针 ----------------
ents = load_entropy()
emb = load_embeddings("embeddings_monthly")
S_emb = build_probe_samples(emb, ents)
probe_emb = ridge_probe(S_emb)
print("probe_emb(锚点):", json.dumps(probe_emb, ensure_ascii=False), flush=True)

# 月度原始面板 (与 owid_reprM.build_dataset 相同归一化协议)
df = pd.read_excel("owid_cleaned_weekly70_v3.xlsx")
df["date"] = pd.to_datetime(df["date"])
df["ym"] = df["date"].dt.strftime("%Y-%m")
cols = [c for c in df.columns if c not in ("country", "date", "ym")]
countries_r = sorted(df["country"].unique())
pv = df.groupby(["country", "ym"])[cols].mean()
M = np.full((len(countries_r), len(MONTHS), len(cols)), np.nan)
ic = {c: i for i, c in enumerate(countries_r)}
im = {m: i for i, m in enumerate(MONTHS)}
for (c, m), row in pv.iterrows():
    M[ic[c], im[m]] = row.values.astype(np.float64)
assert not np.isnan(M).any()
tf = M[:, :TRAIN_END_IDX + 1, :].reshape(-1, len(cols))
log_mask = np.zeros(len(cols), bool)
for j in range(len(cols)):
    v = tf[:, j]
    if v.min() >= 0 and pd.Series(v).skew() > 2:
        log_mask[j] = True
Ml = M.copy()
Ml[:, :, log_mask] = np.log1p(Ml[:, :, log_mask])
tf = Ml[:, :TRAIN_END_IDX + 1, :].reshape(-1, len(cols))
mu, sd = tf.mean(0), tf.std(0)
sd[sd < 1e-8] = 1.0
Z = (Ml - mu) / sd                                    # (141, 34, 38) z 域
RAW = {c: {MONTHS[j]: Z[i, j] for j in range(len(MONTHS))}
       for i, c in enumerate(countries_r)}

K = 3
S_raw = {}
for (c, tgt), (x, y) in S_emb.items():
    i = MONTHS.index(tgt) if tgt in MONTHS else None
    # 历史月 = 表征样本的 K 个输入月: 由 S_emb 键反推目标月的前 K 个月
    # (与 build_probe_samples 相同的面板顺序口径)
    ms_c = [m for m in MONTHS if m <= tgt]
    if len(ms_c) < K + 1 or tgt not in RAW.get(c, {}):
        continue
    hist = ms_c[-K - 1:-1]
    S_raw[(c, tgt)] = (np.concatenate([RAW[c][h] for h in hist]), y)
common = sorted(set(S_emb) & set(S_raw))
print("raw 样本键: %d (与 emb 交集 %d)" % (len(S_raw), len(common)))
probe_raw = ridge_probe({k: S_raw[k] for k in common})
print("probe_raw:", json.dumps(probe_raw, ensure_ascii=False), flush=True)

# ---------------- 2) v2 逐国 R²: 全模型 vs 纯熵消融 ----------------
import torch

from forecast_v4_adapter import V4Forecaster

device = "cuda" if torch.cuda.is_available() else "cpu"
fc = V4Forecaster(ckpt="entropy_forecast_monthly_v2_best.pt", device=device)
vi = fc.val_idx
# K_max 成员的 Xf 表征通道即观测月度表征(口径与事后解释管线一致)
D_max = [mb["D"] for mb in fc.members if mb["K"] == fc.K_max][0]
Em_obs = D_max["Xf"][vi][:, :, :64].astype(np.float32)
y_full = fc.predict_val(Em_obs)
yv = fc.y[vi]
pooled_full = 1 - ((y_full - yv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
print("anchor full pooled R2 = %.4f (期望 0.8320)" % pooled_full, flush=True)

fc_e = V4Forecaster(ckpt="entropy_forecast_monthly_v2_entonly_best.pt",
                    device=device)
y_ent = fc_e.predict_val(np.zeros_like(Em_obs))       # 表征通道置零
pooled_ent = 1 - ((y_ent - yv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
print("anchor ent_only pooled R2 = %.4f (期望 0.7466)" % pooled_ent, flush=True)

assert np.array_equal(fc.cs[vi], fc_e.cs[vi]) and \
    np.array_equal(fc.y[vi], fc_e.y[vi])
cs_v, countries_f = fc.cs[vi], fc.countries


def per_country_r2(pred):
    out = {}
    for ci in range(len(countries_f)):
        m = cs_v == ci
        if m.sum() < 2:
            continue
        y_ = yv[m]
        denom = ((y_ - y_.mean()) ** 2).sum()
        if denom <= 0:
            continue
        out[countries_f[ci]] = float(1 - ((pred[m] - y_) ** 2).sum() / denom)
    return out


r2_full = per_country_r2(y_full)
r2_ent = per_country_r2(y_ent)

# 106 国评估口径: entropy_forecast_monthly_v2_per_country.csv 在册国家
pcsv = pd.read_csv("entropy_forecast_monthly_v2_per_country.csv")
keep = sorted(pcsv["country"].tolist())
print("评估口径国家数: %d" % len(keep))
# 全模型未校准值与 CSV 一致性抽查(信息项; 单代码路径自洽值为准)
cmp = [(c, r2_full.get(c), r_) for c, r_ in
       zip(pcsv["country"], pcsv["r2_uncalibrated"]) if c in r2_full]
diffs = [abs(a - b) for _, a, b in cmp if a is not None]
print("与CSV r2_uncalibrated 最大差: %.4f (n=%d)" % (max(diffs), len(diffs)))

json.dump(dict(
    probe_raw=probe_raw, probe_emb=probe_emb,
    pooled_full_r2=float(pooled_full), pooled_ent_r2=float(pooled_ent),
    eval_countries=keep,
    per_country_full={c: r2_full[c] for c in keep},
    per_country_ent={c: r2_ent.get(c) for c in keep},
), open(OUT, "w"), ensure_ascii=False, indent=1)
print("saved", OUT)
