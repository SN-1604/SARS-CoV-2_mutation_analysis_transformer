# -*- coding: utf-8 -*-
"""finalize_monthly_forecast.py — 定稿月熵预测模型
最终模型: residual 架构 (K=3, repr13 月表征 + 历史熵), 3 种子集成 (均值)。
贡献判据: 同架构同超参消融 —— (MSE_ent_ens - MSE_full_ens) / MSE_ent_ens。
产物:
  entropy_forecast_monthly_best.pt      集成权重 (3 种子 state_dict + 配置)
  entropy_forecast_monthly_metrics.json 验证集指标 + 表征贡献 + 各种子明细
  entropy_forecast_monthly_per_country.csv 逐国验证指标
"""
import json, os
import argparse

import numpy as np
import torch
from torch import nn

from train_entropy_forecast_monthly import build_samples, metrics, per_country_r2

RUNS = "monthly_forecast_runs"
TRAIN_END = "2022-07"
K = 3
H = 128
DO = 0.1
FULL = ["residual_full_K3", "r_full_K3_s42", "r_full_K3_s123"]
ENT = ["residual_ent_K3", "r_ent_K3_s42", "r_ent_K3_s123"]


class RES(nn.Module):
    """与训练脚本一致的 residual 架构 (K=3, hidden=128, dropout=0.1)"""

    def __init__(s):
        super().__init__()
        s.emb_enc = nn.Sequential(nn.Linear(64, H), nn.GELU(), nn.Linear(H, 32))
        s.fuse = nn.Sequential(nn.Linear(K * 34, H), nn.GELU(), nn.Dropout(DO),
                               nn.Linear(H, 1))

    def forward(s, x):
        emb, ent, msk = x[..., :64], x[..., 64:65], x[..., 65:66]
        e = s.emb_enc(emb).reshape(len(x), -1)
        corr = s.fuse(torch.cat([e, ent.squeeze(-1), msk.squeeze(-1)],
                                dim=-1)).squeeze(-1)
        w = msk.squeeze(-1) * torch.arange(1, x.shape[1] + 1, device=x.device)
        idx = w.argmax(1, keepdim=True)
        base = ent.squeeze(-1).gather(1, idx).squeeze(1)
        return base + corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", nargs="+", default=FULL)
    ap.add_argument("--ent", nargs="+", default=ENT)
    ap.add_argument("--emb_dir", default="embeddings_monthly")
    ap.add_argument("--prefix", default="entropy_forecast_monthly")
    ap.add_argument("--save_best", action="store_true")
    a = ap.parse_args()
    FULL_R, ENT_R = a.full, a.ent

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import train_entropy_forecast_monthly as T
    T.EMB_DIR = a.emb_dir
    Xe, Xh, Xm, y, cs, ts, countries = build_samples(K)
    tr = ts <= TRAIN_END
    va = ~tr
    mu, sd = np.log1p(y[tr]).mean(), np.log1p(y[tr]).std() + 1e-8
    Xh_z = ((np.log1p(Xh) - mu) / sd * Xm).astype(np.float32)

    def predict(names, zero_emb):
        preds = []
        for nm in names:
            ck = torch.load(os.path.join(RUNS, nm + ".pt"), map_location=device,
                            weights_only=False)
            net = RES().to(device)
            net.load_state_dict(ck["state_dict"])
            net.eval()
            X = Xe * 0.0 if zero_emb else Xe
            Xf = np.concatenate([X, Xh_z[..., None], Xm[..., None]], axis=2)
            with torch.no_grad():
                pz = net(torch.from_numpy(Xf).to(device)).cpu().numpy()
            preds.append(np.expm1(pz * sd + mu))
        return np.mean(preds, axis=0)

    pv_full = predict(FULL_R, zero_emb=False)[va]
    pv_ent = predict(ENT_R, zero_emb=True)[va]
    yv, cv = y[va], cs[va]

    m_full = metrics(pv_full, yv)
    m_full["per_country_r2_median"] = per_country_r2(pv_full, yv, cv)
    m_ent = metrics(pv_ent, yv)
    m_ent["per_country_r2_median"] = per_country_r2(pv_ent, yv, cv)
    contrib = (m_ent["mse"] - m_full["mse"]) / m_ent["mse"]

    # 逐国指标 (full 集成)
    rows = []
    for ci, c in enumerate(countries):
        m = cv == ci
        if m.sum() < 2:
            continue
        ss_res = ((pv_full[m] - yv[m]) ** 2).sum()
        ss_tot = ((yv[m] - yv[m].mean()) ** 2).sum()
        rows.append(dict(country=c, n=int(m.sum()),
                         r2=round(float(1 - ss_res / ss_tot), 4) if ss_tot > 1e-9 else None,
                         mae=round(float(np.abs(pv_full[m] - yv[m]).mean()), 4)))
    rows.sort(key=lambda r: (r["r2"] is None, -(r["r2"] or 0)))

    # 单种子明细 (读各 run metrics)
    per_seed = {}
    for nm in FULL_R + ENT_R:
        mj = json.load(open(os.path.join(RUNS, nm + "_metrics.json")))
        per_seed[nm] = mj["val"]

    summary = dict(
        final_model="residual_ensemble", K=K, full_runs=FULL_R, ent_runs=ENT_R,
        representation=a.emb_dir,
        split="train: target month <= 2022-07; val: >= 2022-08",
        n_train=int(tr.sum()), n_val=int(va.sum()), countries=len(countries),
        val_full_ensemble=m_full,
        val_entonly_ensemble=m_ent,
        emb_contribution=round(float(contrib), 4),
        emb_contribution_rule="(MSE_entonly_ens - MSE_full_ens) / MSE_entonly_ens",
        persistence_baseline_r2=0.8381,
        per_seed=per_seed,
    )
    json.dump(summary, open(f"{a.prefix}_metrics.json", "w"),
              ensure_ascii=False, indent=1)

    import csv
    with open(f"{a.prefix}_per_country.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["country", "n", "r2", "mae"])
        w.writeheader()
        w.writerows(rows)

    if a.save_best:
        states = []
        for nm in FULL_R:
            ck = torch.load(os.path.join(RUNS, nm + ".pt"), map_location="cpu",
                            weights_only=False)
            states.append(ck["state_dict"])
        torch.save(dict(
            arch="residual", ensemble_state_dicts=states, K=K, hidden=H, dropout=DO,
            y_mu=float(mu), y_sd=float(sd), countries=countries,
            note="预测=3种子均值; 输入=[emb64, ent_z, mask]×K个月, 输出=下一月熵(log1p+z 逆变换)",
            emb_dir=a.emb_dir, ent_dir="entropy_monthly_smoothed"),
            f"{a.prefix}_best.pt")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
