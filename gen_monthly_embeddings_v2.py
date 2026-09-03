# -*- coding: utf-8 -*-
"""gen_monthly_embeddings_v2.py — 用 reprM_best.pt (月度输入模型) 生成月度表征
输出: embeddings_monthly_v2/{国家}_monthly_embeddings.npz
  months (n,) "YYYY-MM" | embeddings (n,emb) f32 | n_windows (n,) i32 (=1)
窗结束月 t 的表征即该月表征; 窗内数据全部 ≤ 当月末, 无前视。
"""
import os

import numpy as np
import pandas as pd
import torch

from owid_reprM import MONTHS, TRAIN_END_IDX, get_torch

XLSX = "owid_cleaned_weekly70_v3.xlsx"
CKPT = "reprM_best.pt"
OUT_DIR = "embeddings_monthly_v2"


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    W = int(ck["W"])
    countries = [str(c) for c in ck["countries"]]
    cols = [str(c) for c in ck["cols"]]
    mu = ck["mu"].astype(np.float64)
    sd = ck["sd"].astype(np.float64)
    log_mask = ck["log_mask"]

    df = pd.read_excel(XLSX)
    df["date"] = pd.to_datetime(df["date"])
    df["ym"] = df["date"].dt.strftime("%Y-%m")
    panel = df.groupby(["country", "ym"])[cols].mean().reset_index()
    M = np.full((len(countries), len(MONTHS), len(cols)), np.nan)
    idx_c = {c: i for i, c in enumerate(countries)}
    idx_m = {m: i for i, m in enumerate(MONTHS)}
    pv = panel.set_index(["country", "ym"]).sort_index()
    for (c, m), row in pv.iterrows():
        M[idx_c[c], idx_m[m]] = row.values.astype(np.float64)
    assert not np.isnan(M).any()
    M[:, :, log_mask] = np.log1p(M[:, :, log_mask])
    Z = ((M - mu) / sd).astype(np.float32)

    from owid_reprM import make_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(len(countries), W,
                       **{k: v for k, v in cfg.items()
                          if k not in ("epochs", "W")}).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    with torch.no_grad():
        for ci, c in enumerate(countries):
            X = np.stack([Z[ci, t - W + 1: t + 1, :]
                          for t in range(W - 1, len(MONTHS))])
            E = []
            for i in range(0, len(X), 1024):
                out = model(torch.from_numpy(X[i:i + 1024]).to(device))
                E.append(out["emb"].cpu().numpy())
            E = np.concatenate(E).astype(np.float32)
            months = MONTHS[W - 1:]
            np.savez_compressed(
                os.path.join(OUT_DIR, f"{c}_monthly_embeddings.npz"),
                months=np.asarray(months), embeddings=E,
                n_windows=np.ones(len(months), np.int32))
    print("DONE ->", OUT_DIR, f"(W={W}, months {MONTHS[W-1]}..{MONTHS[-1]})")


if __name__ == "__main__":
    main()
