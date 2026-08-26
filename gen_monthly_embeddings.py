# -*- coding: utf-8 -*-
"""gen_monthly_embeddings.py — 用 repr13_best.pt 生成各国月度表征 -> embeddings_monthly/
用法 (esm 环境):  python gen_monthly_embeddings.py
输出: embeddings_monthly/{国家}_monthly_embeddings.npz
  months (n,) "YYYY-MM" | embeddings (n,64) f32 | n_windows (n,) i32
周频 13 周窗按【结束日】归入月份后平均: 月 t 的表征只含 ≤ 当月月末的数据, 无前视。
(月度输入版本见 gen_monthly_embeddings_v2.py + reprM_best.pt)
"""
import datetime as dt
import os, pickle

import numpy as np
import torch

PANEL_PKL = "owid_weekly_panel.pkl"
OUT_DIR = "embeddings_monthly"


def month_of(d):
    return f"{d.year:04d}-{d.month:02d}"


def main(batch=512):
    from owid_repr13 import make_model
    ck = torch.load("repr13_best.pt", map_location="cpu", weights_only=False)
    countries = sorted(str(c) for c in ck["countries"])
    cols = [str(c) for c in ck["cols"]]
    mu = ck["mu"].astype(np.float64)
    sd = ck["sd"].astype(np.float64)
    log_mask = ck["log_mask"]
    W = int(ck["W"])

    panel = pickle.load(open(PANEL_PKL, "rb"))
    assert sorted(panel.keys()) == countries, "面板国家与模型元数据不一致"
    M = np.stack([np.asarray(panel[c][cols], dtype=np.float64) for c in countries])
    M[:, :, log_mask] = np.log1p(M[:, :, log_mask])
    Z = ((M - mu) / sd).astype(np.float32)

    w0 = dt.date(2020, 3, 2)
    weeks = [w0 + dt.timedelta(weeks=i) for i in range(148)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {k: v for k, v in ck["cfg"].items() if k != "epochs"}
    model = make_model(len(countries), **cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    with torch.no_grad():
        for ci, c in enumerate(countries):
            wins, owners = [], []
            for t in range(W - 1, 148):
                wins.append(Z[ci, t - W + 1: t + 1, :])
                owners.append(month_of(weeks[t]))              # 按结束周归月
            X = torch.from_numpy(np.stack(wins))
            reps = []
            for i in range(0, len(X), batch):
                out = model(X[i:i + batch].to(device))
                reps.append(out["emb"].cpu().numpy())
            R = np.concatenate(reps).astype(np.float32)
            months = sorted(set(owners))
            E = np.stack([R[[i for i, m in enumerate(owners) if m == mo]].mean(0)
                          for mo in months])
            NW = np.array([owners.count(mo) for mo in months], dtype=np.int32)
            np.savez_compressed(os.path.join(OUT_DIR, f"{c}_monthly_embeddings.npz"),
                                months=np.asarray(months), embeddings=E,
                                n_windows=NW)
            if (ci + 1) % 30 == 0:
                print(f"{ci+1}/{len(countries)}", flush=True)
    print("DONE ->", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
