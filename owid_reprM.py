# -*- coding: utf-8 -*-
"""owid_reprM.py — 月度输入 OWID 表征模型 (镜像 repr13 协议, 周频 -> 月频)
stages:
  data [--W 3]           构建 reprM_dataset_W{W}.npz (月度面板 + 窗口样本 + 归一化)
  train --cfg mX [--W 3] 训练某配置并评估 (kNN/线性探针/聚类) -> reprM_results.json
  summary                  汇总对比表
月度面板: xlsx 日频 -> (国家, 年月) 月均值; 34 个月 2020-03~2022-12。
划分: 窗结束月 ≤2022-07 训练 / ≥2022-08 验证。归一化仅用训练覆盖月。
"""
import argparse, copy, json, os, random, time

import numpy as np
import pandas as pd

XLSX = "owid_cleaned_weekly70_v3.xlsx"
RES_JSON = "reprM_results.json"
MONTHS = [f"{y:04d}-{m:02d}" for y in (2020, 2021, 2022)
          for m in range(1, 13) if (y, m) >= (2020, 3) and (y, m) <= (2022, 12)]  # 34
TRAIN_END_IDX = MONTHS.index("2022-07")     # 窗结束月 <= 2022-07 -> 训练


def get_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


# ---------------- data ----------------
def build_dataset(W):
    df = pd.read_excel(XLSX)
    df["date"] = pd.to_datetime(df["date"])
    df["ym"] = df["date"].dt.strftime("%Y-%m")
    cols = [c for c in df.columns if c not in ("country", "date", "ym")]
    countries = sorted(df["country"].unique())
    panel = (df.groupby(["country", "ym"])[cols].mean().reset_index())
    M = np.full((len(countries), len(MONTHS), len(cols)), np.nan)
    idx_c = {c: i for i, c in enumerate(countries)}
    idx_m = {m: i for i, m in enumerate(MONTHS)}
    pv = panel.set_index(["country", "ym"]).sort_index()
    for (c, m), row in pv.iterrows():
        M[idx_c[c], idx_m[m]] = row.values.astype(np.float64)
    assert not np.isnan(M).any(), "月度面板存在缺失"

    # 归一化: 训练覆盖月 (窗结束<=2022-07 -> 涉及月份 <= 2022-07) 统计
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
    Z = (Ml - mu) / sd

    Xtr, ytr, ttr, nxt_tr, Xva, yva, tva = [], [], [], [], [], [], []
    for ci in range(len(countries)):
        for t in range(W - 1, len(MONTHS)):
            win = Z[ci, t - W + 1: t + 1, :]
            if t <= TRAIN_END_IDX:
                Xtr.append(win); ytr.append(ci); ttr.append(t)
                nxt_tr.append(Z[ci, t + 1, :] if t + 1 <= TRAIN_END_IDX
                              else np.full(len(cols), np.nan))
            else:
                Xva.append(win); yva.append(ci); tva.append(t)
    out = dict(Xtr=np.asarray(Xtr, np.float32), ytr=np.asarray(ytr, np.int64),
               ttr=np.asarray(ttr, np.int64), NTR=np.asarray(nxt_tr, np.float32),
               Xva=np.asarray(Xva, np.float32), yva=np.asarray(yva, np.int64),
               tva=np.asarray(tva, np.int64),
               mu=mu.astype(np.float32), sd=sd.astype(np.float32),
               log_mask=log_mask, countries=np.asarray(countries),
               cols=np.asarray(cols), months=np.asarray(MONTHS), W=np.asarray(W))
    f = f"reprM_dataset_W{W}.npz"
    np.savez_compressed(f, **out)
    print("saved", f, "Xtr", out["Xtr"].shape, "Xva", out["Xva"].shape,
          "log1p cols", int(log_mask.sum()))
    return out


# ---------------- model ----------------
def make_model(ncls, W, d=96, layers=3, heads=4, emb=64, drop=0.1, aux="none"):
    torch, nn, F = get_torch()

    class Repr(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(38, d)
            self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, d * 4, drop,
                                               batch_first=True, norm_first=True,
                                               activation="gelu")
            self.enc = nn.TransformerEncoder(layer, layers)
            self.emb = nn.Sequential(nn.Linear(d, emb), nn.LayerNorm(emb))
            self.cls = nn.Linear(emb, ncls)
            self.aux = aux
            if aux == "next":
                self.nxt = nn.Linear(emb, 38)
            if aux == "mask":
                self.dec = nn.Linear(d, 38)

        def forward(self, x, mask_ratio=0.0):
            h = self.proj(x) + self.pos
            keep = None
            if self.aux == "mask" and mask_ratio > 0:
                m = torch.rand(x.shape[0], x.shape[1], device=x.device) < mask_ratio
                keep = m
                h = h.masked_fill(m.unsqueeze(-1), 0.0)
            h = self.enc(h)
            e = self.emb(h.mean(1))
            out = {"emb": e, "logits": self.cls(e)}
            if self.aux == "next":
                out["next"] = self.nxt(e)
            if self.aux == "mask":
                out["recon"] = self.dec(h)
                out["mask"] = keep
            return out

    return Repr()


def supcon_loss(z, y, tau=0.1):
    torch, nn, F = get_torch()
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    same = (y[:, None] == y[None, :]).float()
    same.fill_diagonal_(0.0)
    logp = F.log_softmax(sim, dim=1)
    denom = same.sum(1).clamp(min=1.0)
    return -(logp * same).sum(1).div(denom).mean()


CFGS = {
    "m1_ce":      dict(W=3, d=96,  layers=3, heads=4, emb=64, drop=0.1, aux="none",   epochs=80),
    "m2_next":    dict(W=3, d=96,  layers=3, heads=4, emb=64, drop=0.1, aux="next",   epochs=80),
    "m3_mask":    dict(W=3, d=128, layers=4, heads=8, emb=64, drop=0.1, aux="mask",   epochs=100),
    "m4_supcon":  dict(W=3, d=96,  layers=3, heads=4, emb=64, drop=0.1, aux="supcon", epochs=80),
    "m5_small":   dict(W=3, d=64,  layers=2, heads=4, emb=48, drop=0.1, aux="none",   epochs=80),
    "m6_w6_supcon": dict(W=6, d=96, layers=3, heads=4, emb=64, drop=0.1, aux="supcon", epochs=80),
    "m7_w6_ce":   dict(W=6, d=96,  layers=3, heads=4, emb=64, drop=0.1, aux="none",   epochs=80),
    # 细化轮 (围绕 m1/m4)
    "m8_ce_drop0":  dict(W=3, d=96,  layers=3, heads=4, emb=64, drop=0.0, aux="none", epochs=80),
    "m9_ce_drop02": dict(W=3, d=96,  layers=3, heads=4, emb=64, drop=0.2, aux="none", epochs=80),
    "m10_ce_d128":  dict(W=3, d=128, layers=4, heads=8, emb=64, drop=0.1, aux="none", epochs=80),
    "m11_ce_emb96": dict(W=3, d=96,  layers=3, heads=4, emb=96, drop=0.1, aux="none", epochs=80),
    "m12_supcon_tau05": dict(W=3, d=96, layers=3, heads=4, emb=64, drop=0.1, aux="supcon", epochs=80),
}


def embed_all(model, X, device, bs=1024):
    torch, _, _ = get_torch()
    model.eval()
    E = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            E.append(model(xb)["emb"].cpu().numpy())
    return np.concatenate(E)


def eval_repr(Etr, ytr, Eva, yva, tag, ncls, res):
    """kNN(cosine,5) 国别 + 线性探针 + KMeans ARI/NMI + silhouette (与 repr13 同)"""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.metrics import (adjusted_rand_score,
                                 normalized_mutual_info_score, silhouette_score)
    from sklearn.preprocessing import normalize
    r = {}
    kn = KNeighborsClassifier(5, metric="cosine", n_jobs=8).fit(Etr, ytr)
    r["knn_acc"] = float(kn.score(Eva, yva))
    lr = LogisticRegression(max_iter=1500, C=1.0, n_jobs=8).fit(Etr, ytr)
    r["lin_acc"] = float(lr.score(Eva, yva))
    En = normalize(Eva)
    km = KMeans(n_clusters=ncls, n_init=4, random_state=0).fit(En)
    r["ari"] = float(adjusted_rand_score(yva, km.labels_))
    r["nmi"] = float(normalized_mutual_info_score(yva, km.labels_))
    r["sil"] = float(silhouette_score(En, yva, sample_size=2500, random_state=0))
    res[tag] = r
    print(tag, {k: round(v, 4) for k, v in r.items()}, flush=True)
    return r


def train_one(cfg_name, seed=42):
    tag = cfg_name if seed == 42 else f"{cfg_name}_s{seed}"
    torch, nn, F = get_torch()
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CFGS[cfg_name]
    W = cfg["W"]
    ds = np.load(f"reprM_dataset_W{W}.npz", allow_pickle=True)
    Xtr, ytr, NTR = ds["Xtr"], ds["ytr"], ds["NTR"]
    Xva, yva = ds["Xva"], ds["yva"]
    ncls = len(ds["countries"])
    model = make_model(ncls, W, **{k: v for k, v in cfg.items()
                                   if k not in ("epochs", "W")}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = cfg["epochs"]
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    n = len(Xtr); bs = 512
    aux = cfg["aux"]
    best = -1.0; best_state = None; bad = 0
    t0 = time.time()
    from sklearn.neighbors import KNeighborsClassifier
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            out = model(xb, mask_ratio=0.2 if aux == "mask" else 0.0)
            loss = F.cross_entropy(out["logits"], yb)
            if aux == "next":
                nb = torch.from_numpy(NTR[idx]).to(device)
                ok = ~torch.isnan(nb).any(1)
                if ok.any():
                    loss = loss + 0.5 * F.mse_loss(out["next"][ok], nb[ok])
            elif aux == "mask":
                m = out["mask"]
                if m.any():
                    loss = loss + 0.5 * F.mse_loss(out["recon"][m], xb[m])
            elif aux == "supcon":
                loss = loss + 0.3 * supcon_loss(out["emb"], yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        sch.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            Etr = embed_all(model, Xtr, device)
            Eva = embed_all(model, Xva, device)
            acc = KNeighborsClassifier(5, metric="cosine", n_jobs=8).fit(Etr, ytr).score(Eva, yva)
            if acc > best:
                best = acc; bad = 0
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            else:
                bad += 1
            print(f"{cfg_name} ep{ep+1} loss {tot/n:.4f} knn {acc:.4f} best {best:.4f}",
                  flush=True)
            if bad >= 3:
                break
    ckpt = {"state_dict": best_state, "cfg": cfg, "cfg_name": cfg_name,
            "val_knn_acc_monitor": best,
            "mu": ds["mu"], "sd": ds["sd"], "log_mask": ds["log_mask"],
            "countries": ds["countries"], "cols": ds["cols"],
            "months": ds["months"], "W": W, "granularity": "monthly"}
    torch.save(ckpt, f"reprM_model_{tag}.pt")
    print("saved", f"reprM_model_{tag}.pt", "train_sec", round(time.time() - t0, 1),
          flush=True)
    model.load_state_dict(best_state)
    Etr = embed_all(model, Xtr, device); Eva = embed_all(model, Xva, device)
    res = json.load(open(RES_JSON, encoding="gb18030")) if os.path.exists(RES_JSON) else {}
    eval_repr(Etr, ytr, Eva, yva, tag, ncls, res)
    json.dump(res, open(RES_JSON, "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(f"reprM_emb_{tag}.npz", Etr=Etr, Eva=Eva, ytr=ytr, yva=yva)


def eval_baseline(W=3):
    ds = np.load(f"reprM_dataset_W{W}.npz", allow_pickle=True)
    res = json.load(open(RES_JSON, encoding="gb18030")) if os.path.exists(RES_JSON) else {}
    eval_repr(ds["Xtr"].reshape(len(ds["Xtr"]), -1), ds["ytr"],
              ds["Xva"].reshape(len(ds["Xva"]), -1), ds["yva"],
              f"RAW_baseline_W{W}", len(ds["countries"]), res)
    json.dump(res, open(RES_JSON, "w"), ensure_ascii=False, indent=1)


def summary():
    res = json.load(open(RES_JSON))
    print(f"{'cfg':<18} {'knn':>7} {'lin':>7} {'ari':>7} {'nmi':>7} {'sil':>7}")
    for k, v in res.items():
        print(f"{k:<18} {v['knn_acc']:7.4f} {v['lin_acc']:7.4f} {v['ari']:7.4f} "
              f"{v['nmi']:7.4f} {v['sil']:7.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "train", "baseline", "summary"])
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--W", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.stage == "data":
        build_dataset(a.W)
    elif a.stage == "train":
        train_one(a.cfg, seed=a.seed)
    elif a.stage == "baseline":
        eval_baseline(a.W)
    else:
        summary()
