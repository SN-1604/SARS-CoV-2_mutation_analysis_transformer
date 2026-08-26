# -*- coding: utf-8 -*-
"""owid_repr13.py — 13周窗口 OWID 数据的 Transformer 表征模型
stages:
  data                     构建 repr13_dataset.npz（窗口样本 + 归一化统计）
  train --cfg v1           训练某个配置并评估（kNN/线性探针/聚类），结果写入 repr13_results.json
  eval                     汇总所有配置 + 原始特征基线，输出对比表
"""
import argparse, json, pickle, datetime as dt, os, random, copy, time
import numpy as np
import pandas as pd

W = 13
VAL_START = dt.date(2022, 8, 1)
TRAIN_END = dt.date(2022, 7, 31)
PANEL_PKL = "owid_weekly_panel.pkl"
XLSX = "owid_cleaned_weekly70_v3.xlsx"
DS_NPZ = "repr13_dataset.npz"
RES_JSON = "repr13_results.json"


# ---------------- data ----------------
def build_dataset():
    panel = pickle.load(open(PANEL_PKL, "rb"))
    cols = [c for c in pd.read_excel(XLSX, nrows=1).columns if c not in ("country", "date")]
    countries = sorted(panel.keys())
    w0 = dt.date(2020, 3, 2)
    weeks = [w0 + dt.timedelta(weeks=i) for i in range(148)]
    # 堆叠 (141,148,38)
    M = np.stack([np.asarray(panel[c][cols], dtype=np.float64) for c in countries])
    assert not np.isnan(M).any(), "panel 应无缺失"

    # 归一化统计：仅用训练窗口覆盖的周（idx 0..125）
    train_flat = M[:, :126, :].reshape(-1, 38)
    log_mask = np.zeros(38, bool)
    for j in range(38):
        v = train_flat[:, j]
        if v.min() >= 0:
            sk = pd.Series(v).skew()
            if sk > 2:
                log_mask[j] = True
    Ml = M.copy()
    Ml[:, :, log_mask] = np.log1p(Ml[:, :, log_mask])
    tf = Ml[:, :126, :].reshape(-1, 38)
    mu, sd = tf.mean(0), tf.std(0)
    sd[sd < 1e-8] = 1.0
    Z = (Ml - mu) / sd  # z 域

    Xtr, ytr, ttr, Xva, yva, tva, nxt_tr = [], [], [], [], [], [], []
    for ci in range(len(countries)):
        for t in range(W - 1, 148):
            win = Z[ci, t - W + 1: t + 1, :]
            d = weeks[t]
            if d <= TRAIN_END:
                Xtr.append(win); ytr.append(ci); ttr.append(t)
                nxt_tr.append(Z[ci, t + 1, :] if t + 1 <= 125 else np.full(38, np.nan))
            elif d >= VAL_START:
                Xva.append(win); yva.append(ci); tva.append(t)
    out = dict(
        Xtr=np.asarray(Xtr, np.float32), ytr=np.asarray(ytr, np.int64),
        ttr=np.asarray(ttr, np.int64), NTR=np.asarray(nxt_tr, np.float32),
        Xva=np.asarray(Xva, np.float32), yva=np.asarray(yva, np.int64),
        tva=np.asarray(tva, np.int64),
        mu=mu.astype(np.float32), sd=sd.astype(np.float32), log_mask=log_mask,
        countries=np.asarray(countries), cols=np.asarray(cols),
        weeks=np.asarray([w.isoformat() for w in weeks]))
    np.savez_compressed(DS_NPZ, **out)
    print("saved", DS_NPZ, "Xtr", out["Xtr"].shape, "Xva", out["Xva"].shape,
          "countries", len(countries), "log1p cols", int(log_mask.sum()))
    return out


# ---------------- model ----------------
def get_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def make_model(ncls, d=96, layers=3, heads=4, emb=64, drop=0.1, aux="none"):
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
                B, T, _ = x.shape
                m = torch.rand(B, T, device=x.device) < mask_ratio
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


CFGS = {
    "v1_ce":        dict(d=96,  layers=3, heads=4, emb=64, drop=0.1, aux="none", epochs=80),
    "v2_next":      dict(d=96,  layers=3, heads=4, emb=64, drop=0.1, aux="next", epochs=80),
    "v3_mask":      dict(d=128, layers=4, heads=8, emb=64, drop=0.1, aux="mask", epochs=100),
    "v4_supcon":    dict(d=96,  layers=3, heads=4, emb=64, drop=0.1, aux="supcon", epochs=80),
    "v5_small":     dict(d=64,  layers=2, heads=4, emb=48, drop=0.1, aux="none", epochs=80),
}


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
    """kNN(cosine,5) 国别准确率 + 线性探针 + k-means ARI/NMI + silhouette"""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
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


def train_one(cfg_name):
    torch, nn, F = get_torch()
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = np.load(DS_NPZ, allow_pickle=True)
    Xtr, ytr, NTR = ds["Xtr"], ds["ytr"], ds["NTR"]
    Xva, yva = ds["Xva"], ds["yva"]
    ncls = len(ds["countries"])
    cfg = CFGS[cfg_name]
    model = make_model(ncls, **{k: v for k, v in cfg.items() if k != "epochs"}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = cfg["epochs"]
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    n = len(Xtr); bs = 512
    aux = cfg["aux"]
    best = -1.0; best_state = None; patience = 15; bad = 0
    t0 = time.time()
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
        # 早停监控：val kNN（每5轮）
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            Etr = embed_all(model, Xtr, device)
            Eva = embed_all(model, Xva, device)
            from sklearn.neighbors import KNeighborsClassifier
            acc = KNeighborsClassifier(5, metric="cosine", n_jobs=8).fit(Etr, ytr).score(Eva, yva)
            if acc > best:
                best = acc; bad = 0
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            else:
                bad += 1
            print(f"{cfg_name} ep{ep+1} loss {tot/n:.4f} knn {acc:.4f} best {best:.4f}", flush=True)
            if bad >= patience // 5:
                break
    model.load_state_dict(best_state)
    ckpt = {"state_dict": best_state, "cfg": cfg, "cfg_name": cfg_name,
            "val_knn_acc_monitor": best,
            "mu": ds["mu"], "sd": ds["sd"], "log_mask": ds["log_mask"],
            "countries": ds["countries"], "cols": ds["cols"], "W": W}
    torch.save(ckpt, f"repr13_model_{cfg_name}.pt")
    print("saved", f"repr13_model_{cfg_name}.pt", "train_sec", round(time.time() - t0, 1))
    # 完整评估
    Etr = embed_all(model, Xtr, device); Eva = embed_all(model, Xva, device)
    res = json.load(open(RES_JSON, encoding="gb18030")) if os.path.exists(RES_JSON) else {}
    eval_repr(Etr, ytr, Eva, yva, cfg_name, ncls, res)
    json.dump(res, open(RES_JSON, "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(f"repr13_emb_{cfg_name}.npz", Etr=Etr, Eva=Eva, ytr=ytr, yva=yva)


def eval_baseline():
    """原始特征（13×38 拉平）基线"""
    ds = np.load(DS_NPZ, allow_pickle=True)
    Xtr = ds["Xtr"].reshape(len(ds["Xtr"]), -1)
    Xva = ds["Xva"].reshape(len(ds["Xva"]), -1)
    res = json.load(open(RES_JSON, encoding="gb18030")) if os.path.exists(RES_JSON) else {}
    eval_repr(Xtr, ds["ytr"], Xva, ds["yva"], "RAW_baseline", len(ds["countries"]), res)
    json.dump(res, open(RES_JSON, "w"), ensure_ascii=False, indent=1)


def eval_summary():
    res = json.load(open(RES_JSON, encoding="gb18030"))
    print(f"{'config':<16}{'knn_acc':>9}{'lin_acc':>9}{'ARI':>8}{'NMI':>8}{'sil':>8}")
    for k, r in sorted(res.items()):
        print(f"{k:<16}{r['knn_acc']:>9.4f}{r['lin_acc']:>9.4f}{r['ari']:>8.4f}{r['nmi']:>8.4f}{r['sil']:>8.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "train", "eval", "baseline"])
    ap.add_argument("--cfg", default="v1_ce")
    a = ap.parse_args()
    if a.stage == "data":
        build_dataset()
    elif a.stage == "train":
        train_one(a.cfg)
    elif a.stage == "baseline":
        eval_baseline()
    else:
        eval_summary()
