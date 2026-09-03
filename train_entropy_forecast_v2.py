# -*- coding: utf-8 -*-
"""train_entropy_forecast_v2.py — 月熵预测 v2 (不改原脚本, 面向逐国 R2 优化)

相对 train_entropy_forecast_monthly.py 的新增:
  --target_norm {global,country,window}  目标/熵史归一化方式
      global : 与原脚本一致 (训练段全局 log1p+z)
      country: 逐国训练段 log1p+z (推理按国查表逆变换; <5 点回退全局)
      window : 用样本自身 K 月已观测熵史的均值/方差归一化 (不涉国家信息)
  --calibrate        逐国仿射校准 (训练段预测上拟合 y≈a+b*pred, 向 a=0,b=1 岭收缩)
  --cal_lambda       校准岭收缩强度 (默认 10)
  --half_life        训练样本时间衰减半衰期(月, 相对 2022-07; 0=关闭)
  --balanced_sampler 按国均衡采样 (仅训练侧)
  --delta            熵史差分特征
  架构: res2v2 (熵主路径+表征零初始化校正), resv2 (持续性残差), mlp, gru, transformer
  --mode full/ent_only (ent_only=表征通道置零的消融对照)
约束: 无国家身份输入特征; full 模式强制保留表征通道。
每轮写 monthly_forecast_runs_v2/{name}_metrics.json 与 {name}.pt
运行: G:/Anaconda3/envs/esm/python.exe train_entropy_forecast_v2.py ...
"""
import argparse, json, os, time

import numpy as np
import torch
from torch import nn

import train_entropy_forecast_monthly as T1
from train_entropy_forecast_monthly import build_samples, MIDX, TRAIN_END

RUNS = "monthly_forecast_runs_v2"


# ---------------------------------------------------------------- 归一化
class Norm:
    """目标与熵史的 log1p 归一化/逆变换 (global | country | window)"""

    def __init__(self, mode, y, cs, tr, countries):
        self.mode = mode
        lg = np.log1p(y)
        self.g_mu = float(lg[tr].mean())
        self.g_sd = float(lg[tr].std() + 1e-8)
        self.c_mu = {}
        self.c_sd = {}
        if mode == "country":
            for ci, c in enumerate(countries):
                m = (cs == ci) & tr
                if m.sum() >= 5:
                    v = lg[m]
                    self.c_mu[c] = float(v.mean())
                    self.c_sd[c] = float(max(v.std(), 0.1 * self.g_sd, 1e-3))

    def stats(self, Xh_raw, Xm, cs, countries):
        """返回逐样本 (mu[N], sd[N]) 用于 log1p 空间归一化"""
        N = len(Xh_raw)
        if self.mode == "window":
            lg = np.log1p(np.where(Xm > 0, Xh_raw, np.nan))   # (N,K)
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(lg, axis=1)
                sd = np.nanstd(lg, axis=1)
            bad = ~(np.isfinite(mu) & np.isfinite(sd)) | (sd < 1e-3) \
                | ((Xm > 0).sum(1) < 2)
            mu = np.where(bad, self.g_mu, mu)
            sd = np.where(bad, self.g_sd, np.maximum(sd, 1e-3))
            return mu.astype(np.float64), sd.astype(np.float64)
        mu = np.full(N, self.g_mu)
        sd = np.full(N, self.g_sd)
        if self.mode == "country":
            for i in range(N):
                c = countries[cs[i]]
                if c in self.c_mu:
                    mu[i] = self.c_mu[c]
                    sd[i] = self.c_sd[c]
        return mu, sd

    @staticmethod
    def forward(lg, mu, sd):
        return (lg - mu) / sd

    @staticmethod
    def inverse(z, mu, sd):
        return np.expm1(z * sd + mu)

    def state(self):
        return dict(mode=self.mode, g_mu=self.g_mu, g_sd=self.g_sd,
                    c_mu=self.c_mu, c_sd=self.c_sd)

    @classmethod
    def from_state(cls, st):
        obj = cls.__new__(cls)
        obj.mode = st["mode"]
        obj.g_mu = st["g_mu"]
        obj.g_sd = st["g_sd"]
        obj.c_mu = {str(k): float(v) for k, v in st["c_mu"].items()}
        obj.c_sd = {str(k): float(v) for k, v in st["c_sd"].items()}
        return obj


def load_counts(path):
    """seqcounts_monthly.csv -> {country: {month: n}}"""
    import csv as _csv
    out = {}
    for r in _csv.DictReader(open(path, encoding="utf-8")):
        out.setdefault(r["country"], {})[r["month"]] = int(r["n_seq"] or 0)
    return out


def prepare_data(K, target_norm, delta, mode, ent_dir=None, counts_file=None):
    """构建样本并按配置变换, 返回 dict; mode=ent_only 时表征通道置零
    ent_dir: 熵目录 (默认 entropy_monthly_smoothed; 尾窗口径传
    entropy_monthly_smoothed_trailing)"""
    T1.ENT_DIR = ent_dir or "entropy_monthly_smoothed"
    Xe, Xh_raw, Xm, y, cs, ts, countries = build_samples(K)
    tr = ts <= TRAIN_END
    va = ~tr
    norm = Norm(target_norm, y, cs, tr, countries)
    mu, sd = norm.stats(Xh_raw, Xm, cs, countries)
    yz = norm.forward(np.log1p(y), mu, sd).astype(np.float32)
    Xh_z = (norm.forward(np.log1p(np.where(Xm > 0, Xh_raw, 1.0)),
                         mu[:, None], sd[:, None]) * Xm).astype(np.float32)
    feats = [Xe, Xh_z[..., None], Xm[..., None]]
    if delta:
        feats.append(np.diff(Xh_z, axis=1, prepend=Xh_z[:, :1])[..., None])
    Xf = np.concatenate(feats, axis=2)
    if mode == "ent_only":
        Xf = Xf.copy()
        Xf[..., :64] = 0.0
    n_seq = None
    if counts_file:
        cnt = load_counts(counts_file)
        n_seq = np.array([max(cnt.get(countries[cs[i]], {}).get(ts[i], 0), 1)
                          for i in range(len(y))], dtype=np.float64)
    return dict(Xf=Xf.astype(np.float32), yz=yz, y=y, cs=cs, ts=ts,
                countries=countries, tr=tr, va=va, norm=norm, mu=mu, sd=sd,
                F=Xf.shape[2], ent_dir=T1.ENT_DIR, n_seq=n_seq)


# ---------------------------------------------------------------- 模型
def make_model(model, K, hidden, layers, dropout, F):
    F_E = F - 64
    H, DO = hidden, dropout

    class RES2V2(nn.Module):
        """熵主路径 MLP + 表征零初始化校正路径 (full 模式表征强制保留)"""

        def __init__(s):
            super().__init__()
            s.ent_mlp = nn.Sequential(
                nn.Linear(K * F_E, H), nn.GELU(), nn.Dropout(DO),
                nn.Linear(H, H // 2), nn.GELU(), nn.Linear(H // 2, 1))
            s.emb_enc = nn.Sequential(nn.Linear(64, H), nn.GELU(),
                                      nn.Linear(H, 32))
            s.corr = nn.Sequential(nn.Linear(K * 32, H), nn.GELU(),
                                   nn.Dropout(DO), nn.Linear(H, 1))
            nn.init.zeros_(s.corr[-1].weight)
            nn.init.zeros_(s.corr[-1].bias)

        def forward(s, x):
            e = s.emb_enc(x[..., :64]).reshape(len(x), -1)
            entf = x[..., 64:].reshape(len(x), -1)
            return s.ent_mlp(entf).squeeze(-1) + s.corr(e).squeeze(-1)

    class RESV2(nn.Module):
        """持续性残差: ŷ = 最近观测熵(z) + g(表征, 熵史, mask), g 零初始化"""

        def __init__(s):
            super().__init__()
            s.emb_enc = nn.Sequential(nn.Linear(64, H), nn.GELU(),
                                      nn.Linear(H, 32))
            s.fuse = nn.Sequential(nn.Linear(K * (32 + F_E), H), nn.GELU(),
                                   nn.Dropout(DO), nn.Linear(H, 1))
            nn.init.zeros_(s.fuse[-1].weight)
            nn.init.zeros_(s.fuse[-1].bias)

        def forward(s, x):
            ent = x[..., 64:65]
            msk = x[..., 65:66]
            e = s.emb_enc(x[..., :64]).reshape(len(x), -1)
            parts = [e, ent.squeeze(-1), msk.squeeze(-1)]
            if F_E > 2:
                parts.append(x[..., 66:].squeeze(-1))
            corr = s.fuse(torch.cat(parts, dim=-1)).squeeze(-1)
            w = msk.squeeze(-1) * torch.arange(1, K + 1, device=x.device)
            idx = w.argmax(1, keepdim=True)
            base = ent.squeeze(-1).gather(1, idx).squeeze(1)
            return base + corr

    class MLP(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(
                nn.Linear(K * F, H * 2), nn.GELU(), nn.Dropout(DO),
                nn.Linear(H * 2, H), nn.GELU(), nn.Dropout(DO),
                nn.Linear(H, 1))

        def forward(s, x):
            return s.net(x.reshape(len(x), -1)).squeeze(-1)

    class GRU(nn.Module):
        def __init__(s):
            super().__init__()
            s.gru = nn.GRU(F, H, layers, batch_first=True,
                           dropout=DO if layers > 1 else 0.0)
            s.head = nn.Sequential(nn.Dropout(DO), nn.Linear(H, 1))

        def forward(s, x):
            h, _ = s.gru(x)
            return s.head(h[:, -1]).squeeze(-1)

    class TFM(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = nn.Linear(F, 64)
            s.pos = nn.Parameter(torch.randn(1, K, 64) * 0.02)
            layer = nn.TransformerEncoderLayer(
                64, 4, H, dropout=DO, batch_first=True, norm_first=True)
            s.enc = nn.TransformerEncoder(layer, layers)
            s.head = nn.Linear(64, 1)

        def forward(s, x):
            h = s.enc(s.inp(x) + s.pos)
            return s.head(h.mean(1)).squeeze(-1)

    return {"res2v2": RES2V2, "resv2": RESV2, "mlp": MLP, "gru": GRU,
            "transformer": TFM}[model]()


# ---------------------------------------------------------------- 评估
def pooled_metrics(pred, y):
    pred = np.clip(pred, 0, None)
    ss_res = float(((pred - y) ** 2).sum())
    return dict(r2=round(1 - ss_res / float(((y - y.mean()) ** 2).sum()), 4),
                mae=round(float(np.abs(pred - y).mean()), 4),
                rmse=round(float(np.sqrt(((pred - y) ** 2).mean())), 4),
                mse=round(float(((pred - y) ** 2).mean()), 4))


def country_r2_list(pred, y, cs):
    out = []
    for c in np.unique(cs):
        m = cs == c
        if m.sum() < 2:
            continue
        ss_res = float(((np.clip(pred[m], 0, None) - y[m]) ** 2).sum())
        ss_tot = float(((y[m] - y[m].mean()) ** 2).sum())
        out.append((int(c), int(m.sum()),
                    1 - ss_res / ss_tot if ss_tot > 1e-9 else None))
    return out


def summarize(pred, y, cs):
    r = pooled_metrics(pred, y)
    r2s = [x[2] for x in country_r2_list(pred, y, cs) if x[2] is not None]
    r["per_country_r2_median"] = round(float(np.median(r2s)), 4) if r2s else None
    r["n_countries_eval"] = len(r2s)
    r["n_countries_r2_ge_08"] = int(sum(1 for v in r2s if v >= 0.8))
    r["frac_countries_r2_ge_08"] = \
        round(sum(1 for v in r2s if v >= 0.8) / len(r2s), 4) if r2s else None
    r["n"] = int(len(y))
    return r


# ---------------------------------------------------------------- 校准
def fit_calibration(pred_tr_raw, y_tr, cs_tr, countries, lam):
    """逐国 y≈a+b*pred, 岭收缩向 (0,1); <5 点 -> (0,1)"""
    cal = {}
    for ci, c in enumerate(countries):
        m = cs_tr == ci
        if m.sum() < 5:
            cal[c] = (0.0, 1.0)
            continue
        p = pred_tr_raw[m].astype(np.float64)
        t = y_tr[m].astype(np.float64)
        A = np.array([[len(p), p.sum()], [p.sum(), (p * p).sum()]])
        A[0, 0] += lam
        A[1, 1] += lam
        bv = np.array([t.sum(), (p * t).sum() + lam])
        a_, b_ = np.linalg.solve(A, bv)
        cal[c] = (float(a_), float(b_))
    return cal


def apply_calibration(pred_raw, cs, countries, cal):
    out = pred_raw.astype(np.float64).copy()
    for ci, c in enumerate(countries):
        m = cs == ci
        if m.any():
            a_, b_ = cal.get(c, (0.0, 1.0))
            out[m] = a_ + b_ * out[m]
    return np.clip(out, 0, None)


# ---------------------------------------------------------------- 滚动在线校准
def _affine_fit(p, t, lam):
    """min ||t - a - b p||^2 + lam*(a^2 + (b-1)^2) 闭式解"""
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    A = np.array([[len(p), p.sum()], [p.sum(), (p * p).sum()]])
    A[0, 0] += lam
    A[1, 1] += lam
    bv = np.array([t.sum(), (p * t).sum() + lam])
    return np.linalg.solve(A, bv)


def rolling_calibrate(pred, y, cs, ts, countries, lam=10.0, win=6, min_pts=3):
    """滚动在线校准 (严格因果, 无前视): 目标月 m 的逐国仿射 y≈a+b*pred 仅用
    该国 <m 的最近 win 个 (pred, y) 样本拟合; 点数<min_pts 回退到全体国家
    训练段(<=TRAIN_END)静态仿射。验证段内较早月份的 (pred, true) 对随时间
    滚动纳入 (在线学习, 合法)。"""
    tr = ts <= TRAIN_END
    a_g, b_g = _affine_fit(pred[tr], y[tr], lam)
    out = np.empty(len(pred), dtype=np.float64)
    for ci in range(len(countries)):
        idx = np.nonzero(cs == ci)[0]
        idx = idx[np.argsort(ts[idx])]          # YYYY-MM 字典序即时间序
        for r, i in enumerate(idx):
            hist = idx[max(0, r - win):r]
            if len(hist) >= min_pts:
                a_, b_ = _affine_fit(pred[hist], y[hist], lam)
            else:
                a_, b_ = a_g, b_g
            out[i] = a_ + b_ * pred[i]
    return np.clip(out, 0, None)


def filtered_country_set(D, counts, thr=10.0):
    """验证月平均序列数 >= thr 的国家索引集合 (counts: {country:{month:n}})"""
    ok = set()
    if counts is None:
        return set(range(len(D["countries"])))
    cs, ts, va = D["cs"], D["ts"], D["va"]
    for ci, c in enumerate(D["countries"]):
        ns = [counts.get(c, {}).get(m, 0)
              for m in ts[va][cs[va] == ci]]
        if ns and float(np.mean(ns)) >= thr:
            ok.add(ci)
    return ok


def summarize_on(pred, y, cs, keep):
    """仅在 keep 国家集合的样本上汇总 (pooled + 逐国)"""
    m = np.isin(cs, list(keep))
    return summarize(pred[m], y[m], cs[m])


# ---------------------------------------------------------------- 训练
def train_one(a):
    """a: argparse 命名空间; 返回 result dict 并保存 ckpt/metrics"""
    name = a.name or (
        f"v2_{a.model}_{a.mode}_{a.target_norm}"
        f"{'_cal' if a.calibrate else ''}_K{a.K}_h{a.hidden}_L{a.layers}"
        f"_do{a.dropout}_ed{a.ent_dropout}_hl{a.half_life}"
        f"{'_bal' if a.balanced_sampler else ''}{'_d' if a.delta else ''}"
        f"_s{a.seed}")
    runs = getattr(a, "runs", None) or RUNS
    os.makedirs(runs, exist_ok=True)
    t0 = time.time()

    D = prepare_data(a.K, a.target_norm, a.delta, a.mode,
                     getattr(a, "ent_dir", None),
                     getattr(a, "counts_file", None))
    Xf, yz, y, cs, countries = D["Xf"], D["yz"], D["y"], D["cs"], D["countries"]
    tr, va, norm = D["tr"], D["va"], D["norm"]
    ts = D["ts"]
    print(f"[{name}] N={len(y)} train={tr.sum()} val={va.sum()} "
          f"countries={len(countries)}", flush=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    itr = np.nonzero(tr)[0]
    rng.shuffle(itr)
    n_hold = max(1, int(0.1 * len(itr)))
    ihold, ifit = itr[:n_hold], itr[n_hold:]

    X_t = torch.from_numpy(Xf).to(device)
    yz_t = torch.from_numpy(yz).to(device)
    ih_t = torch.from_numpy(ihold).to(device)

    if a.half_life > 0:
        age = np.array([MIDX["2022-07"] - MIDX[t] for t in ts[ifit]],
                       dtype=np.float64)
        sw_np = 0.5 ** (age / a.half_life)
    else:
        sw_np = np.ones(len(ifit), dtype=np.float64)
    # 序列数精度加权 (小样本噪声目标降权, 均值归一)
    sw_mode = getattr(a, "seq_weight", "none")
    if sw_mode != "none" and D.get("n_seq") is not None:
        n = D["n_seq"][ifit]
        if sw_mode == "n50":
            w2 = n / (n + 50.0)
        elif sw_mode == "sqrt":
            w2 = np.sqrt(n)
        elif sw_mode == "log":
            w2 = np.log1p(n)
        else:
            w2 = np.ones_like(n)
        sw_np = sw_np * (w2 / max(w2.mean(), 1e-12))
    sw = torch.from_numpy(sw_np.astype(np.float32)).to(device)

    net = make_model(a.model, a.K, a.hidden, a.layers, a.dropout,
                     D["F"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.SmoothL1Loss(beta=0.5, reduction="none")

    if a.balanced_sampler:
        cnt = np.bincount(cs[ifit], minlength=len(countries)).astype(np.float64)
        pw = torch.from_numpy(1.0 / cnt[cs[ifit]]).to(device)

    best = (1e9, None)
    bad = 0
    for ep in range(a.epochs):
        net.train()
        if a.balanced_sampler:
            order = torch.multinomial(pw, len(ifit), replacement=True)
        else:
            order = torch.randperm(len(ifit), device=device)
        for i in range(0, len(order), a.batch):
            sel = order[i:i + a.batch]
            idx = torch.from_numpy(ifit[sel.cpu().numpy()]).to(device)
            xb = X_t[idx]
            if a.ent_dropout > 0:
                drop = torch.rand(len(idx), device=device) < a.ent_dropout
                xb = xb.clone()
                xb[drop, :, 64:] = 0.0
            l = lossf(net(xb), yz_t[idx])
            w = sw[sel.to(device)] if not a.balanced_sampler else sw[sel]
            loss = (l * w).sum() / w.sum().clamp_min(1e-8)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            hl = lossf(net(X_t[ih_t]), yz_t[ih_t]).mean().item()
        if hl < best[0] - 1e-5:
            best = (hl, {k: v.detach().clone()
                         for k, v in net.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= 30:
                break
    net.load_state_dict(best[1])
    net.eval()

    with torch.no_grad():
        pz_all = net(X_t).cpu().numpy().astype(np.float64)
    pred_raw = Norm.inverse(pz_all, D["mu"], D["sd"])

    result = dict(name=name, model=a.model, mode=a.mode,
                  target_norm=a.target_norm, calibrate=bool(a.calibrate),
                  cal_lambda=a.cal_lambda, half_life=a.half_life,
                  balanced_sampler=bool(a.balanced_sampler),
                  delta=bool(a.delta), ent_dropout=a.ent_dropout,
                  K=a.K, hidden=a.hidden, layers=a.layers, dropout=a.dropout,
                  lr=a.lr, wd=a.wd, seed=a.seed,
                  seq_weight=getattr(a, "seq_weight", "none"),
                  n_train=int(tr.sum()), n_val=int(va.sum()),
                  countries=len(countries), hold_loss=round(best[0], 5),
                  epochs_run=ep + 1, ent_dir=D["ent_dir"])
    result["val"] = summarize(pred_raw[va], y[va], cs[va])
    print("[val]", json.dumps(result["val"], ensure_ascii=False), flush=True)

    cal = None
    if a.calibrate:
        cal = fit_calibration(pred_raw[tr], y[tr], cs[tr], countries,
                              a.cal_lambda)
        pred_cal = apply_calibration(pred_raw[va], cs[va], countries, cal)
        result["val_calibrated"] = summarize(pred_cal, y[va], cs[va])
        print("[val_calibrated]",
              json.dumps(result["val_calibrated"], ensure_ascii=False),
              flush=True)

    torch.save(dict(state_dict=net.state_dict(), model=a.model, mode=a.mode,
                    target_norm=a.target_norm, delta=bool(a.delta),
                    K=a.K, hidden=a.hidden, layers=a.layers,
                    dropout=a.dropout, F=D["F"], norm=norm.state(),
                    calibration=cal, countries=countries,
                    seq_weight=getattr(a, "seq_weight", "none"),
                    ent_dir=D["ent_dir"]),
               os.path.join(runs, f"{name}.pt"))
    result["train_sec"] = round(time.time() - t0, 1)
    json.dump(result, open(os.path.join(runs, f"{name}_metrics.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", os.path.join(runs, f"{name}_metrics.json"), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["res2v2", "resv2", "mlp", "gru",
                                        "transformer"], required=True)
    ap.add_argument("--mode", choices=["full", "ent_only"], default="full")
    ap.add_argument("--target_norm", choices=["global", "country", "window"],
                    default="global")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--cal_lambda", type=float, default=10.0)
    ap.add_argument("--half_life", type=float, default=0.0)
    ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--delta", action="store_true")
    ap.add_argument("--ent_dropout", type=float, default=0.0)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--name", default=None)
    ap.add_argument("--ent_dir", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--counts_file", default=None)
    ap.add_argument("--seq_weight", choices=["none", "n50", "sqrt", "log"],
                    default="none")
    train_one(ap.parse_args())


if __name__ == "__main__":
    main()
