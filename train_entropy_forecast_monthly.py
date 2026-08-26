# -*- coding: utf-8 -*-
"""train_entropy_forecast_monthly.py — 月表征(+历史熵) -> 下一月平滑熵 时间序列预测
输入: 过去 K 个自然月的 64 维月表征 (+ 该月历史熵, --mode 控制)
  --mode full     : 表征 + 历史熵
  --mode ent_only : 仅历史熵   (消融对照)
  --mode emb_only : 仅表征     (消融对照)
表征贡献判据: contribution = (MSE_ent_only - MSE_full) / MSE_ent_only (验证集)
划分: 目标月 ≤2022-07 训练 / ≥2022-08 验证; 目标与历史熵均 log1p+z (训练段统计)
每轮结果写 monthly_forecast_runs/{name}_metrics.json, 权重 {name}.pt
"""
import argparse, json, os, time

import numpy as np

EMB_DIR = "embeddings_monthly"
ENT_DIR = "entropy_monthly_smoothed"
RUNS = "monthly_forecast_runs"
TRAIN_END = "2022-07"
MIN_MONTHS = 15
MONTHS = [f"{y:04d}-{m:02d}" for y in (2020, 2021, 2022)
          for m in range(1, 13) if (y, m) >= (2020, 5) and (y, m) <= (2022, 12)]
MIDX = {m: i for i, m in enumerate(MONTHS)}


def build_samples(K, min_months=MIN_MONTHS):
    """X_emb[N,K,64], X_ent[N,K](原始熵), X_msk[N,K](熵是否观测), y[N], cs, ts, countries"""
    Xe, Xh, Xm, ys, cs, ts = [], [], [], [], [], []
    countries = []
    for f in sorted(os.listdir(EMB_DIR)):
        if not f.endswith("_monthly_embeddings.npz"):
            continue
        c = f[:-len("_monthly_embeddings.npz")]
        ep = os.path.join(ENT_DIR, f"{c}_monthly_entropy.npy")
        if not os.path.exists(ep):
            continue
        ent = {str(k): float(v) for k, v in
               np.load(ep, allow_pickle=True).item().items()}
        if len(ent) < min_months:
            continue
        z = np.load(os.path.join(EMB_DIR, f))
        emb = {str(m): e for m, e in zip(z["months"], z["embeddings"])}
        ci = len(countries)
        countries.append(c)
        for m_tgt, y in ent.items():
            if m_tgt not in MIDX:
                continue
            i = MIDX[m_tgt]
            if i < 1:
                continue
            xe = np.zeros((K, 64), np.float32)
            xh = np.zeros(K, np.float32)
            xm = np.zeros(K, np.float32)
            jj = 0
            for j in range(max(0, i - K), i):
                m = MONTHS[j]
                e = emb.get(m)
                if e is not None:
                    xe[jj] = e
                if m in ent:                    # 历史熵 (原始尺度, 之后统一变换)
                    xh[jj] = ent[m]
                    xm[jj] = 1.0
                jj += 1
            Xe.append(xe); Xh.append(xh); Xm.append(xm)
            ys.append(y); cs.append(ci); ts.append(m_tgt)
    return (np.stack(Xe), np.stack(Xh), np.stack(Xm), np.array(ys),
            np.array(cs), np.array(ts), countries)


def metrics(pred, y):
    pred = np.clip(pred, 0, None)
    ss_res = float(((pred - y) ** 2).sum())
    return dict(r2=round(1 - ss_res / float(((y - y.mean()) ** 2).sum()), 4),
                mae=round(float(np.abs(pred - y).mean()), 4),
                rmse=round(float(np.sqrt(((pred - y) ** 2).mean())), 4),
                mse=round(float(((pred - y) ** 2).mean()), 4))


def per_country_r2(pred, y, cs):
    out = []
    for c in np.unique(cs):
        m = cs == c
        if m.sum() < 2:
            continue
        ss_res = ((pred[m] - y[m]) ** 2).sum()
        ss_tot = ((y[m] - y[m].mean()) ** 2).sum()
        if ss_tot > 1e-9:
            out.append(1 - ss_res / ss_tot)
    return round(float(np.median(out)), 4) if out else None


def evaluate(tag, pred, y, cs):
    r = metrics(pred, y)
    r["per_country_r2_median"] = per_country_r2(pred, y, cs)
    r["n"] = int(len(y))
    print(f"[{tag}]", json.dumps(r, ensure_ascii=False), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ridge", "mlp", "gru", "transformer",
                                        "residual", "dual", "res2", "twostage"],
                    required=True)
    ap.add_argument("--ent_dropout", type=float, default=0.0)
    ap.add_argument("--delta", action="store_true")
    ap.add_argument("--emb_dir", default=None)
    ap.add_argument("--mode", choices=["full", "ent_only", "emb_only"],
                    default="full")
    ap.add_argument("--name", default=None)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    name = a.name or f"{a.model}_{a.mode}_K{a.K}_h{a.hidden}_L{a.layers}_lr{a.lr}_wd{a.wd}_do{a.dropout}_s{a.seed}"
    os.makedirs(RUNS, exist_ok=True)

    if a.emb_dir:                      # 覆盖表征目录 (如 embeddings_monthly_exp9)
        global EMB_DIR
        EMB_DIR = a.emb_dir
    Xe, Xh, Xm, y, cs, ts, countries = build_samples(a.K)
    tr = ts <= TRAIN_END
    va = ~tr
    print(f"samples N={len(y)} train={tr.sum()} val={va.sum()} "
          f"countries={len(countries)} mode={a.mode}", flush=True)
    # 目标与历史熵统一 log1p+z (训练段统计)
    mu, sd = np.log1p(y[tr]).mean(), np.log1p(y[tr]).std() + 1e-8
    yz = ((np.log1p(y) - mu) / sd).astype(np.float32)
    Xh = ((np.log1p(Xh) - mu) / sd * Xm).astype(np.float32)   # 缺测保持 0
    if a.mode == "emb_only":
        Xh[:] = 0.0; Xm[:] = 0.0
    if a.mode == "ent_only":
        Xe[:] = 0.0

    result = dict(name=name, model=a.model, mode=a.mode, K=a.K,
                  n_train=int(tr.sum()), n_val=int(va.sum()),
                  countries=len(countries))
    # 基线 (参考): 持续性 = 最近一个观测月真实熵
    base = []
    for c, m in zip(cs[va], ts[va]):
        i = MIDX[m]
        v = 0.0
        for j in range(i - 1, -1, -1):
            ep = os.path.join(ENT_DIR, f"{countries[c]}_monthly_entropy.npy")
            ent = {str(k): float(x) for k, x in
                   np.load(ep, allow_pickle=True).item().items()}
            if MONTHS[j] in ent:
                v = ent[MONTHS[j]]
                break
        base.append(v)
    evaluate("baseline_persistence", np.array(base), y[va], cs[va])

    t0 = time.time()
    D = 64 + 2                       # 每月特征: 表征 + 熵 + 观测标志
    Xflat = np.concatenate([Xe, Xh[..., None], Xm[..., None]], axis=2)

    if a.model == "ridge":
        from sklearn.linear_model import RidgeCV
        mdl = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(
            Xflat[tr].reshape(tr.sum(), -1), yz[tr])
        pred = np.expm1(mdl.predict(Xflat[va].reshape(va.sum(), -1)) * sd + mu)
        result["alpha"] = float(mdl.alpha_)
        result["val"] = evaluate(name, pred, y[va], cs[va])
        result["train_sec"] = round(time.time() - t0, 1)
    else:
        import torch
        from torch import nn
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rng = np.random.default_rng(a.seed)
        itr = np.nonzero(tr)[0]
        rng.shuffle(itr)
        n_hold = max(1, int(0.1 * len(itr)))
        ihold, ifit = itr[:n_hold], itr[n_hold:]
        X_t = torch.from_numpy(Xflat).to(device)
        yz_t = torch.from_numpy(yz).to(device)
        ih_t = torch.from_numpy(ihold).to(device)

        class MLP(nn.Module):
            def __init__(s):
                super().__init__()
                s.net = nn.Sequential(
                    nn.Linear(a.K * D, a.hidden * 2), nn.GELU(), nn.Dropout(a.dropout),
                    nn.Linear(a.hidden * 2, a.hidden), nn.GELU(), nn.Dropout(a.dropout),
                    nn.Linear(a.hidden, 1))

            def forward(s, x):
                return s.net(x.reshape(len(x), -1)).squeeze(-1)

        class GRU(nn.Module):
            def __init__(s):
                super().__init__()
                s.gru = nn.GRU(D, a.hidden, a.layers, batch_first=True,
                               dropout=a.dropout if a.layers > 1 else 0.0)
                s.head = nn.Sequential(nn.Dropout(a.dropout),
                                       nn.Linear(a.hidden, 1))

            def forward(s, x):
                h, _ = s.gru(x)
                return s.head(h[:, -1]).squeeze(-1)

        class TFM(nn.Module):
            def __init__(s):
                super().__init__()
                s.inp = nn.Linear(D, 64)
                s.pos = nn.Parameter(torch.randn(1, a.K, 64) * 0.02)
                layer = nn.TransformerEncoderLayer(
                    64, 4, a.hidden, dropout=a.dropout, batch_first=True,
                    norm_first=True)
                s.enc = nn.TransformerEncoder(layer, a.layers)
                s.head = nn.Linear(64, 1)

            def forward(s, x):
                h = s.enc(s.inp(x) + s.pos)
                return s.head(h.mean(1)).squeeze(-1)

        class RES(nn.Module):
            """残差架构: ŷ = 最近观测熵 + g(表征, 熵史, mask)
            g 输出层零初始化 -> 起步=持续性基线, 训练只在校正有效时偏离基线;
            表征贡献可被隔离: 对比 emb 通道置零的消融。"""

            def __init__(s):
                super().__init__()
                s.emb_enc = nn.Sequential(nn.Linear(64, a.hidden), nn.GELU(),
                                          nn.Linear(a.hidden, 32))
                s.fuse = nn.Sequential(nn.Linear(a.K * 34, a.hidden), nn.GELU(),
                                       nn.Dropout(a.dropout),
                                       nn.Linear(a.hidden, 1))
                nn.init.zeros_(s.fuse[-1].weight)
                nn.init.zeros_(s.fuse[-1].bias)

            def forward(s, x):
                emb, ent, msk = x[..., :64], x[..., 64:65], x[..., 65:66]
                e = s.emb_enc(emb).reshape(len(x), -1)
                corr = s.fuse(torch.cat([e, ent.squeeze(-1), msk.squeeze(-1)],
                                        dim=-1)).squeeze(-1)
                w = msk.squeeze(-1) * torch.arange(1, x.shape[1] + 1,
                                                   device=x.device)
                idx = w.argmax(1, keepdim=True)          # 全 0 -> 0, 该位 ent 本就为 0
                base = ent.squeeze(-1).gather(1, idx).squeeze(1)
                return base + corr

        class DUAL(nn.Module):
            """双塔: 熵史 GRU 塔 + 表征 MLP 塔, 低维融合"""

            def __init__(s):
                super().__init__()
                s.ent_gru = nn.GRU(2, a.hidden // 2, batch_first=True)
                s.emb_mlp = nn.Sequential(nn.Linear(64, a.hidden), nn.GELU(),
                                          nn.Linear(a.hidden, 32))
                s.head = nn.Sequential(nn.Linear(a.hidden // 2 + a.K * 32, a.hidden),
                                       nn.GELU(), nn.Dropout(a.dropout),
                                       nn.Linear(a.hidden, 1))

            def forward(s, x):
                emb, ent, msk = x[..., :64], x[..., 64:65], x[..., 65:66]
                h, _ = s.ent_gru(torch.cat([ent, msk], dim=-1))
                e = s.emb_mlp(emb).reshape(len(x), -1)
                return s.head(torch.cat([h[:, -1], e], dim=-1)).squeeze(-1)

        class RES2(nn.Module):
            """熵主路径(MLP) + 表征零初始化校正路径; 可选表征差分特征
            ŷ = ent_mlp(熵史, mask) + corr(表征[, Δ表征]); corr 零初始化"""

            def __init__(s):
                super().__init__()
                din = 128 if a.delta else 64
                s.ent_mlp = nn.Sequential(nn.Linear(a.K * 2, a.hidden), nn.GELU(),
                                          nn.Dropout(a.dropout),
                                          nn.Linear(a.hidden, a.hidden // 2),
                                          nn.GELU(), nn.Linear(a.hidden // 2, 1))
                s.emb_enc = nn.Sequential(nn.Linear(din, a.hidden), nn.GELU(),
                                          nn.Linear(a.hidden, 32))
                s.corr = nn.Sequential(nn.Linear(a.K * 32, a.hidden), nn.GELU(),
                                       nn.Dropout(a.dropout), nn.Linear(a.hidden, 1))
                nn.init.zeros_(s.corr[-1].weight)
                nn.init.zeros_(s.corr[-1].bias)

            def forward(s, x):
                emb, ent, msk = x[..., :64], x[..., 64:65], x[..., 65:66]
                base = s.ent_mlp(torch.cat([ent, msk], dim=-1)
                                 .reshape(len(x), -1)).squeeze(-1)
                if a.delta:
                    d = torch.diff(emb, dim=1, prepend=emb[:, :1])
                    emb = torch.cat([emb, d], dim=-1)
                e = s.emb_enc(emb).reshape(len(x), -1)
                return base + s.corr(e).squeeze(-1)

        net = {"mlp": MLP, "gru": GRU, "transformer": TFM,
               "residual": RES, "dual": DUAL, "res2": RES2}[a.model]().to(device) \
            if a.model != "twostage" else None

        if a.model == "twostage":
            # 阶段1: 熵路径 MLP (输入 K*2) 早停训练
            ent_mlp = nn.Sequential(
                nn.Linear(a.K * 2, a.hidden * 2), nn.GELU(), nn.Dropout(a.dropout),
                nn.Linear(a.hidden * 2, a.hidden), nn.GELU(), nn.Dropout(a.dropout),
                nn.Linear(a.hidden, 1)).to(device)

            def ent_in(xb):
                return xb[:, :, 64:].reshape(len(xb), -1)

            def fit(netm, get_x, get_y, tag):
                opt = torch.optim.AdamW(netm.parameters(), lr=a.lr,
                                        weight_decay=a.wd)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
                lossf = nn.SmoothL1Loss(beta=0.5)
                best = (1e9, None)
                bad = 0
                for ep in range(a.epochs):
                    netm.train()
                    perm = torch.randperm(len(ifit), device=device)
                    for i in range(0, len(perm), a.batch):
                        idx = torch.from_numpy(
                            ifit[perm[i:i + a.batch].cpu().numpy()]).to(device)
                        loss = lossf(netm(get_x(X_t[idx])).squeeze(-1),
                                     get_y(idx))
                        opt.zero_grad(); loss.backward(); opt.step()
                    sch.step()
                    netm.eval()
                    with torch.no_grad():
                        hl = lossf(netm(get_x(X_t[ih_t])).squeeze(-1),
                                   get_y(ih_t)).item()
                    if hl < best[0] - 1e-5:
                        best = (hl, {k: v.detach().clone()
                                     for k, v in netm.state_dict().items()})
                        bad = 0
                    else:
                        bad += 1
                        if bad >= 30:
                            break
                netm.load_state_dict(best[1])
                netm.eval()
                print(f"  [{tag}] hold_loss={best[0]:.5f}", flush=True)
                return netm

            yzt = yz_t
            ent_mlp = fit(ent_mlp, ent_in, lambda idx: yzt[idx], "stage1_ent")
            for p in ent_mlp.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                base_fit = ent_mlp(ent_in(X_t)).squeeze(-1)
            # 阶段2: 表征残差校正路径 (零初始化), 只在熵路径残差上学
            corr = nn.Sequential(
                nn.Linear(64, a.hidden), nn.GELU(), nn.Linear(a.hidden, 32),
                nn.Flatten(), nn.Linear(a.K * 32, a.hidden), nn.GELU(),
                nn.Dropout(a.dropout), nn.Linear(a.hidden, 1)).to(device)
            nn.init.zeros_(corr[-1].weight)
            nn.init.zeros_(corr[-1].bias)
            resid = yz_t - base_fit
            corr = fit(corr, lambda xb: xb[:, :, :64], lambda idx: resid[idx],
                       "stage2_corr")
            with torch.no_grad():
                predz = (base_fit + corr(X_t[:, :, :64]).squeeze(-1))
                predz = predz[torch.from_numpy(np.nonzero(va)[0]).to(device)] \
                    .cpu().numpy()
            pred = np.expm1(predz * sd + mu)
            result["val"] = evaluate(name, pred, y[va], cs[va])
            # 表征贡献: 相对纯熵路径的验证 MSE 下降率
            with torch.no_grad():
                predz_e = base_fit[torch.from_numpy(np.nonzero(va)[0])
                                   .to(device)].cpu().numpy()
            pred_e = np.expm1(predz_e * sd + mu)
            r_e = evaluate(name + "|entpath", pred_e, y[va], cs[va])
            contrib = (r_e["mse"] - result["val"]["mse"]) / r_e["mse"]
            result["val_entpath_only"] = r_e
            result["emb_contribution"] = round(float(contrib), 4)
            result["train_sec"] = round(time.time() - t0, 1)
            torch.save(dict(ent_mlp=ent_mlp.state_dict(), corr=corr.state_dict(),
                            model="twostage",
                            hp=dict(K=a.K, hidden=a.hidden, dropout=a.dropout),
                            y_mu=float(mu), y_sd=float(sd), countries=countries),
                       os.path.join(RUNS, f"{name}.pt"))
            json.dump(result, open(os.path.join(RUNS, f"{name}_metrics.json"), "w"),
                      ensure_ascii=False, indent=1)
            print("saved", os.path.join(RUNS, f"{name}_metrics.json"), flush=True)
            return
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        lossf = nn.SmoothL1Loss(beta=0.5)
        best = (1e9, None)
        bad = 0
        for ep in range(a.epochs):
            net.train()
            perm = torch.randperm(len(ifit), device=device)
            for i in range(0, len(perm), a.batch):
                idx = torch.from_numpy(ifit[perm[i:i + a.batch].cpu().numpy()]).to(device)
                xb = X_t[idx]
                if a.ent_dropout > 0:      # 随机屏蔽熵通道, 迫使表征通路承载信息
                    drop = torch.rand(len(idx), device=device) < a.ent_dropout
                    xb = xb.clone()
                    xb[drop, :, 64:] = 0.0
                loss = lossf(net(xb), yz_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
            net.eval()
            with torch.no_grad():
                hl = lossf(net(X_t[ih_t]), yz_t[ih_t]).item()
            if hl < best[0] - 1e-5:
                best = (hl, {k: v.detach().clone() for k, v in net.state_dict().items()})
                bad = 0
            else:
                bad += 1
                if bad >= 30:
                    break
        net.load_state_dict(best[1])
        net.eval()
        with torch.no_grad():
            predz = net(X_t[torch.from_numpy(np.nonzero(va)[0]).to(device)]).cpu().numpy()
        pred = np.expm1(predz * sd + mu)
        result["val"] = evaluate(name, pred, y[va], cs[va])
        result["train_sec"] = round(time.time() - t0, 1)
        torch.save(dict(state_dict=net.state_dict(), model=a.model, mode=a.mode,
                        hp=dict(K=a.K, hidden=a.hidden, layers=a.layers,
                                dropout=a.dropout, D=D),
                        y_mu=float(mu), y_sd=float(sd), countries=countries),
                   os.path.join(RUNS, f"{name}.pt"))

    json.dump(result, open(os.path.join(RUNS, f"{name}_metrics.json"), "w"),
              ensure_ascii=False, indent=1)
    print("saved", os.path.join(RUNS, f"{name}_metrics.json"), flush=True)


if __name__ == "__main__":
    main()
