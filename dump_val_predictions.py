# -*- coding: utf-8 -*-
"""dump_val_predictions.py — 用最终集成模型对全部验证样本推理, 输出逐月预测
输入: entropy_forecast_monthly_best.pt (3 种子 RES 集成, K=3)
输出: val_pred_monthly.npz  (countries[N], months[N], y_true[N], y_pred[N])
运行: G:/Anaconda3/envs/esm/python.exe dump_val_predictions.py
"""
import numpy as np
import torch
from torch import nn

from train_entropy_forecast_monthly import build_samples, TRAIN_END

CKPT = "entropy_forecast_monthly_best.pt"
OUT = "val_pred_monthly.npz"


class RES(nn.Module):
    """与 train_entropy_forecast_monthly.py 中 RES 相同 (hidden=128, K=3)"""

    def __init__(s, K, hidden):
        super().__init__()
        s.emb_enc = nn.Sequential(nn.Linear(64, hidden), nn.GELU(),
                                  nn.Linear(hidden, 32))
        s.fuse = nn.Sequential(nn.Linear(K * 34, hidden), nn.GELU(),
                               nn.Dropout(0.1), nn.Linear(hidden, 1))

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
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    K, hidden = ck["K"], ck["hidden"]
    mu, sd = ck["y_mu"], ck["y_sd"]
    sds = ck["ensemble_state_dicts"]
    print(f"ckpt: arch={ck['arch']} K={K} hidden={hidden} seeds={len(sds)}")

    Xe, Xh, Xm, y, cs, ts, countries = build_samples(K)
    va = ts > TRAIN_END
    Xh = ((np.log1p(Xh) - mu) / sd * Xm).astype(np.float32)
    Xflat = np.concatenate([Xe, Xh[..., None], Xm[..., None]], axis=2)
    X_t = torch.from_numpy(Xflat[va])

    preds = []
    with torch.no_grad():
        for state in sds:
            net = RES(K, hidden)
            net.load_state_dict(state)
            net.eval()
            preds.append(net(X_t).numpy())
    pred = np.expm1(np.mean(preds, axis=0) * sd + mu)

    cnames = np.array([countries[i] for i in cs[va]])
    np.savez(OUT, countries=cnames, months=ts[va],
             y_true=y[va], y_pred=pred)
    ss_res = float(((np.clip(pred, 0, None) - y[va]) ** 2).sum())
    ss_tot = float(((y[va] - y[va].mean()) ** 2).sum())
    print(f"saved {OUT}: n={va.sum()} countries={len(np.unique(cnames))} "
          f"R2={1 - ss_res / ss_tot:.4f}")


if __name__ == "__main__":
    main()
