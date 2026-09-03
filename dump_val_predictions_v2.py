# -*- coding: utf-8 -*-
"""dump_val_predictions_v2.py — 用 v2 最终集成模型对全部验证样本推理
输入: entropy_forecast_monthly_v2_best.pt (finalize_monthly_forecast_v2 产物,
      members 结构, 支持跨配置/跨 K/跨归一化成员 + 逐国校准)
输出: val_pred_monthly_v2.npz (countries[N], months[N], y_true[N], y_pred[N])
      (不覆盖 v1 的 val_pred_monthly.npz)
运行: G:/Anaconda3/envs/esm/python.exe dump_val_predictions_v2.py
"""
import json

import numpy as np
import torch

import train_entropy_forecast_v2 as T
from train_entropy_forecast_monthly import TRAIN_END

CKPT = "entropy_forecast_monthly_v2_best.pt"
OUT = "val_pred_monthly_v2.npz"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--counts_file", default=None)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    members = ck["members"]
    print(f"ckpt: {len(members)} member(s), "
          f"seeds/member={len(members[0]['ensemble_state_dicts'])})")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ps, D0 = [], None
    for mem in members:
        D = T.prepare_data(mem["K"], mem["target_norm"], mem["delta"], "full",
                           ent_dir=mem.get("ent_dir"))
        X = torch.from_numpy(D["Xf"]).to(device)
        sub = []
        with torch.no_grad():
            for st in mem["ensemble_state_dicts"]:
                net = T.make_model(mem["model"], mem["K"], mem["hidden"],
                                   mem["layers"], mem["dropout"],
                                   mem["F"]).to(device)
                net.load_state_dict(st)
                net.eval()
                sub.append(net(X).cpu().numpy().astype(np.float64))
        ps.append(T.Norm.inverse(np.mean(sub, axis=0), D["mu"], D["sd"]))
        D0 = D
    pf = np.mean(np.stack(ps), axis=0)

    y, cs, ts, countries = D0["y"], D0["cs"], D0["ts"], D0["countries"]
    va = ts > TRAIN_END
    if ck.get("rolling"):                    # 滚动在线校准 (严格因果)
        pf = T.rolling_calibrate(pf, y, cs, ts, countries,
                                 lam=ck["rolling"].get("lam", 10.0),
                                 win=ck["rolling"].get("win", 6))
    else:
        cal = {k: tuple(v) for k, v in (ck.get("calibration") or {}).items()}
        if cal:
            pf = T.apply_calibration(pf, cs, countries, cal)

    va_idx = np.nonzero(va)[0]
    cnames = np.array([countries[i] for i in cs[va_idx]])
    np.savez(a.out, countries=cnames, months=ts[va_idx],
             y_true=y[va_idx], y_pred=pf[va_idx])

    m = T.summarize(pf[va_idx], y[va_idx], cs[va_idx])
    print(f"saved {a.out}: n={len(va_idx)} countries={len(np.unique(cnames))}")
    print(json.dumps(m, ensure_ascii=False, indent=1))
    if a.counts_file:
        counts = T.load_counts(a.counts_file)
        thr = ck.get("exclude_mean_seq_below", 10.0)
        D0v = dict(cs=cs[va_idx], ts=ts[va_idx],
                   va=np.ones(len(va_idx), bool), countries=countries)
        keep = T.filtered_country_set(D0v, counts, thr=thr)
        mf = T.summarize_on(pf[va_idx], y[va_idx], cs[va_idx], keep)
        tgt = int(np.ceil(0.5 * mf["n_countries_eval"]))
        print(f"过滤口径(月均序列>={thr:g}): R2>=0.8 国家 "
              f"{mf['n_countries_r2_ge_08']}/{mf['n_countries_eval']} "
              f"(目标 >= {tgt}), pooled R2={mf['r2']}, "
              f"中位={mf['per_country_r2_median']}")
    else:
        tgt = int(np.ceil(0.75 * m["n_countries_eval"]))
        print(f"验收: R2>=0.8 国家 {m['n_countries_r2_ge_08']}/"
              f"{m['n_countries_eval']} (目标 >= {tgt})")


if __name__ == "__main__":
    main()
