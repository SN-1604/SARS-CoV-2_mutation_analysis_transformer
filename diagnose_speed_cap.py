# -*- coding: utf-8 -*-
"""diagnose_speed_cap.py — 接种速度上限处理方式的通道分解诊断 (存档证据)
背景: 旧实现用 np.minimum(扰动值, cap) 做值截断, 把观测峰值已超 1%/日的国家-周
向下拉低, 经覆盖率联动使"+10% 速度"反而降低覆盖率 -> fig2 速度因素符号反转。
本脚本在 δ=±10% 分解各通道, 对比 旧值截断 / 新增量钳制 / 不截断:
  A_speed_only      仅速度列扰动 (增量钳制, 无联动无病例反事实)
  D_speed_cf        速度列 + 病例/死亡反事实 (无联动)
  B_cov_coupled     仅联动引起的覆盖率变化 (速度列保持观测)
  C_full_clamp      完整新实现 (增量钳制+联动+反事实) == factor_sensitivity 当前结果
  C_full_oldclip    完整旧实现 (值截断+联动+反事实)     == 旧 factor_effects 结果
  C_full_nocap      完整但不施加任何速度上限
输出: speed_cap_diagnostic.json
"""
import datetime as dt
import json, pickle

import numpy as np
import torch

from owid_repr13 import make_model
from finalize_monthly_forecast import RES
from train_entropy_forecast_monthly import build_samples, MONTHS, MIDX
from policy_response_models import counterfactual_panel
from factor_sensitivity import (GROUPS, POLICY_FLAT, apply_speed_to_coverage,
                                clamp_speed_increase,
                                build_counterfactual_panel)

PANEL_PKL = "owid_weekly_panel.pkl"
REPR_CKPT = "repr13_best.pt"
FORE_CKPT = "entropy_forecast_monthly_best.pt"
TRAIN_END = "2022-07"
K = 3


def month_of(d):
    return f"{d.year:04d}-{d.month:02d}"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPR_CKPT, map_location="cpu", weights_only=False)
    cols = [str(c) for c in ck["cols"]]
    mu = ck["mu"].astype(np.float64)
    sd = ck["sd"].astype(np.float64)
    log_mask = ck["log_mask"].copy()
    W = int(ck["W"])
    countries_repr = sorted(str(c) for c in ck["countries"])
    cfg = {k: v for k, v in ck["cfg"].items() if k != "epochs"}
    repr_model = make_model(len(countries_repr), **cfg).to(device)
    repr_model.load_state_dict(ck["state_dict"])
    repr_model.eval()

    fck = torch.load(FORE_CKPT, map_location="cpu", weights_only=False)
    fore = []
    for st in fck["ensemble_state_dicts"]:
        net = RES().to(device)
        net.load_state_dict(st)
        net.eval()
        fore.append(net)
    y_mu, y_sd = fck["y_mu"], fck["y_sd"]
    pr_models = pickle.load(open("policy_response_models.pkl", "rb"))["models"]

    panel = pickle.load(open(PANEL_PKL, "rb"))
    raw = np.stack([np.asarray(panel[c][cols], dtype=np.float64)
                    for c in countries_repr])
    w0 = dt.date(2020, 3, 2)
    week_month = [month_of(w0 + dt.timedelta(weeks=i)) for i in range(148)]
    month_of_year = sorted(set(week_month))
    midx_panel = {m: i for i, m in enumerate(month_of_year)}
    NC = len(countries_repr)

    Xe, Xh, Xm, y, cs, ts, countries = build_samples(K)
    val = np.nonzero(ts > TRAIN_END)[0]
    use_ci = np.array([countries_repr.index(countries[cs[i]]) for i in val])
    POP = raw[:, 0, cols.index("population")]
    spd_idx = [cols.index(c) for c in GROUPS["vacc_speed"]]
    CAP_PH = cols.index("new_people_vaccinated_smoothed_per_hundred")
    CAP_PM = cols.index("new_vaccinations_smoothed_per_million")

    def forward_from_M(M):
        M[:, :, log_mask] = np.log1p(M[:, :, log_mask])
        Z = ((M - mu) / sd).astype(np.float32)
        X = np.stack([Z[:, t - W + 1: t + 1, :]
                      for t in range(W - 1, 148)], axis=1)
        X = X.reshape(NC * 136, W, 38)
        E = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                E.append(repr_model(torch.from_numpy(X[i:i + 8192])
                                    .to(device))["emb"].cpu().numpy())
        E = np.concatenate(E).reshape(NC, 136, 64)
        Em = np.zeros((NC, len(month_of_year), 64), np.float32)
        Cm = np.zeros((NC, len(month_of_year), 1), np.float32)
        for t in range(W - 1, 148):
            mi = midx_panel[week_month[t]]
            Em[:, mi] += E[:, t - W + 1]
            Cm[:, mi] += 1
        Em /= np.maximum(Cm, 1)
        out = np.zeros((len(val), K, 64), np.float32)
        for n, si in enumerate(val):
            i = MIDX[ts[si]]
            jj = 0
            for j in range(max(0, i - K), i):
                m = MONTHS[j]
                if m in midx_panel:
                    out[n, jj] = Em[use_ci[n], midx_panel[m]]
                jj += 1
        return out

    def predict(Em):
        xh = ((np.log1p(Xh[val]) - y_mu) / y_sd * Xm[val]).astype(np.float32)
        xm = Xm[val][:, :, None].astype(np.float32)
        Xb = np.concatenate([Em, xh[:, :, None], xm], axis=2)
        with torch.no_grad():
            pz = torch.stack([net(torch.from_numpy(Xb).to(device))
                              for net in fore]).mean(0).cpu().numpy()
        return np.clip(np.expm1(pz * y_sd + y_mu), 0, None)

    def old_value_clip(M, d):
        """旧实现: np.minimum(扰动值, cap) 值截断"""
        if d == 0:
            return M
        M[:, :, CAP_PH] = np.minimum(M[:, :, CAP_PH], 1.0)
        M[:, :, CAP_PM] = np.minimum(M[:, :, CAP_PM], 1e4)
        for col in ("new_vaccinations_smoothed", "new_people_vaccinated_smoothed"):
            j = cols.index(col)
            M[:, :, j] = np.minimum(M[:, :, j], (0.01 * POP)[:, None])
        return M

    def build_M(d, cap_mode, coupling, cf):
        """cap_mode: 'clamp' (新增量钳制) / 'oldclip' (旧值截断) / 'none'"""
        M = raw.copy()
        M[:, :, spd_idx] = raw[:, :, spd_idx] * (1.0 + d)
        if cap_mode == "clamp":
            sc = np.ones(len(cols))
            sc[spd_idx] = 1.0 + d
            M = clamp_speed_increase(M, raw, sc, cols, POP)
        elif cap_mode == "oldclip":
            M = old_value_clip(M, d)
        if coupling:
            M = apply_speed_to_coverage(raw, M, cols, 1.0)
        if cf:
            M = counterfactual_panel(raw, M, cols, pr_models, countries_repr)
        return M

    y0 = predict(forward_from_M(raw.copy()))
    r2 = 1 - ((y0 - y[val]) ** 2).sum() / ((y[val] - y[val].mean()) ** 2).sum()
    print(f"anchor R2 = {r2:.4f}", flush=True)

    out = {"anchor_r2": float(r2), "note":
           "vacc_speed cap diagnosis: old value-clip pulls observed >cap weeks "
           "down, coupling then lowers coverage (sign flip); increment clamp fixes.",
           "channels": {}}
    for d in (0.1, -0.1):
        sc = np.ones(len(cols))
        sc[spd_idx] = 1.0 + d
        variants = {
            "A_speed_only": build_M(d, "clamp", False, False),
            "D_speed_cf": build_M(d, "clamp", False, True),
            "C_full_clamp": build_counterfactual_panel(
                raw, sc, cols, pr_models, countries_repr),
            "C_full_oldclip": build_M(d, "oldclip", True, True),
            "C_full_nocap": build_M(d, "none", True, True),
        }
        if d > 0:
            Mb = apply_speed_to_coverage(
                raw, build_M(d, "clamp", False, False), cols, 1.0)
            Mb[:, :, spd_idx] = raw[:, :, spd_idx]
            variants["B_cov_coupled"] = Mb
        res = {}
        for name, M in variants.items():
            yp = predict(forward_from_M(M.copy()))
            rel = (yp - y0) / np.maximum(y0, 1e-6)
            res[name] = dict(mean_rel=float(rel.mean()),
                             sign_share=float((np.sign(yp - y0)
                                               == np.sign(d)).mean()))
            print(f"d={d:+.1f} {name:<16} mean_rel={rel.mean():+.5f}",
                  flush=True)
        out["channels"][f"{d:+.1f}"] = res
    json.dump(out, open("speed_cap_diagnostic.json", "w"),
              ensure_ascii=False, indent=1)
    print("saved speed_cap_diagnostic.json", flush=True)


if __name__ == "__main__":
    main()
