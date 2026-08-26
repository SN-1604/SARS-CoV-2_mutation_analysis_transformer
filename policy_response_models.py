# -*- coding: utf-8 -*-
"""policy_response_models.py — 政策→病例/死亡比例 辅助响应模型
目标(周频逐国): log1p(new_cases_smoothed_per_million) / log1p(new_deaths_smoothed_per_million)
  (per-million log 标度; 早期版本按人口比例的 ~1e-7 量级导致迭代模型欠拟合)
特征(严格因果): 政策杆当期+滞后2周(覆盖率/接种速度/管控) + 病例/死亡 log 比各 1-4 周滞后
  + 4 周拖尾均值(t-1..t-4) + 时间趋势 + 国家 one-hot
模型: Ridge vs HistGBR(平方损失, depth8/lr0.05/iter1200), 训练 ≤2022-07, 验证 ≥2022-08, 选优。
反事实施加 (差分法, 锚定观测轨迹防漂移):
  y_cf[t] = y_obs[t] + g(X_cf[t]) − g(X_obs[t])   (自回归特征取观测)
  流量列按 expm1 比率缩放, total 列按观测+累计差分(下限0); 无政策扰动时恒等。
输出: policy_response_models.pkl, policy_response_val_pred.npz
"""
import datetime as dt
import pickle

import numpy as np

PANEL_PKL = "owid_weekly_panel.pkl"
OUT_PKL = "policy_response_models.pkl"
VAL_NPZ = "policy_response_val_pred.npz"
W0 = dt.date(2020, 3, 2)
TRAIN_END = dt.date(2022, 7, 31)
VAL_START = dt.date(2022, 8, 1)

POLICY = ["people_vaccinated_per_hundred",
          "new_people_vaccinated_smoothed_per_hundred",
          "stringency_index"]
CASES_FLOW = ["new_cases", "new_cases_smoothed", "new_cases_per_million",
              "new_cases_smoothed_per_million"]
DEATHS_FLOW = ["new_deaths", "new_deaths_smoothed", "new_deaths_per_million",
               "new_deaths_smoothed_per_million"]
CASE_TARGET_COL = "new_cases_smoothed_per_million"
DEATH_TARGET_COL = "new_deaths_smoothed_per_million"


def load_panel():
    panel = pickle.load(open(PANEL_PKL, "rb"))
    countries = sorted(panel.keys())
    cols = list(panel[countries[0]].columns)
    raw = np.stack([np.asarray(panel[c][cols], dtype=np.float64)
                    for c in countries])
    return countries, cols, raw


def _lag(a, lag):
    r = np.roll(a, lag, axis=1)
    r[:, :lag] = a[:, :1]
    return r


def build_features(raw, cols, countries, policy_override=None):
    """(NC,148,F) 特征 + (NC,148,2) 目标。policy_override: (NC,148,3) 反事实政策列。
    自回归/趋势/国家特征不随政策变化 (差分法中保持观测)。"""
    NC = len(countries)
    yc = np.log1p(raw[:, :, cols.index(CASE_TARGET_COL)])
    yd = np.log1p(raw[:, :, cols.index(DEATH_TARGET_COL)])
    pol = (np.stack([raw[:, :, cols.index(p)] for p in POLICY], axis=-1)
           if policy_override is None else policy_override)
    feats = []
    for lag in range(3):                       # 政策当期,t-1,t-2
        feats.append(_lag(pol, lag))
    for lag in (1, 2, 3, 4):                   # 病例/死亡 log比 各4周滞后
        feats.append(_lag(yc, lag)[..., None])
        feats.append(_lag(yd, lag)[..., None])
    tm_c = (_lag(yc, 1) + _lag(yc, 2) + _lag(yc, 3) + _lag(yc, 4)) / 4
    tm_d = (_lag(yd, 1) + _lag(yd, 2) + _lag(yd, 3) + _lag(yd, 4)) / 4
    feats.append(tm_c[..., None])
    feats.append(tm_d[..., None])
    feats.append((np.arange(148) / 148.0)[None, :, None].repeat(NC, 0))
    feats.append(np.eye(NC)[:, None, :].repeat(148, 1))     # 国家 one-hot
    X = np.concatenate(feats, axis=-1)
    Y = np.stack([yc, yd], axis=-1)
    return X, Y


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    countries, cols, raw = load_panel()
    X, Y = build_features(raw, cols, countries)
    weeks = [W0 + dt.timedelta(weeks=i) for i in range(148)]
    tr_mask = np.array([w <= TRAIN_END for w in weeks])
    tr_mask[:4] = False                        # 需要4周滞后
    va_mask = np.array([w >= VAL_START for w in weeks])

    def r2(p, t):
        return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())

    models = {}
    val_pred = {}
    for ti, tname in enumerate(["cases", "deaths"]):
        Xf = X[:, tr_mask].reshape(-1, X.shape[-1])
        yf = Y[:, tr_mask, ti].reshape(-1)
        Xv = X[:, va_mask].reshape(-1, X.shape[-1])
        yv = Y[:, va_mask, ti].reshape(-1)
        cands = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xf, yf),
            "gbr_sq": HistGradientBoostingRegressor(
                max_depth=8, learning_rate=0.05, max_iter=1200,
                early_stopping=True, random_state=7).fit(Xf, yf),
        }
        res = {nm: r2(m.predict(Xv), yv) for nm, m in cands.items()}
        best = max(res, key=res.get)
        models[tname] = dict(kind="sk", obj=cands[best], name=best)
        val_pred[tname] = (cands[best].predict(Xv), yv)
        print(f"{tname}: " + " ".join(f"{k} R2={v:.4f}" for k, v in res.items())
              + f" -> pick {best}", flush=True)

    spec = dict(policy_cols=POLICY, cases_flow=CASES_FLOW, deaths_flow=DEATHS_FLOW,
                targets=[CASE_TARGET_COL, DEATH_TARGET_COL],
                note="delta method in log1p(per-million) space: y_cf = y_obs + g(X_cf)-g(X_obs)")
    pickle.dump(dict(models=models, spec=spec, countries=countries, cols=cols),
                open(OUT_PKL, "wb"))
    np.savez_compressed(VAL_NPZ,
                        cases_pred=val_pred["cases"][0], cases_true=val_pred["cases"][1],
                        deaths_pred=val_pred["deaths"][0], deaths_true=val_pred["deaths"][1])
    print("saved", OUT_PKL, "and", VAL_NPZ)


def pr_predict(entry, X):
    return entry["obj"].predict(X)


def counterfactual_panel(raw, M, cols, models, countries):
    """在已缩放/已施加约束的面板 M 上, 用辅助模型合成病例/死亡反事实列 (就地改写并返回)"""
    NC = raw.shape[0]
    pol_cf = np.stack([M[:, :, cols.index(p)] for p in POLICY], axis=-1)
    X_obs, Y = build_features(raw, cols, countries)
    X_cf, _ = build_features(raw, cols, countries, policy_override=pol_cf)
    F = X_obs.shape[-1]
    for ti, tname in enumerate(["cases", "deaths"]):
        g = models[tname]
        d = (pr_predict(g, X_cf.reshape(-1, F))
             - pr_predict(g, X_obs.reshape(-1, F))).reshape(NC, 148)
        tcol = CASE_TARGET_COL if tname == "cases" else DEATH_TARGET_COL
        obs = raw[:, :, cols.index(tcol)]
        ratio = np.expm1(np.log1p(obs) + d) / np.maximum(obs, 1e-9)
        ratio = np.clip(ratio, 0, 100.0)       # 防极端外推
        flows = CASES_FLOW if tname == "cases" else DEATHS_FLOW
        for col in flows:
            M[:, :, cols.index(col)] = raw[:, :, cols.index(col)] * ratio
        newcol = "new_cases_smoothed" if tname == "cases" else "new_deaths_smoothed"
        totcol = "total_cases" if tname == "cases" else "total_deaths"
        new_cf = raw[:, :, cols.index(newcol)] * ratio
        cumdelta = np.cumsum(new_cf - raw[:, :, cols.index(newcol)], axis=1)
        M[:, :, cols.index(totcol)] = np.maximum(
            raw[:, :, cols.index(totcol)] + cumdelta, 0.0)
    return M


if __name__ == "__main__":
    main()
