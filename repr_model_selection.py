# -*- coding: utf-8 -*-
"""repr_model_selection.py — 受控对比 exp9 / repr13 月度表征, 遴选最优表征模型
探针 (训练目标月 ≤2022-07, 验证 ≥2022-08; 两模型用相同 (国家,月) 样本集):
  a) 国家识别 LogReg 探针: 月表征 -> 国家 (141 类)
  b) 月熵预测 Ridge 探针: 过去 K=3 个月表征 -> 下一月平滑熵 (主判据)
输出 repr_model_selection.json
"""
import json, os

import numpy as np

DIRS = {"exp9": "embeddings_monthly_exp9", "repr13": "embeddings_monthly_repr13"}
ENT_DIR = "entropy_monthly_smoothed"
TRAIN_END = "2022-07"      # 目标月 ≤ 此 -> 训练
K = 3


def load_embeddings(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith("_monthly_embeddings.npz"):
            c = f[:-len("_monthly_embeddings.npz")]
            z = np.load(d + "/" + f, allow_pickle=True)
            out[c] = {str(m): e for m, e in zip(z["months"], z["embeddings"])}
    return out


def load_entropy():
    out = {}
    for f in sorted(os.listdir(ENT_DIR)):
        if f.endswith("_monthly_entropy.npy"):
            c = f[:-len("_monthly_entropy.npy")]
            out[c] = {str(k): float(v) for k, v in
                      np.load(ENT_DIR + "/" + f, allow_pickle=True).item().items()}
    return out


def build_probe_samples(emb, ent):
    """K=3 月历史 -> 下一月熵; 返回 dict[(c, target_month)] = (x, y)"""
    S = {}
    for c, months in emb.items():
        if c not in ent:
            continue
        ms = sorted(months.keys())
        idx = {m: i for i, m in enumerate(ms)}
        for m_t in ms:
            i = idx[m_t]
            if i < K - 1:
                continue
            hist = ms[i - K + 1: i + 1]
            if len(hist) < K:
                continue
            # 目标 = 下一月 (按月份序, 不一定是 hist 之后紧挨的自然月; 用面板顺序)
            if i + 1 >= len(ms):
                continue
            tgt = ms[i + 1]
            if tgt not in ent[c]:
                continue
            x = np.concatenate([months[h] for h in hist]).astype(np.float64)
            S[(c, tgt)] = (x, ent[c][tgt])
    return S


def ridge_probe(S):
    from sklearn.linear_model import RidgeCV
    keys = sorted(S.keys())
    tr = [k for k in keys if k[1] <= TRAIN_END]
    va = [k for k in keys if k[1] > TRAIN_END]
    Xtr = np.stack([S[k][0] for k in tr]); ytr = np.array([S[k][1] for k in tr])
    Xva = np.stack([S[k][0] for k in va]); yva = np.array([S[k][1] for k in va])
    mu, sd = np.log1p(ytr).mean(), np.log1p(ytr).std() + 1e-8
    mdl = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xtr, (np.log1p(ytr) - mu) / sd)
    pred = np.expm1(mdl.predict(Xva) * sd + mu)
    pred = np.clip(pred, 0, None)
    ss_res = float(((pred - yva) ** 2).sum())
    ss_tot = float(((yva - yva.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    mae = float(np.abs(pred - yva).mean())
    # 对照基线: 训练段国家均值
    cmap = {}
    for k in tr:
        cmap.setdefault(k[0], []).append(S[k][1])
    cm = {c: np.mean(v) for c, v in cmap.items()}
    gm = float(ytr.mean())
    base = np.array([cm.get(k[0], gm) for k in va])
    ss_res_b = float(((base - yva) ** 2).sum())
    r2_base = 1 - ss_res_b / ss_tot
    return dict(n_train=len(tr), n_val=len(va), r2=round(r2, 4),
                mae=round(mae, 4), alpha=float(mdl.alpha_),
                baseline_country_mean_r2=round(r2_base, 4))


def country_probe(emb):
    from sklearn.linear_model import LogisticRegression
    rows, ys = [], []
    for c, months in emb.items():
        for m, e in months.items():
            rows.append((c, m, e))
    tr = [(c, e) for c, m, e in rows if m <= TRAIN_END]
    va = [(c, e) for c, m, e in rows if m > TRAIN_END]
    Xtr = np.stack([e for _, e in tr]); ytr = [c for c, _ in tr]
    Xva = np.stack([e for _, e in va]); yva = [c for c, _ in va]
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
    return dict(n_train=len(tr), n_val=len(va),
                val_acc=round(float(clf.score(Xva, yva)), 4))


def main():
    ents = load_entropy()
    embs = {m: load_embeddings(d) for m, d in DIRS.items()}
    # 相同样本集: 两模型样本键取交集
    S = {m: build_probe_samples(embs[m], ents) for m in DIRS}
    common = sorted(set(S["exp9"]) & set(S["repr13"]))
    print(f"probe samples: exp9={len(S['exp9'])} repr13={len(S['repr13'])} common={len(common)}")
    res = {}
    for m in DIRS:
        Sc = {k: S[m][k] for k in common}
        res[m] = dict(entropy_ridge=ridge_probe(Sc),
                      country_id=country_probe(embs[m]))
        print(m, json.dumps(res[m], ensure_ascii=False), flush=True)
    # 判定: 主判据 ridge R2, 辅以 country_id
    winner = max(DIRS, key=lambda m: (res[m]["entropy_ridge"]["r2"],
                                      res[m]["country_id"]["val_acc"]))
    res["winner"] = winner
    res["criterion"] = "primary: entropy ridge val R2; secondary: country-id val acc"
    json.dump(res, open("repr_model_selection.json", "w"),
              ensure_ascii=False, indent=1)
    print("WINNER:", winner)


if __name__ == "__main__":
    main()
