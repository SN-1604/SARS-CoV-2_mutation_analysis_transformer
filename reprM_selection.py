# -*- coding: utf-8 -*-
"""reprM_selection.py — repr13月均 vs reprM(m1_ce) vs reprM(m4_supcon) 三方对比
相同样本集: a) 国家识别 LogReg 探针  b) 月熵 Ridge 探针 (K=3 -> 下一月平滑熵)
输出 reprM_selection.json
"""
import json
import numpy as np

from repr_model_selection import (DIRS, load_embeddings, load_entropy,
                                  build_probe_samples, ridge_probe,
                                  country_probe)

SETS = {"repr13": "embeddings_monthly",
        "reprM_m1": "embeddings_monthly_v2_m1",
        "reprM_m4": "embeddings_monthly_v2_m4"}


def main():
    ents = load_entropy()
    embs = {m: load_embeddings(d) for m, d in SETS.items()}
    S = {m: build_probe_samples(embs[m], ents) for m in SETS}
    common = sorted(set.intersection(*[set(s) for s in S.values()]))
    print("common samples:", len(common))
    res = {}
    for m in SETS:
        Sc = {k: S[m][k] for k in common}
        res[m] = dict(entropy_ridge=ridge_probe(Sc),
                      country_id=country_probe(embs[m]))
        print(m, json.dumps(res[m], ensure_ascii=False), flush=True)
    winner = max(SETS, key=lambda m: (res[m]["entropy_ridge"]["r2"],
                                      res[m]["country_id"]["val_acc"]))
    res["winner"] = winner
    json.dump(res, open("reprM_selection.json", "w"), ensure_ascii=False, indent=1)
    print("WINNER:", winner)


if __name__ == "__main__":
    main()
