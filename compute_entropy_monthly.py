# -*- coding: utf-8 -*-
"""compute_entropy_monthly.py — 计算 MSA_monthly 各国每月比对的香侬熵
定义 (与旧管线 weekly_entropy 一致, 已对账确认):
  entropy(file) = Σ_j H_j,  H_j = -Σ_s p_s·log2(p_s)
  p_s = 第 j 列各字符 (A/C/G/T/-/IUPAC 简并码) 的出现频率; 对全部列求和, 单位 bit。
输出: entropy_monthly/{国家}_monthly_entropy.npy  (dict {"YYYY-MM": float}, 同周熵格式)
"""
import json, os, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SRC = "MSA_monthly"
DST = "entropy_monthly"
WORKERS = 16


def entropy_file(path):
    """单个比对文件的熵: 逐列符号分布的香侬熵(log2)之和"""
    seqs = []
    cur = []
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                if cur:
                    seqs.append(b"".join(cur))
                    cur = []
            else:
                s = line.strip()
                if s:
                    cur.append(s)
    if cur:
        seqs.append(b"".join(cur))
    n = len(seqs)
    if n == 0:
        return 0.0
    arr = np.frombuffer(b"".join(seqs), dtype=np.uint8).reshape(n, -1)
    symbols = np.unique(arr)
    counts = np.stack([(arr == s).sum(0) for s in symbols]).astype(np.float64)  # (S, L)
    p = counts / n
    with np.errstate(divide="ignore"):
        lg = np.where(p > 0, np.log2(p), 0.0)
    H = -(p * lg).sum(0)          # 每列熵
    return float(H.sum())


def job(args):
    country, fn = args
    return country, fn[:-6], entropy_file(os.path.join(SRC, country, fn))


def main():
    tasks = []
    for d in sorted(os.listdir(SRC)):
        dp = os.path.join(SRC, d)
        if not os.path.isdir(dp):
            continue
        for fn in sorted(os.listdir(dp)):
            if fn.endswith(".fasta"):
                tasks.append((d, fn))
    tasks.sort(key=lambda t: -os.path.getsize(os.path.join(SRC, t[0], t[1])))
    print(f"files: {len(tasks)}", flush=True)

    os.makedirs(DST, exist_ok=True)
    per_country = {}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for country, ym, h in ex.map(job, tasks, chunksize=8):
            per_country.setdefault(country, {})[ym] = h
            done += 1
            if done % 500 == 0:
                print(f"[{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)

    summary = {}
    for c, d in sorted(per_country.items()):
        np.save(os.path.join(DST, f"{c}_monthly_entropy.npy"), d)
        vals = list(d.values())
        summary[c] = dict(months=len(d), min=round(min(vals), 4), max=round(max(vals), 4))
    json.dump(summary, open("entropy_monthly_summary.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"DONE in {time.time()-t0:.0f}s. countries={len(per_country)}", flush=True)


if __name__ == "__main__":
    main()
