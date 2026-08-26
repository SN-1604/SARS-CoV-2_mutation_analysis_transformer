# -*- coding: utf-8 -*-
"""qc_clean_msa.py — 核对并清洗 MSA_monthly 比对结果
对每个 MSA 文件:
  1) 结构核对: 序列数与 GISAID_MSA_monthly 输入一致、所有序列等长、字符 ⊆ ACGT-
     硬失败 -> 记入 hard_fail, 文件保留待复核
  2) 去除明显错误 gap 列: n>=500 时删除非 gap 占用数 <= max(2, n*0.002) 的列
     (错位/伪插入特征; 真实 indel 在月文件中的频率远高于此阈值)
  3) 错位序列修复: gap 数 > max(25, 5*中位数+10) 的离群序列, 去 gap 后与一致序列
     (逐列多数表决) 做双序列 mafft 重比对, 按一致序列坐标映射回 MSA; 相对一致序列
     的插入残基丢弃并计数; 重比对后仍离群 -> 恢复原样并记入 unfixable
清洗后原子写回原路径。报告: msa_qc_report.json
"""
import json, os, queue, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SRC = "GISAID_MSA_monthly"
DST = "MSA_monthly"
TMP = "_qc_tmp"
MAFFT = r"H:\Bioinfo\mafft-win\mafft.bat"
REPORT = "msa_qc_report.json"
WORKERS = 16
BASES = b"ACGT"
ALPHABET = set(b"ACGTURYSWKMBDHVN-")   # IUPAC 核苷酸全集 + gap (源序列含合法简并碱基)
GAP = ord("-")


def read_aln(path):
    hdrs, seqs = [], []
    cur = None
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                hdrs.append(line.rstrip(b"\r\n"))
                if cur is not None:
                    seqs.append(b"".join(cur))
                cur = []
            else:
                s = line.strip()
                if s:
                    cur.append(s)
    if cur is not None:
        seqs.append(b"".join(cur))
    return hdrs, seqs


def count_src_seqs(path):
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                n += 1
    return n


def realign_to_consensus(consensus, seq_ungapped, ws):
    """双序列 mafft 重比对, 返回映射回一致序列坐标的序列(bytes)及丢弃的插入残基数"""
    os.makedirs(ws, exist_ok=True)
    in_f = os.path.join(ws, "q_in.fasta")
    out_f = os.path.join(ws, "q_out.fasta")
    with open(in_f, "wb") as w:
        w.write(b">consensus\n" + consensus + b"\n>query\n" + seq_ungapped + b"\n")
    cmd = ["cmd", "/c", MAFFT, "--auto", "--preservecase", "--quiet",
           "--thread", "2", "q_in.fasta"]
    with open(out_f, "wb") as w, open(os.path.join(ws, "q_err.txt"), "wb") as e:
        r = subprocess.run(cmd, stdout=w, stderr=e, cwd=ws)
    if r.returncode != 0:
        raise RuntimeError(f"mafft rc={r.returncode}")
    _, pair = read_aln(out_f)
    if len(pair) != 2:
        raise RuntimeError("unexpected pair output")
    c_aln, o_aln = pair
    # 一致序列第 j 个非 gap 字符 -> 双序列比对中的列号
    cmap = []
    for i, ch in enumerate(c_aln):
        if ch != GAP:
            cmap.append(i)
    if len(cmap) != len(consensus):
        raise RuntimeError("consensus map mismatch")
    dropped = 0
    out = bytearray(len(consensus))
    for j, q in enumerate(cmap):
        out[j] = o_aln[q]
    dropped = sum(1 for i, ch in enumerate(o_aln) if ch != GAP and c_aln[i] == GAP)
    return bytes(out), dropped


def qc_file(rel, ws):
    src = os.path.join(SRC, rel)
    dst = os.path.join(DST, rel)
    rec = dict(rel=rel)
    hdrs, seqs = read_aln(dst)
    n = len(seqs)
    rec["n"] = n
    # 1) 结构核对
    if n != count_src_seqs(src):
        rec["hard_fail"] = "seq count mismatch"
        return rec
    if len({len(s) for s in seqs}) != 1:
        rec["hard_fail"] = "unequal lengths"
        return rec
    arr = np.frombuffer(b"".join(seqs), dtype=np.uint8).reshape(n, -1).copy()
    bad = ~np.isin(arr, list(ALPHABET))
    if bad.any():
        rec["hard_fail"] = f"bad chars x{int(bad.sum())}"
        return rec
    L0 = arr.shape[1]
    rec["L_before"] = int(L0)
    changed = False
    # 2) 删除低占用列 (明显错误 gap / 伪插入)
    occ = (arr != GAP).sum(axis=0)
    thr = max(2, int(n * 0.002)) if n >= 500 else 0
    if thr:
        keep = occ > thr
        rec["cols_removed"] = int((~keep).sum())
        rec["residues_removed"] = int(occ[~keep].sum())
        if rec["cols_removed"]:
            arr = arr[:, keep]
            changed = True
    # 3) 离群 (错位) 序列修复
    g = (arr == GAP).sum(axis=1)
    med = float(np.median(g))
    mad = float(np.median(np.abs(g - med)))
    cut = max(25.0, 5 * med + 10, med + 6 * 1.4826 * mad)
    idx = np.nonzero(g > cut)[0]
    rec["outliers"] = int(len(idx))
    realigned = unfix = dropped_ins = 0
    if len(idx):
        # 逐列多数表决一致序列
        counts = np.stack([(arr == b).sum(0) for b in BASES])
        consensus = np.array(list(BASES), dtype=np.uint8)[counts.argmax(0)].tobytes()
        for i in idx:
            ung = arr[i][arr[i] != GAP].tobytes()
            try:
                new, d = realign_to_consensus(consensus, ung, ws)
                dropped_ins += d
                if new.count(GAP) <= cut:
                    arr[i] = np.frombuffer(new, dtype=np.uint8)
                    realigned += 1
                    changed = True
                else:
                    unfix += 1
            except Exception as ex:
                rec.setdefault("realign_errors", []).append(str(ex)[:200])
                unfix += 1
    rec["realigned"] = realigned
    rec["unfixable"] = unfix
    rec["insert_residues_dropped"] = dropped_ins
    rec["L_after"] = int(arr.shape[1])
    if changed:
        tmp = dst + ".tmp"
        with open(tmp, "wb") as w:
            for i in range(n):
                w.write(hdrs[i] + b"\n" + arr[i].tobytes() + b"\n")
        os.replace(tmp, dst)
    return rec


def main():
    tasks = []
    for d in sorted(os.listdir(DST)):
        dp = os.path.join(DST, d)
        if not os.path.isdir(dp):
            continue
        for fn in sorted(os.listdir(dp)):
            if fn.endswith(".fasta"):
                tasks.append(os.path.join(d, fn))
    tasks.sort(key=lambda rel: -os.path.getsize(os.path.join(DST, rel)))
    print(f"qc files: {len(tasks)}", flush=True)
    ws_q = queue.Queue()
    for k in range(WORKERS):
        ws_q.put(os.path.join(TMP, f"w{k}"))

    def run(rel):
        ws = ws_q.get()
        try:
            return qc_file(rel, ws)
        except Exception as ex:
            return dict(rel=rel, hard_fail=f"exception: {ex}"[:300])
        finally:
            ws_q.put(ws)

    t0 = time.time()
    recs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, rec in enumerate(ex.map(run, tasks)):
            recs.append(rec)
            if (i + 1) % 500 == 0:
                print(f"[{i+1}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    acted = [r for r in recs if r.get("cols_removed") or r.get("realigned")
             or r.get("hard_fail") or r.get("outliers")]
    summary = dict(
        files=len(recs),
        hard_fail=[r for r in recs if r.get("hard_fail")],
        cols_removed=sum(r.get("cols_removed", 0) for r in recs),
        residues_removed=sum(r.get("residues_removed", 0) for r in recs),
        outliers=sum(r.get("outliers", 0) for r in recs),
        realigned=sum(r.get("realigned", 0) for r in recs),
        unfixable=sum(r.get("unfixable", 0) for r in recs),
        files_acted=len(acted),
    )
    json.dump(dict(summary=summary, files=recs), open(REPORT, "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v))
                      for k, v in summary.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
