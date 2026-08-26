# -*- coding: utf-8 -*-
"""align_monthly.py — 用 MAFFT 将 GISAID_MSA_monthly 逐文件比对到 MSA_monthly
调度 (32 核):
  超大文件 (n>4000): 串行逐文件, 内部 "分块并行比对 + mafft --merge 合并"
    — 实测本版本 mafft 的 --parttree 路径 (splittbfast) 不传 -C, 永远单线程;
      --merge 走 tbfast 支持多线程且保持各子比对不变 (官方文档)。
    分块: round-robin 切成 ~1800 条/块, 8 路并行 --auto --thread 4;
    合并: mafft --merge table input --thread 32。
  中文件 (2000<n<=4000): 串行, --auto --thread 32。
  小文件 (n<=2000): 16 路并行, --auto --thread 2; n=1 直接复制。
调用要点 (已实测): mafft.bat 不支持 stdin; %* 按空格切参 -> 输入复制到无空格
临时目录 _aln_tmp/, 以其为 cwd 相对路径调用; --preservecase 保留大写。
断点续跑: 输出存在且非空则跳过 (写出经 tmp+os.replace, 无半成品)。
失败记录 align_monthly_errors.json; 进度 align_monthly_log.json。
用法: python align_monthly.py [smoke]
"""
import json, os, queue, shutil, subprocess, sys, threading, time

SRC = "GISAID_MSA_monthly"
DST = "MSA_monthly"
TMP = "_aln_tmp"
MAFFT = r"H:\Bioinfo\mafft-win\mafft.bat"
LOG = "align_monthly_log.json"
ERR = "align_monthly_errors.json"
BIG = 2000            # 小文件上限
HUGE = 4000           # 超过则 split+merge
CHUNK = 1800          # 分块大小
CH_WORKERS = 8        # 块间并行
CH_THREADS = 4        # 块内线程
THREADS_SMALL = 2
WORKERS = 16
THREADS_BIG = 32

_lock = threading.Lock()
_log = []
_errors = []


def robust_replace(src, dst, tries=15, wait=2.0):
    """Windows 下 mafft 孙进程继承句柄释放滞后, os.replace 偶发 WinError 32 ->
    重试; 仍失败则退回 copyfile+remove (读锁定的源文件通常允许)"""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == tries - 1:
                break
            time.sleep(wait)
    shutil.copyfile(src, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    os.remove(src)


def count_seqs(path):
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                n += 1
    return n


def check_out(path, n):
    """校验 mafft 输出: 序列数一致、非空、所有序列等长"""
    cnt = 0
    cur = 0
    L = -1
    try:
        with open(path, "rb") as f:
            for line in f:
                if line.startswith(b">"):
                    if cnt > 0 and L >= 0 and cur != L:
                        return False
                    if cnt > 0:
                        L = cur
                    cnt += 1
                    cur = 0
                else:
                    cur += len(line.strip())
        if cnt > 0:
            if L < 0:
                L = cur
            elif cur != L:
                return False
    except OSError:
        return False
    return cnt == n and L > 0


def run_mafft(in_name, out_name, ws, extra_flags, threads):
    """在 ws 目录内调 mafft.bat (相对路径), 返回 (rc, err_tail)"""
    cmd = ["cmd", "/c", MAFFT, *extra_flags, "--preservecase", "--quiet",
           "--thread", str(threads), in_name]
    err_f = os.path.join(ws, "err.txt")
    with open(out_name, "wb") as w, open(err_f, "wb") as e:
        r = subprocess.run(cmd, stdout=w, stderr=e, cwd=ws)
    tail = open(err_f, "rb").read()[-500:].decode("latin-1", "replace")
    return r.returncode, tail


def align_one(rel, n, threads, ws, extra=None):
    """中小文件: 单次 mafft 调用"""
    src = os.path.join(SRC, rel)
    dst = os.path.join(DST, rel)
    t0 = time.time()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if n == 1:
        shutil.copyfile(src, dst + ".tmp")
        robust_replace(dst + ".tmp", dst)
        return rel, n, time.time() - t0, "copy"
    os.makedirs(ws, exist_ok=True)
    shutil.copyfile(src, os.path.join(ws, "in.fasta"))
    out_f = os.path.join(ws, "out.fasta")
    flags = ["--auto"] + (extra or [])
    rc, tail = run_mafft("in.fasta", out_f, ws, flags, threads)
    if rc != 0 or not check_out(out_f, n):
        raise RuntimeError(f"rc={rc} err={tail}")
    robust_replace(out_f, dst)
    return rel, n, time.time() - t0, "mafft"


def split_round_robin(src, k, ws):
    """把 fasta 记录 round-robin 切成 k 块, 返回各块记录数"""
    outs = [open(os.path.join(ws, f"c{i}.fasta"), "wb") for i in range(k)]
    counts = [0] * k
    idx = 0
    with open(src, "rb") as f:
        cur = None
        for line in f:
            if line.startswith(b">"):
                if cur is not None:
                    outs[idx % k].write(b"".join(cur))
                    counts[idx % k] += 1
                    idx += 1
                cur = [line]
            else:
                if cur is not None:
                    cur.append(line)
        if cur is not None:
            outs[idx % k].write(b"".join(cur))
            counts[idx % k] += 1
    for o in outs:
        o.close()
    return counts


def align_big(rel, n, ws):
    """超大文件: 分块并行比对 + mafft --merge 合并"""
    from concurrent.futures import ThreadPoolExecutor
    src = os.path.join(SRC, rel)
    dst = os.path.join(DST, rel)
    t0 = time.time()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(ws):
        shutil.rmtree(ws)
    os.makedirs(ws)
    k = (n + CHUNK - 1) // CHUNK
    counts = split_round_robin(src, k, ws)

    def align_chunk(i):
        cdir = os.path.join(ws, f"c{i}")
        os.makedirs(cdir, exist_ok=True)
        os.replace(os.path.join(ws, f"c{i}.fasta"), os.path.join(cdir, "in.fasta"))
        out_f = os.path.join(cdir, "out.fasta")
        rc, tail = run_mafft("in.fasta", out_f, cdir, ["--auto"], CH_THREADS)
        if rc != 0 or not check_out(out_f, counts[i]):
            raise RuntimeError(f"chunk{i} rc={rc} err={tail}")
        return out_f

    with ThreadPoolExecutor(max_workers=CH_WORKERS) as ex:
        chunk_msas = list(ex.map(align_chunk, range(k)))
    # 合并: input = 各子比对顺序拼接, table = 各子比对记录区间 (1-based)
    in_f = os.path.join(ws, "min.fasta")
    with open(in_f, "wb") as w:
        for p in chunk_msas:
            with open(p, "rb") as r:
                shutil.copyfileobj(r, w)
    start = 1
    with open(os.path.join(ws, "table.txt"), "w") as t:
        for i, c in enumerate(counts):
            t.write(" ".join(str(x) for x in range(start, start + c)) + f" # chunk{i}\n")
            start += c
    out_f = os.path.join(ws, "mout.fasta")
    rc, tail = run_mafft("min.fasta", out_f, ws,
                         ["--merge", "table.txt", "--retree", "1"], THREADS_BIG)
    if rc != 0 or not check_out(out_f, n):
        raise RuntimeError(f"merge rc={rc} err={tail}")
    robust_replace(out_f, dst)
    shutil.rmtree(ws, ignore_errors=True)
    return rel, n, time.time() - t0, f"split{k}+merge"


def done(rel):
    p = os.path.join(DST, rel)
    return os.path.exists(p) and os.path.getsize(p) > 0


def main():
    from concurrent.futures import ThreadPoolExecutor
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    tasks = []
    for d in sorted(os.listdir(SRC)):
        dp = os.path.join(SRC, d)
        if not os.path.isdir(dp):
            continue
        for fn in sorted(os.listdir(dp)):
            if fn.endswith(".fasta"):
                p = os.path.join(dp, fn)
                tasks.append((os.path.join(d, fn), count_seqs(p)))
    print(f"total files: {len(tasks)}", flush=True)
    if smoke:
        big1 = max(tasks, key=lambda t: t[1])
        mid = next(t for t in tasks if 2000 < t[1] <= 4000)
        sp = next(t for t in tasks if " " in t[0] and 100 < t[1] <= 2000)
        one = next(t for t in tasks if t[1] == 1)
        sm = next(t for t in tasks if 50 < t[1] < 100)
        tasks = [big1, mid, sp, one, sm]
        print("smoke tasks:", tasks, flush=True)
    huge = sorted([t for t in tasks if t[1] > HUGE], key=lambda t: -t[1])
    mid = sorted([t for t in tasks if BIG < t[1] <= HUGE], key=lambda t: -t[1])
    small = sorted([t for t in tasks if t[1] <= BIG], key=lambda t: -t[1])
    print(f"huge: {len(huge)}, mid: {len(mid)}, small: {len(small)}", flush=True)

    t_start = time.time()
    # 阶段1: 超大文件串行 (内部并行) + 中文件串行 (打满线程)
    for tag, group, thr in (("huge", huge, None), ("mid", mid, THREADS_BIG)):
        for i, (rel, n) in enumerate(group):
            if done(rel):
                continue
            try:
                if thr is None:
                    _, _, sec, mode = align_big(rel, n, os.path.join(TMP, "wbig"))
                else:
                    _, _, sec, mode = align_one(rel, n, thr, os.path.join(TMP, "wmid"),
                                                extra=["--retree", "1"])
                with _lock:
                    _log.append(dict(rel=rel, n=n, sec=round(sec, 1), mode=mode))
                print(f"[{tag} {i+1}/{len(group)}] {rel} n={n} {sec:.0f}s", flush=True)
            except Exception as ex:
                with _lock:
                    _errors.append(dict(rel=rel, n=n, err=str(ex)[-500:]))
                print(f"[{tag} {i+1}/{len(group)}] FAIL {rel}: {str(ex)[:200]}", flush=True)
    # 阶段2: 小文件并行
    ws_q = queue.Queue()
    for w in range(WORKERS):
        ws_q.put(os.path.join(TMP, f"w{w}"))

    def worker(task):
        rel, n = task
        if done(rel):
            return None
        ws = ws_q.get()
        try:
            _, _, sec, mode = align_one(rel, n, THREADS_SMALL, ws)
            return dict(rel=rel, n=n, sec=round(sec, 1), mode=mode)
        except Exception as ex:
            return dict(rel=rel, n=n, err=str(ex)[-500:])
        finally:
            ws_q.put(ws)

    cnt = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(worker, small):
            cnt += 1
            with _lock:
                if res is None:
                    pass
                elif "err" in res:
                    _errors.append(res)
                else:
                    _log.append(res)
            if cnt % 500 == 0:
                print(f"[small {cnt}/{len(small)}] elapsed {time.time()-t_start:.0f}s", flush=True)
    json.dump(_log, open(LOG, "w"), ensure_ascii=False, indent=1)
    json.dump(_errors, open(ERR, "w"), ensure_ascii=False, indent=1)
    print(f"DONE in {time.time()-t_start:.0f}s. aligned={len(_log)} failed={len(_errors)}", flush=True)


if __name__ == "__main__":
    main()
