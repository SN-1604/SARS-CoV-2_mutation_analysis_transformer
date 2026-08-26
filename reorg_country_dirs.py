# -*- coding: utf-8 -*-
"""reorg_country_dirs.py — 梳理 GISAID_MSA_monthly, 剔除非国家目录
动物名(bat/cat/mink/deer...)、env、Europe/Africa、Country、PDL、a 等非国家目录的
表头末字段(或毒株名第3段)含真实国家(已实测核查):
  hCoV-19/mink/Netherlands/...|Netherlands  -> Netherlands
  hCoV-19/env/Wuhan/...|China               -> China
  hCoV-19/Country/PPHRL-6/...|Pakistan      -> Pakistan
解析规则: 末字段 -> CANON; 失败则毒株名第3段 -> CANON; 均失败 -> 剔除。
可恢复记录按 (国家, 月) 追加到目标月文件(对现有内容+新增统一 md5 去重),
处理完毕后删除非国家目录。报告写入 reorg_report.json。
"""
import hashlib, json, os, shutil
from collections import defaultdict
from extract_spike_monthly import CANON, OUT

# 实测确认的 31 个非国家目录 (小写)
JUNK_LC = {"africa", "country", "europe", "pdl", "tadarida brasiliensis", "a",
           "armadillo", "bat", "binturong", "canine", "cat", "coati", "deer",
           "dog", "env", "ferret", "fishing cat", "gorilla", "hamster", "hippo",
           "hyena", "leopard", "lion", "mink", "mouse", "mule deer", "otter",
           "puma", "snow leopard", "syrian hamster", "tiger"}


def main():
    dirs = sorted(d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d)))
    junk = [d for d in dirs if d.lower() in JUNK_LC]
    real = [d for d in dirs if d.lower() not in JUNK_LC]
    GEO = set(real) | set(CANON.values())   # 可接受的规范地理名
    print(f"real dirs: {len(real)}, junk dirs: {len(junk)}")

    # 1) 收集非国家目录中的记录, 按 (目标国家, 月) 分组
    groups = defaultdict(list)              # (country, ym) -> [(hdr, seq)]
    stats = {}                              # junk_dir -> dict
    for jd in junk:
        jdp = os.path.join(OUT, jd)
        st = dict(files=0, recovered=0, dropped=0, targets=defaultdict(int))
        for fn in sorted(os.listdir(jdp)):
            if not fn.endswith(".fasta"):
                continue
            st["files"] += 1
            ym = fn[:-6]                    # YYYY-MM
            hdr = None
            with open(os.path.join(jdp, fn), "rb") as f:
                for line in f:
                    if line.startswith(b">"):
                        hdr = line
                        continue
                    s = line.strip()
                    if not s:
                        continue
                    parts = hdr[1:].decode("latin-1").rstrip("\r\n").split("|")
                    segs = parts[1].split("/") if len(parts) > 1 else []
                    cands = [parts[-1].strip()]
                    if len(segs) > 2:
                        cands.append(segs[2].strip())
                    tgt = None
                    for cand in cands:
                        c = CANON.get(cand, cand)
                        if c in GEO:
                            tgt = c
                            break
                    if tgt is None:
                        st["dropped"] += 1
                        continue
                    groups[(tgt, ym)].append((hdr, s))
                    st["recovered"] += 1
                    st["targets"][tgt] += 1
        stats[jd] = st
        print(f"{jd}: files={st['files']} recovered={st['recovered']} dropped={st['dropped']}", flush=True)

    # 2) 追加到目标国家月文件 (对现有内容+新增统一去重)
    added = dup_skipped = 0
    per_target = defaultdict(int)
    for (tgt, ym), recs in sorted(groups.items()):
        tdir = os.path.join(OUT, tgt)
        os.makedirs(tdir, exist_ok=True)
        tpath = os.path.join(tdir, f"{ym}.fasta")
        seen = set()
        if os.path.exists(tpath):
            with open(tpath, "rb") as f:
                for line in f:
                    if not line.startswith(b">"):
                        s = line.strip()
                        if s:
                            seen.add(hashlib.md5(s).digest())
        with open(tpath, "ab") as w:
            for h, s in recs:
                m = hashlib.md5(s).digest()
                if m in seen:
                    dup_skipped += 1
                    continue
                seen.add(m)
                w.write(h if h.endswith(b"\n") else h + b"\n")
                w.write(s + b"\n")
                added += 1
                per_target[tgt] += 1

    # 3) 删除非国家目录
    for jd in junk:
        shutil.rmtree(os.path.join(OUT, jd))

    rep = {jd: dict(files=st["files"], recovered=st["recovered"],
                    dropped=st["dropped"], targets=dict(st["targets"]))
           for jd, st in stats.items()}
    json.dump(dict(junk_dirs=rep, total_recovered=sum(s["recovered"] for s in stats.values()),
                   total_dropped=sum(s["dropped"] for s in stats.values()),
                   added_to_country_files=added, dup_skipped=dup_skipped,
                   per_target=dict(per_target), remaining_dirs=sorted(
                       d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d)))),
              open("reorg_report.json", "w"), ensure_ascii=False, indent=1)
    print(f"added={added} dup_skipped={dup_skipped} junk_dirs_removed={len(junk)}")


if __name__ == "__main__":
    main()
