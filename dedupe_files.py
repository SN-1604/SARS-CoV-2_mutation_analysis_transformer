# -*- coding: utf-8 -*-
"""dedupe_files.py — 全量扫描 GISAID_MSA_monthly, 对含重复序列的文件原地重洗(保留首次出现)
背景: Windows 文件系统大小写不敏感导致 BRAZIL/Brazil 等大小写变体写入同一物理目录,
而 chunk 阶段去重状态按原始名分键, 产生少量重复记录 (实测 19 文件 53 条)。
"""
import hashlib, os, json
from extract_spike_monthly import OUT


def dedupe_file(path):
    """返回 (记录数, 删除重复数); 无重复返回 None"""
    seen = set()
    recs = []
    n = dropped = 0
    hdr = None
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                hdr = line
            else:
                s = line.strip()
                if not s:
                    continue
                n += 1
                m = hashlib.md5(s).digest()
                if m in seen:
                    dropped += 1
                    continue
                seen.add(m)
                recs.append((hdr, s))
    if not dropped:
        return None
    tmp = path + ".tmp"
    with open(tmp, "wb") as w:
        for h, s in recs:
            w.write(h if h.endswith(b"\n") else h + b"\n")
            w.write(s + b"\n")
    os.replace(tmp, path)
    return n, dropped


def main():
    fixed = {}
    total_files = total_dropped = 0
    for d in sorted(os.listdir(OUT)):
        dp = os.path.join(OUT, d)
        if not os.path.isdir(dp):
            continue
        for fn in sorted(os.listdir(dp)):
            if not fn.endswith(".fasta"):
                continue
            total_files += 1
            r = dedupe_file(os.path.join(dp, fn))
            if r:
                fixed[f"{d}/{fn}"] = dict(recs=r[0], dropped=r[1])
                total_dropped += r[1]
                print(f"fixed {d}/{fn}: recs={r[0]} dropped={r[1]}", flush=True)
    json.dump(dict(files=total_files, fixed=fixed, dropped=total_dropped),
              open("dedupe_report.json", "w"), ensure_ascii=False, indent=1)
    print(f"scanned {total_files} files, fixed {len(fixed)}, dropped {total_dropped} dup records")


if __name__ == "__main__":
    main()
