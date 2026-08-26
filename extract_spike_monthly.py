# -*- coding: utf-8 -*-
"""extract_spike_monthly.py — 从 spikenuc1116.fasta (64.65GB) 提取 spike 核苷酸序列并按月归档
规则:
  - 长度 3800-3900 bp; 采集日期 ≤ 2022-12-31 (需含年月)
  - 去除含 X/x/N/n 的序列; 序列统一大写
  - 按国家(毒株名第2段)+月份归档: GISAID_MSA_monthly/{国家}/{YYYY-MM}.fasta
  - 每国每月序列去重 (md5); 末段做国名变体合并 (finalize; 中国省市/港澳台全部并入 China)
用法:
  python extract_spike_monthly.py chunk --i k --n 10     # 处理第 k 块
  python extract_spike_monthly.py finalize               # 国名合并+报告
"""
import argparse, hashlib, os, pickle, re, sys, time
from collections import defaultdict

SRC = "spikenuc1116.fasta"
OUT = "GISAID_MSA_monthly"
STATE = "extract_spike_state.pkl"
LOG = "extract_spike_log.json"
DATE_CUT = (2022, 12, 31)

_re_date = re.compile(r"^(\d{4})(?:-(\d{2}|XX))?(?:-(\d{2}|XX))?")


def parse_header(hline):
    """返回 (规范国家名, (y,m)) 或 None
    国家取毒株名第2段; 若为空或落在动物/env/洲等非国家名, 依次回退到表头末字段、
    毒株名第3段; 最终结果经 CANON 规范化。均无法解析出国家则返回 None (剔除)。"""
    try:
        parts = hline[1:].decode("latin-1").rstrip("\r\n").split("|")
        if len(parts) < 3:
            return None
        seg = parts[1].split("/")
        place = seg[1].strip() if len(seg) > 1 else ""
        if not place or place.lower() in JUNK_PLACE:
            place = ""
            cands = [parts[-1].strip()]
            if len(seg) > 2:
                cands.append(seg[2].strip())
            for cand in cands:
                c = CANON.get(cand, cand)
                if c in GEO_PLACES:
                    place = c
                    break
            if not place:
                return None
        else:
            place = CANON.get(place, place)
        d = parts[2].strip()
        m = _re_date.match(d)
        if not m:
            return None
        y = int(m.group(1))
        mm = m.group(2)
        dd = m.group(3)
        if y > 2022:
            return None
        if mm in (None, "XX"):      # 无月份无法归入月文件
            return None
        mo = int(mm)
        if mo < 1 or mo > 12:
            return None
        if dd not in (None, "XX") and y == 2022 and mo == 12 and int(dd) > 31:
            return None
        return place, (y, mo)
    except Exception:
        return None


def run_chunk(i, n):
    size = os.path.getsize(SRC)
    bounds = [size * k // n for k in range(n)] + [size]
    start, end = bounds[i], bounds[i + 1]
    t0 = time.time()
    state = defaultdict(lambda: defaultdict(set))
    if os.path.exists(STATE):
        state.update(pickle.load(open(STATE, "rb")))   # pickle 中是普通 dict, update 保留 defaultdict 行为
    os.makedirs(OUT, exist_ok=True)
    bufs = defaultdict(list)
    bufsz = 0
    n_seen = n_len_ok = n_date_ok = n_clean = n_new = 0

    def flush():
        nonlocal bufsz
        for (pl, ym), chunks in bufs.items():
            d = os.path.join(OUT, pl)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{ym[0]:04d}-{ym[1]:02d}.fasta"), "ab") as w:
                for c in chunks:
                    w.write(c)
        bufs.clear()
        bufsz = 0

    f = open(SRC, "rb", buffering=4 * 1024 * 1024)
    f.seek(start)
    if start > 0:
        f.readline()                      # 跳过跨界残行
    cur = None                            # (place, ym, header_bytes)
    seq_parts = []

    def process():
        nonlocal n_len_ok, n_date_ok, n_clean, n_new, bufsz
        if cur is None:
            return
        pl, ym, hb = cur
        seq = b"".join(seq_parts).upper()
        if not (3800 <= len(seq) <= 3900):
            return
        n_len_ok += 1
        if b"N" in seq or b"X" in seq:
            return
        n_clean += 1
        h = hashlib.md5(seq).digest()
        if h in state[pl][ym]:
            return
        state[pl][ym].add(h)
        n_new += 1
        rec = hb if hb.endswith(b"\n") else hb + b"\n"
        bufs[(pl, ym)].append(rec + seq + b"\n")
        bufsz += len(hb) + len(seq) + 2
        if bufsz > 256 * 1024 * 1024:
            flush()

    while True:
        line_start = f.tell()
        line = f.readline()
        if not line:
            process()
            break
        if line.startswith(b">"):
            if line_start >= end:
                break                       # 属于下一块
            process()
            ph = parse_header(line)
            n_seen += 1
            if ph is None:
                cur = None
            else:
                n_date_ok += 1
                cur = (ph[0], ph[1], line)
            seq_parts = []
        else:
            if cur is not None:
                s = line.strip()
                if s:
                    seq_parts.append(s)
    f.close()
    flush()
    pickle.dump(dict(state), open(STATE, "wb"))
    import json
    log = json.load(open(LOG, encoding="gb18030")) if os.path.exists(LOG) else {}
    log[f"chunk{i}"] = dict(start=start, end=end, seen=n_seen, date_ok=n_date_ok,
                           len_ok=n_len_ok, clean=n_clean, new=n_new,
                           sec=round(time.time() - t0, 1))
    json.dump(log, open(LOG, "w"), ensure_ascii=False, indent=1)
    print(f"chunk{i} done: seen={n_seen} date_ok={n_date_ok} len_ok={n_len_ok} "
          f"clean={n_clean} new={n_new} sec={time.time()-t0:.0f}", flush=True)


# 国名变体 -> 规范名 (以 OWID/MSA 命名为基准; 覆盖全文件 413 种地名普查发现的变体)
CANON = {
    # 美国
    "USA": "United States", "United States of America": "United States",
    "US": "United States", "USA-TN": "United States", "USA DUD": "United States",
    # 英国
    "UK": "United Kingdom", "Great Britain": "United Kingdom",
    "England": "United Kingdom", "Scotland": "United Kingdom",
    "Wales": "United Kingdom", "Northern Ireland": "United Kingdom",
    # 韩国/朝鲜
    "South Korea": "South Korea", "SouthKorea": "South Korea",
    "Korea, South": "South Korea", "Korea": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea, North": "North Korea",
    # 俄罗斯/伊朗/越南/捷克
    "Russia": "Russia", "Russian Federation": "Russia",
    "Iran": "Iran", "IRAN": "Iran", "Iran (Islamic Republic of)": "Iran",
    "Vietnam": "Vietnam", "Viet Nam": "Vietnam",
    "Czechia": "Czechia", "Czech Republic": "Czechia",
    "Czech_Republic": "Czechia", "CzechRepublic": "Czechia",
    # 科特迪瓦/刚果(金)/刚果(布)
    "Cote d'Ivoire": "Cote d'Ivoire", "Ivory Coast": "Cote d'Ivoire",
    "CotedIvoire": "Cote d'Ivoire", "Cote d Ivoire": "Cote d'Ivoire",
    "Democratic Republic of Congo": "Democratic Republic of Congo",
    "Democratic Republic of the Congo": "Democratic Republic of Congo",
    "DRC": "Democratic Republic of Congo",
    "Congo": "Congo", "Republic of Congo": "Congo", "Republic of the Congo": "Congo",
    # 拉美
    "Bolivia": "Bolivia", "Bolivia (Plurinational State of)": "Bolivia",
    "Venezuela": "Venezuela", "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Brazil": "Brazil", "Brasil": "Brazil", "BRAZIL": "Brazil",
    "Peru": "Peru", "PERU": "Peru",
    "Mexico": "Mexico", "mexico": "Mexico", "MEX": "Mexico", "MexicoMEX": "Mexico",
    "Chile": "Chile", "Chile-MA": "Chile",
    # 亚洲/大洋洲拼写变体
    "Australia": "Australia", "Austraila": "Australia",
    "India": "India", "INDIA": "India", "india": "India",
    "Sri Lanka": "Sri Lanka", "SriLanka": "Sri Lanka", "LKA": "Sri Lanka",
    "Indonesia": "Indonesia", "IndonesiA": "Indonesia", "indonesia": "Indonesia",
    "Mongolia": "Mongolia", "mongolia": "Mongolia",
    "Cambodia": "Cambodia", "CAM": "Cambodia",
    "Afghanistan": "Afghanistan", "Afganistan": "Afghanistan",
    "Laos": "Laos", "laos": "Laos", "Lao PDR": "Laos",
    "Lao People's Democratic Republic": "Laos",
    "Saudi Arabia": "Saudi Arabia", "Saudi": "Saudi Arabia",
    "Iraq": "Iraq", "Erbil": "Iraq",
    "UAE": "United Arab Emirates",
    # 欧洲拼写变体
    "Italy": "Italy", "ITA": "Italy", "Ialy": "Italy", "Ital": "Italy", "taly": "Italy",
    "Spain": "Spain", "spain": "Spain", "Spaiin": "Spain", "Spin": "Spain",
    "Spain and plub": "Spain",
    "Fance": "France",
    "Romania": "Romania", "romania": "Romania", "Romnaia": "Romania",
    "Malta": "Malta", "MAlta": "Malta", "malta": "Malta",
    "Andorra": "Andorra", "Andorre": "Andorra",
    "Liechtenstein": "Liechtenstein", "Liechstenstein": "Liechtenstein",
    "North Macedonia": "North Macedonia", "Macedonia": "North Macedonia",
    "NorthMacedonia": "North Macedonia",
    "Moldova": "Moldova", "Republic of Moldova": "Moldova",
    "Slovakia": "Slovakia", "Slovak Republic": "Slovakia",
    "Turkey": "Turkey", "Turkiye": "Turkey", "Türkiye": "Turkey",
    "Netherlands": "Netherlands", "The Netherlands": "Netherlands",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia": "Bosnia and Herzegovina",
    # 非洲
    "South Africa": "South Africa", "SouthAfrica": "South Africa",
    "South_Africa": "South Africa", "South-Africa": "South Africa",
    "Senegal": "Senegal", "senegal": "Senegal",
    "Mauritania": "Mauritania", "Mauritanie": "Mauritania",
    "Cameroon": "Cameroon", "Cameroun": "Cameroon",
    "Botswana": "Botswana", "Botswna": "Botswana",
    "Zambia": "Zambia", "Zambai": "Zambia",
    "Central African Republic": "Central African Republic",
    "CAR": "Central African Republic",
    "Djibouti": "Djibouti", "DJI": "Djibouti",
    "Guinea-Bissau": "Guinea-Bissau", "Guinea Bissau": "Guinea-Bissau",
    "Burkina Faso": "Burkina Faso", "BurkinaFaso": "Burkina Faso",
    "Cabo Verde": "Cabo Verde", "Cabo verde": "Cabo Verde", "Cape Verde": "Cabo Verde",
    "Eswatini": "Eswatini", "Swaziland": "Eswatini",
    "Tanzania": "Tanzania", "Tanzania, United Republic of": "Tanzania",
    "United Republic of Tanzania": "Tanzania",
    "Gambia": "Gambia", "Gambia, The": "Gambia", "The Gambia": "Gambia",
    "Egypt": "Egypt", "Egypt, Arab Rep.": "Egypt",
    "Yemen": "Yemen", "Yemen, Rep.": "Yemen",
    # 加勒比/拉美领地
    "Dominican Republic": "Dominican Republic", "DominicanRepublic": "Dominican Republic",
    "DOM": "Dominican Republic",
    "Haiti": "Haiti", "HAITI": "Haiti",
    "El Salvador": "El Salvador", "El_Salvador": "El Salvador",
    "Trinidad and Tobago": "Trinidad and Tobago", "Trinidad": "Trinidad and Tobago",
    "Antigua and Barbuda": "Antigua and Barbuda", "Antigua": "Antigua and Barbuda",
    "Saint Lucia": "Saint Lucia", "SaintLucia": "Saint Lucia",
    "Saint Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Saint Vincent and The Grenadines": "Saint Vincent and the Grenadines",
    "Saint Martin": "Saint Martin", "Saint-Martin": "Saint Martin",
    "U.S. Virgin Islands": "U.S. Virgin Islands", "US Virgin Islands": "U.S. Virgin Islands",
    "Turks and Caicos Islands": "Turks and Caicos Islands",
    "Turks and Caicos": "Turks and Caicos Islands",
    "Montserrat": "Montserrat", "Monsterrat": "Montserrat",
    # 法国海外领地/太平洋
    "French Guiana": "French Guiana", "FrenchGuiana": "French Guiana",
    "French_Guiana": "French Guiana", "Guyane": "French Guiana",
    "French Polynesia": "French Polynesia", "FrenchPolynesia": "French Polynesia",
    "Tahiti": "French Polynesia",
    "New Caledonia": "New Caledonia", "NewCaledonia": "New Caledonia",
    "New Caledonie": "New Caledonia",
    # 其他
    "Timor": "Timor", "Timor-Leste": "Timor", "East Timor": "Timor",
    "Syria": "Syria", "Syrian Arab Republic": "Syria",
    "Brunei": "Brunei", "Brunei Darussalam": "Brunei",
    "Palestine": "Palestine", "State of Palestine": "Palestine",
    "Palestinian Territory": "Palestine",
    "Myanmar": "Myanmar", "Burma": "Myanmar",
    "Micronesia": "Micronesia", "Micronesia (Federated States of)": "Micronesia",
    "Kyrgyzstan": "Kyrgyzstan", "Kyrgyz Republic": "Kyrgyzstan",
    "Bahamas": "Bahamas", "Bahamas, The": "Bahamas", "The Bahamas": "Bahamas",
    "Puerto Rico": "Puerto Rico",
    "Canary Islands": "Canary Islands",
    "Sao Tome and Principe": "Sao Tome and Principe",
    "Saint Kitts and Nevis": "Saint Kitts and Nevis",
    # 中国: 省市/港澳台/拼写变体全部并入
    "China": "China", "Chinay": "China",
    "Hong Kong": "China", "Hong Kong SAR": "China",
    "Macao": "China", "Macau": "China", "Macao SAR": "China", "Taiwan": "China",
    "Anhui": "China", "Beijing": "China", "Chongqing": "China", "Fujian": "China",
    "Gansu": "China", "Guangdong": "China", "Guangxi": "China", "Guizhou": "China",
    "Hainan": "China", "Hebei": "China", "Heilongjiang": "China", "Henan": "China",
    "Hubei": "China", "Hunan": "China", "Inner Mongolia": "China", "Jiangsu": "China",
    "Jiangxi": "China", "Jilin": "China", "Liaoning": "China", "Ningxia": "China",
    "Qinghai": "China", "Shaanxi": "China", "Shandong": "China", "Shanghai": "China",
    "Shanxi": "China", "Sichuan": "China", "Tianjin": "China", "Tibet": "China",
    "Xinjiang": "China", "Yunnan": "China", "Zhejiang": "China",
    "Tianjn": "China", "Shannxi": "China", "Xizang": "China",
    "Wuhan": "China", "Hangzhou": "China", "Guangzhou": "China", "Shenzhen": "China",
    "Weifang": "China", "Shaoxing": "China", "Nanchang": "China", "Nan Chang": "China",
    "Shangrao": "China", "Lu'an": "China", "Lishui": "China", "Harbin": "China",
    "Foshan": "China", "Fuzhou": "China", "Pingxiang": "China", "Qingdao": "China",
    "Yichun": "China", "Jian": "China", "Jining": "China", "Jiujiang": "China",
    "Xinyu": "China", "Kashgar": "China", "Urumqi": "China", "Yingtan": "China",
    "Ganzhou": "China", "Jingzhou": "China", "Changde": "China", "Changzhou": "China",
    "Shulan": "China", "Tianmen": "China", "Chengdu": "China", "Nanjing": "China",
    "Wenzhou": "China",
}

# 非国家名 (动物/env/洲/占位符等, 小写); 毒株名第2段落在此集合时回退解析
JUNK_PLACE = {
    "africa", "country", "europe", "pdl", "tadarida brasiliensis", "a",
    "armadillo", "bat", "binturong", "canine", "cat", "coati", "deer",
    "dog", "env", "ferret", "fishing cat", "gorilla", "hamster", "hippo",
    "hyena", "leopard", "lion", "mink", "mouse", "mule deer", "otter",
    "puma", "snow leopard", "syrian hamster", "tiger", "pangolin", "monkey",
    "bird", "lynx", "northern greater galago", "unknown", "na",
}

# 规范地理名集合 (现有目录名 ∪ CANON 目标名), 用于回退解析时的合法性校验
GEO_PLACES = {
    "Afghanistan", "Albania", "Algeria", "American Samoa",
    "Andorra", "Angola", "Anguilla", "Antigua and Barbuda",
    "Argentina", "Armenia", "Aruba", "Australia",
    "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Barbados", "Belarus", "Belgium",
    "Belize", "Benin", "Bermuda", "Bhutan",
    "Bolivia", "Bonaire", "Bosnia and Herzegovina", "Botswana",
    "Brazil", "British Virgin Islands", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
    "Cameroon", "Canada", "Canary Islands", "Cayman Islands",
    "Central African Republic", "Chad", "Chile", "China",
    "Colombia", "Comoros", "Congo", "Cook Islands",
    "Costa Rica", "Cote d'Ivoire", "Crimea", "Croatia",
    "Cuba", "Curacao", "Cyprus", "Czechia",
    "Democratic Republic of Congo", "Denmark", "Djibouti", "Dominica",
    "Dominican Republic", "Ecuador", "Egypt", "El Salvador",
    "Equatorial Guinea", "Estonia", "Eswatini", "Ethiopia",
    "Faroe Islands", "Fiji", "Finland", "France",
    "French Guiana", "French Polynesia", "Gabon", "Gambia",
    "Georgia", "Germany", "Ghana", "Gibraltar",
    "Greece", "Grenada", "Guadeloupe", "Guam",
    "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Honduras", "Hungary", "Iceland",
    "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Jamaica",
    "Japan", "Jordan", "Kazakhstan", "Kenya",
    "Kiribati", "Kosovo", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho",
    "Liberia", "Libya", "Liechtenstein", "Lithuania",
    "Luxembourg", "Madagascar", "Malawi", "Malaysia",
    "Maldives", "Mali", "Malta", "Marshall Islands",
    "Martinique", "Mauritania", "Mauritius", "Mayotte",
    "Mexico", "Micronesia", "Moldova", "Monaco",
    "Mongolia", "Montenegro", "Montserrat", "Morocco",
    "Mozambique", "Myanmar", "Namibia", "Nepal",
    "Netherlands", "New Caledonia", "New Zealand", "Nicaragua",
    "Niger", "Nigeria", "Niue", "North Korea",
    "North Macedonia", "Northern Mariana Islands", "Norway", "Oman",
    "Pakistan", "Palau", "Palestine", "Panama",
    "Papua New Guinea", "Paraguay", "Peru", "Philippines",
    "Poland", "Portugal", "Puerto Rico", "Qatar",
    "Reunion", "Romania", "Russia", "Rwanda",
    "Saint Barthelemy", "Saint Kitts and Nevis", "Saint Lucia", "Saint Martin",
    "Saint Vincent and the Grenadines", "Samoa", "Sao Tome and Principe", "Saudi Arabia",
    "Senegal", "Serbia", "Seychelles", "Sierra Leone",
    "Singapore", "Sint Eustatius", "Sint Maarten", "Slovakia",
    "Slovenia", "Solomon Islands", "Somalia", "South Africa",
    "South Korea", "South Sudan", "Spain", "Sri Lanka",
    "Sudan", "Suriname", "Sweden", "Switzerland",
    "Syria", "Tanzania", "Thailand", "Timor",
    "Togo", "Tonga", "Trinidad and Tobago", "Tunisia",
    "Turkey", "Turks and Caicos Islands", "U.S. Virgin Islands", "Uganda",
    "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
    "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela",
    "Vietnam", "Wallis and Futuna", "Yemen", "Zambia",
    "Zimbabwe",
}


def finalize():
    import hashlib, shutil, json
    dirs = sorted(os.listdir(OUT))
    groups = defaultdict(list)
    for d in dirs:
        if not os.path.isdir(os.path.join(OUT, d)):
            continue
        groups[CANON.get(d, d)].append(d)
    merged = {k: v for k, v in groups.items() if len(v) > 1 or v[0] != k}
    for canon, raws in merged.items():
        tgt = os.path.join(OUT, canon)
        os.makedirs(tgt, exist_ok=True)
        for r in raws:
            if r == canon:
                continue
            for fn in os.listdir(os.path.join(OUT, r)):
                src_f = os.path.join(OUT, r, fn)
                dst_f = os.path.join(tgt, fn)
                if not os.path.exists(dst_f):
                    shutil.move(src_f, dst_f)
                else:
                    # 合并并去重 (按序列 md5)
                    seen = set()
                    outrecs = []
                    for fp in (dst_f, src_f):
                        h = None
                        with open(fp, "rb") as rd:
                            for line in rd:
                                if line.startswith(b">"):
                                    h = line
                                else:
                                    s = line.strip()
                                    if not s:
                                        continue
                                    m = hashlib.md5(s).digest()
                                    if m not in seen:
                                        seen.add(m)
                                        outrecs.append((h, s))
                    with open(dst_f, "wb") as w:
                        for h, s in outrecs:
                            w.write(h if h.endswith(b"\n") else h + b"\n")
                            w.write(s + b"\n")
                    os.remove(src_f)
            shutil.rmtree(os.path.join(OUT, r))
    # 报告
    rep = {}
    total_files = total_recs = 0
    for d in sorted(os.listdir(OUT)):
        dp = os.path.join(OUT, d)
        if not os.path.isdir(dp):
            continue
        n = 0
        for fn in os.listdir(dp):
            if fn.endswith(".fasta"):
                n += 1
                with open(os.path.join(dp, fn), "rb") as rd:
                    total_recs += sum(1 for ln in rd if ln.startswith(b">"))
        rep[d] = n
        total_files += n
    json.dump(dict(countries=rep, total_files=total_files, total_records=total_recs,
                   merged=merged),
              open("extract_spike_report.json", "w"), ensure_ascii=False, indent=1)
    print("finalize done. merged groups:", {k: v for k, v in merged.items()})
    print("countries:", len(rep), "files:", total_files, "records:", total_recs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["chunk", "finalize"])
    ap.add_argument("--i", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    if a.mode == "chunk":
        run_chunk(a.i, a.n)
    else:
        finalize()
