# -*- coding: utf-8 -*-
"""search_forecast_v2.py — v2 月熵预测两阶段搜索
Stage A: target_norm{global,country,window} × K{3,6} × arch{res2v2,resv2,gru}
         × delta{0,1} 共 36 配置 × (full + ent_only 消融), seed=7, --calibrate
         过滤: full pooled R2>=0.80 且表征贡献>=0.30; 按逐国 R2>=0.8 占比排序
Stage B: top5 × seeds{7,42,123,2024,31337} (无逐run校准) + ent_only 同种子,
         5 种子集成 + 集成级逐国校准后评估
Stage C: 若最优未达 0.75, 对 half_life{12,24}/balanced/cal_lambda{3,30}/
         dropout{0,0.2}/ent_dropout{0.2} 逐一贪心 (3 种子 full)
日志: forecast_v2_search_log.jsonl; 最优配置: forecast_v2_best_config.json
运行: G:/Anaconda3/envs/esm/python.exe search_forecast_v2.py [--stage A|B|C]
"""
import argparse, itertools, json, os, subprocess, sys

import numpy as np
import torch

import train_entropy_forecast_v2 as T

PY = sys.executable
LOG = "forecast_v2_search_log.jsonl"
BEST_CFG = "forecast_v2_best_config.json"
TOP5 = "forecast_v2_stageA_top5.json"
ENT_DIR = "entropy_monthly_smoothed"
COUNTS_FILE = None
COUNTS = None
ROLL = dict(lam=10.0, win=6)
SEEDS_B = [7, 42, 123, 2024, 31337]
SEEDS_C = [7, 42, 123]


def log(ev):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def cfg_name(stage, cfg, mode, seed):
    """规范命名: 仅当取值不同于默认时才附加修饰 (dropout 默认 0.1,
    ent_dropout 默认 0, half_life 默认 0, cal_lambda 默认 10)"""
    d = "_d" if cfg.get("delta") else ""
    nm = (f"{stage}_{cfg['model']}_{cfg['target_norm']}_K{cfg['K']}{d}"
          f"_h{cfg.get('hidden', 128)}")
    if cfg.get("half_life", 0.0):
        nm += f"_hl{cfg['half_life']:g}"
    if cfg.get("dropout", 0.1) != 0.1:
        nm += f"_dr{cfg['dropout']:g}"
    if cfg.get("ent_dropout", 0.0):
        nm += f"_ed{cfg['ent_dropout']:g}"
    if cfg.get("balanced_sampler"):
        nm += "_bal"
    if cfg.get("seq_weight", "none") != "none":
        nm += f"_sw{cfg['seq_weight']}"
    if cfg.get("cal_lambda", 10.0) != 10.0:
        nm += f"_cl{cfg['cal_lambda']:g}"
    return f"{nm}_{mode}_s{seed}"


def _gbm_fit(stage, cfg, seed):
    """HistGradientBoosting 基线: 同归一化/校准口径, full+ent_only 一并拟合,
    预测结果存 {name}_preds.npz 供集成评估复用"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    nm_f = cfg_name(stage, cfg, "full", seed)
    nm_e = cfg_name(stage, cfg, "ent_only", seed)
    out = {}
    store = {}
    for tag, nm in (("full", nm_f), ("ent_only", nm_e)):
        D = T.prepare_data(cfg["K"], cfg["target_norm"], cfg["delta"], tag,
                           ent_dir=ENT_DIR)
        X = D["Xf"].reshape(len(D["Xf"]), -1)
        mdl = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=6,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=seed)
        mdl.fit(X[D["tr"]], D["yz"][D["tr"]])
        pz = mdl.predict(X)
        pred = T.Norm.inverse(pz.astype(np.float64), D["mu"], D["sd"])
        np.savez(os.path.join(T.RUNS, f"{nm}_preds.npz"),
                 pred_all=pred, y=D["y"], cs=D["cs"], ts=D["ts"],
                 countries=np.array(D["countries"]),
                 K=cfg["K"], target_norm=cfg["target_norm"],
                 delta=cfg["delta"])
        store[tag] = D
        m = T.summarize(pred[D["va"]], D["y"][D["va"]], D["cs"][D["va"]])
        cal = T.fit_calibration(pred[D["tr"]], D["y"][D["tr"]],
                                D["cs"][D["tr"]], D["countries"],
                                cfg.get("cal_lambda", 10.0))
        pc = T.apply_calibration(pred[D["va"]], D["cs"][D["va"]],
                                 D["countries"], cal)
        out[tag] = (m, T.summarize(pc, D["y"][D["va"]], D["cs"][D["va"]]))
    res = dict(name=nm_f, model="gbm", val=out["full"][0],
               val_calibrated=out["full"][1], val_ent=out["ent_only"][0],
               val_ent_calibrated=out["ent_only"][1])
    mj = os.path.join(T.RUNS, f"{nm_f}_metrics.json")
    json.dump(res, open(mj, "w"), ensure_ascii=False, indent=1)
    log(dict(stage=stage, kind="run", cfg=cfg, mode="gbm", seed=seed,
             name=nm_f, val=out["full"][0], val_calibrated=out["full"][1]))
    return res


def run_once(stage, cfg, mode, seed, epochs=300):
    """训练单轮 (有缓存); 返回 metrics dict"""
    if cfg["model"] == "gbm":
        nm = cfg_name(stage, cfg, mode, seed)
        if mode == "ent_only":
            npz = os.path.join(T.RUNS, f"{nm}_preds.npz")
            if not os.path.exists(npz):
                _gbm_fit(stage, cfg, seed)
            mj = os.path.join(T.RUNS,
                              f"{cfg_name(stage, cfg, 'full', seed)}_metrics.json")
            r = json.load(open(mj, encoding="utf-8"))
            return dict(name=nm, val=r["val_ent"],
                        val_calibrated=r["val_ent_calibrated"])
        mj = os.path.join(T.RUNS, f"{nm}_metrics.json")
        if os.path.exists(mj):
            return json.load(open(mj, encoding="utf-8"))
        return _gbm_fit(stage, cfg, seed)
    name = cfg_name(stage, cfg, mode, seed)
    mj = os.path.join(T.RUNS, f"{name}_metrics.json")
    if os.path.exists(mj):
        return json.load(open(mj, encoding="utf-8"))
    cmd = [PY, "train_entropy_forecast_v2.py", "--model", cfg["model"],
           "--mode", mode, "--target_norm", cfg["target_norm"],
           "--ent_dir", ENT_DIR, "--runs", T.RUNS,
           "--counts_file", COUNTS_FILE or "seqcounts_monthly.csv",
           "--seq_weight", cfg.get("seq_weight", "none"),
           "--K", str(cfg["K"]), "--hidden", str(cfg.get("hidden", 128)),
           "--layers", str(cfg.get("layers", 1)),
           "--dropout", str(cfg.get("dropout", 0.1)),
           "--ent_dropout", str(cfg.get("ent_dropout", 0.0)),
           "--half_life", str(cfg.get("half_life", 0.0)),
           "--cal_lambda", str(cfg.get("cal_lambda", 10.0)),
           "--epochs", str(epochs), "--seed", str(seed), "--name", name]
    if cfg.get("delta"):
        cmd.append("--delta")
    if cfg.get("balanced_sampler"):
        cmd.append("--balanced_sampler")
    if cfg.get("calibrate"):
        cmd.append("--calibrate")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0 or not os.path.exists(mj):
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"run failed: {name}")
    res = json.load(open(mj, encoding="utf-8"))
    log(dict(stage=stage, kind="run", cfg=cfg, mode=mode, seed=seed,
             name=name, val=res["val"],
             val_calibrated=res.get("val_calibrated")))
    return res


# ------------------------------------------------------------ 集成评估
_DATA_CACHE = {}


def _data(ck):
    key = (ck["K"], ck["target_norm"], ck["delta"], ck["mode"],
           ck.get("ent_dir", "entropy_monthly_smoothed"))
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = T.prepare_data(*key[:4], ent_dir=key[4])
    return _DATA_CACHE[key]


def predict_run(name):
    """单 run 对全部样本的原始尺度预测 (N,), 以及数据 dict"""
    npz_path = os.path.join(T.RUNS, f"{name}_preds.npz")
    if os.path.exists(npz_path):          # GBM 等表格模型: 预测已持久化
        z = np.load(npz_path, allow_pickle=True)
        ts = z["ts"].astype(str)
        D = dict(y=z["y"], cs=z["cs"], ts=ts,
                 countries=[str(c) for c in z["countries"]],
                 tr=ts <= T.TRAIN_END, va=ts > T.TRAIN_END)
        return z["pred_all"].astype(np.float64), D
    ck = torch.load(os.path.join(T.RUNS, f"{name}.pt"), map_location="cpu",
                    weights_only=False)
    D = _data(ck)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = T.make_model(ck["model"], ck["K"], ck["hidden"], ck["layers"],
                       ck["dropout"], ck["F"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    with torch.no_grad():
        pz = net(torch.from_numpy(D["Xf"]).to(device)).cpu().numpy()
    return T.Norm.inverse(pz.astype(np.float64), D["mu"], D["sd"]), D


def ensemble_eval(cfg, stage, seeds, cal_lambda=10.0):
    """n 种子集成: 原始尺度均值 -> 静态逐国校准 + 滚动在线校准 (严格因果);
    同配置 ent_only 集成核算贡献; 若 COUNTS 可用, 另报剔除<10序列国后的
    80 国口径指标 (f80 后缀)。返回 dict。"""
    names_f = [cfg_name(stage, cfg, "full", s) for s in seeds]
    names_e = [cfg_name(stage, cfg, "ent_only", s) for s in seeds]
    preds_f, D = zip(*(predict_run(nm) for nm in names_f))
    pf = np.mean(np.stack(preds_f), axis=0)
    D0 = D[0]
    y, cs, ts, countries, tr, va = (D0["y"], D0["cs"], D0["ts"],
                                    D0["countries"], D0["tr"], D0["va"])
    preds_e = np.mean(np.stack([predict_run(nm)[0] for nm in names_e]), axis=0)

    m_full = T.summarize(pf[va], y[va], cs[va])
    m_ent = T.summarize(preds_e[va], y[va], cs[va])
    contrib = (m_ent["mse"] - m_full["mse"]) / m_ent["mse"]

    cal = T.fit_calibration(pf[tr], y[tr], cs[tr], countries, cal_lambda)
    pf_c = T.apply_calibration(pf[va], cs[va], countries, cal)
    m_full_c = T.summarize(pf_c, y[va], cs[va])
    # 滚动在线校准 (全体样本上滚动, 取验证段)
    pf_r = T.rolling_calibrate(pf, y, cs, ts, countries,
                               lam=ROLL["lam"], win=ROLL["win"])
    pe_r = T.rolling_calibrate(preds_e, y, cs, ts, countries,
                               lam=ROLL["lam"], win=ROLL["win"])
    m_full_r = T.summarize(pf_r[va], y[va], cs[va])
    m_ent_r = T.summarize(pe_r[va], y[va], cs[va])
    contrib_r = (m_ent_r["mse"] - m_full_r["mse"]) / m_ent_r["mse"]

    out = dict(cfg=cfg, seeds=seeds, cal_lambda=cal_lambda,
               full=m_full, full_cal=m_full_c, ent=m_ent,
               full_roll=m_full_r, ent_roll=m_ent_r,
               contrib=round(float(contrib), 4),
               contrib_roll=round(float(contrib_r), 4),
               calibration=cal, runs_full=names_f, runs_ent=names_e)
    if COUNTS is not None:
        keep = T.filtered_country_set(D0, COUNTS, thr=10.0)
        m_f = T.summarize_on(pf[va], y[va], cs[va], keep)
        m_e = T.summarize_on(preds_e[va], y[va], cs[va], keep)
        m_fc = T.summarize_on(pf_c, y[va], cs[va], keep)
        m_fr = T.summarize_on(pf_r[va], y[va], cs[va], keep)
        m_er = T.summarize_on(pe_r[va], y[va], cs[va], keep)
        out.update(n_keep=len(keep),
                   f80_full=m_f, f80_full_cal=m_fc, f80_full_roll=m_fr,
                   f80_ent=m_e, f80_ent_roll=m_er,
                   contrib_f80=round(float((m_e["mse"] - m_f["mse"])
                                           / m_e["mse"]), 4),
                   contrib_f80_roll=round(float((m_er["mse"] - m_fr["mse"])
                                                / m_er["mse"]), 4))
    log(dict(stage=stage, kind="ensemble",
             **{k: v for k, v in out.items() if k != "calibration"}))
    return out


def frac_of(m):
    return m["frac_countries_r2_ge_08"] or 0.0


def sel_frac(ev):
    """选型主指标: 滚动校准+80 国口径 frac (无 counts 时回退 静态校准)"""
    if "f80_full_roll" in ev:
        return frac_of(ev["f80_full_roll"])
    return frac_of(ev.get("full_cal") or ev["full"])


def sel_r2(ev):
    if "f80_full" in ev:
        return ev["f80_full"]["r2"]
    return ev["full"]["r2"]


def sel_contrib(ev):
    return ev.get("contrib_f80", ev["contrib"])


# ------------------------------------------------------------ Stage A
def _eval_a_grid(grid, stage="A"):
    rows = []
    for i, cfg in enumerate(grid):
        rf = run_once(stage, {**cfg, "calibrate": True}, "full", 7)
        re = run_once(stage, {**cfg, "calibrate": True}, "ent_only", 7)
        contrib = (re["val"]["mse"] - rf["val"]["mse"]) / re["val"]["mse"]
        row = dict(cfg=cfg, contrib=round(contrib, 4),
                   full=rf["val"], full_cal=rf.get("val_calibrated"))
        rows.append(row)
        log(dict(stage=stage, kind="config", **row))
        f = rf.get("val_calibrated") or rf["val"]
        print(f"{stage} {i+1:02d}/{len(grid)} {cfg['model']:7s} "
              f"{cfg['target_norm']:7s} K{cfg['K']} d{int(cfg['delta'])}"
              f"{(' h' + str(cfg['hidden'])) if 'hidden' in cfg else ''} "
              f"frac={frac_of(f):.3f} r2={rf['val']['r2']:.3f} "
              f"contrib={contrib:.3f}", flush=True)
    return rows


def stage_a2():
    """扩展扫描: resv2 × norm{global,country} × K{2,4,5} × hidden{64,128,256}
    × delta{0,1} + GBM 基线(norm{global,country} × K{2,3,4} × delta{0,1});
    与 Stage A 结果合并重选 top5"""
    grid = []
    for norm, K, h, d in itertools.product(
            ["global", "country"], [2, 4, 5], [64, 128, 256], [False, True]):
        grid.append(dict(model="resv2", target_norm=norm, K=K, hidden=h,
                         delta=d))
    rows = _eval_a_grid(grid, "A2")
    for norm, K, d in itertools.product(["global", "country"], [2, 3, 4],
                                        [False, True]):
        cfg = dict(model="gbm", target_norm=norm, K=K, delta=d)
        try:
            rf = run_once("A2", {**cfg, "calibrate": True}, "full", 7)
            re = run_once("A2", {**cfg, "calibrate": True}, "ent_only", 7)
        except Exception as e:
            print("gbm failed", cfg, e)
            continue
        contrib = (re["val"]["mse"] - rf["val"]["mse"]) / re["val"]["mse"]
        rows.append(dict(cfg=cfg, contrib=round(contrib, 4),
                         full=rf["val"], full_cal=rf.get("val_calibrated")))
        log(dict(stage="A2", kind="config", cfg=cfg,
                 contrib=round(contrib, 4), full=rf["val"],
                 full_cal=rf.get("val_calibrated")))
        f = rf.get("val_calibrated") or rf["val"]
        print(f"A2 gbm {norm:7s} K{K} d{int(d)} frac={frac_of(f):.3f} "
              f"r2={rf['val']['r2']:.3f} contrib={contrib:.3f}", flush=True)
    # 与 Stage A 结果合并重选
    prev = [json.loads(l) for l in open(LOG, encoding="utf-8")
            if '"stage": "A"' in l and '"kind": "config"' in l] \
        if os.path.exists(LOG) else []
    allrows = rows + prev
    ok = [r for r in allrows
          if r["full"]["r2"] >= 0.80 and r["contrib"] >= 0.30]
    ok.sort(key=lambda r: (frac_of(r["full_cal"] or r["full"]),
                           r["full"]["r2"]), reverse=True)
    print(f"\n== A2 merged top ({len(ok)} pass filters) ==")
    for r in ok[:10]:
        f = r["full_cal"] or r["full"]
        print(f"  {r['cfg']} frac={frac_of(f):.3f} r2={r['full']['r2']:.4f} "
              f"contrib={r['contrib']}")
    json.dump([r["cfg"] for r in ok[:5]],
              open(TOP5, "w", encoding="utf-8"), ensure_ascii=False)
    return ok[:5]
    grid = []
    for norm, K, arch, d in itertools.product(
            ["global", "country", "window"], [3, 6],
            ["res2v2", "resv2", "gru"], [False, True]):
        grid.append(dict(model=arch, target_norm=norm, K=K, delta=d))
    rows = _eval_a_grid(grid, "A")
    ok = [r for r in rows if r["full"]["r2"] >= 0.80 and r["contrib"] >= 0.30]
    ok.sort(key=lambda r: (frac_of(r["full_cal"] or r["full"]),
                           r["full"]["r2"]), reverse=True)
    print("\n== Stage A top (filtered pooled>=0.80 & contrib>=0.30) ==")
    for r in ok[:10]:
        f = r["full_cal"] or r["full"]
        print(f"  {r['cfg']} frac={frac_of(f):.3f} r2={r['full']['r2']:.4f} "
              f"contrib={r['contrib']}")
    json.dump([r["cfg"] for r in ok[:5]],
              open(TOP5, "w", encoding="utf-8"), ensure_ascii=False)
    return ok[:5]


def stage_a3():
    """尾窗口径扫描 (v3): resv2+res2v2 × norm{global,country} × K{2,3,4,5}
    × delta{0,1} × h{64,128} = 64 配置"""
    grid = []
    for arch, norm, K, d, h in __import__("itertools").product(
            ["resv2", "res2v2"], ["global", "country"], [2, 3, 4, 5],
            [False, True], [64, 128]):
        grid.append(dict(model=arch, target_norm=norm, K=K, hidden=h,
                         delta=d))
    rows = _eval_a_grid(grid, "A3")
    ok = [r for r in rows
          if r["full"]["r2"] >= 0.80 and r["contrib"] >= 0.30]
    if not ok:
        print("A3 WARNING: 无配置通过双过滤, 回退为按校准 frac 排序 (尾窗"
              "口径下护栏不可达, 验收时将如实报告)")
        ok = list(rows)
    ok.sort(key=lambda r: (frac_of(r["full_cal"] or r["full"]),
                           r["full"]["r2"]), reverse=True)
    print(f"\n== A3 top ({len(ok)} pass filters) ==")
    for r in ok[:10]:
        f = r["full_cal"] or r["full"]
        print(f"  {r['cfg']} frac={frac_of(f):.3f} r2={r['full']['r2']:.4f} "
              f"contrib={r['contrib']}")
    json.dump([r["cfg"] for r in ok[:5]],
              open(TOP5, "w", encoding="utf-8"), ensure_ascii=False)
    return ok[:5]


def stage_a4():
    """v4 扫描 (序列数加权): resv2 × norm{global,country} × K{3,4,5}
    × delta{0,1} × seq_weight{n50,sqrt,log} = 36 配置"""
    grid = []
    for norm, K, d, w in itertools.product(
            ["global", "country"], [3, 4, 5], [False, True],
            ["n50", "sqrt", "log"]):
        grid.append(dict(model="resv2", target_norm=norm, K=K, hidden=128,
                         delta=d, seq_weight=w))
    rows = _eval_a_grid(grid, "A4")
    ok = [r for r in rows
          if r["full"]["r2"] >= 0.80 and r["contrib"] >= 0.30]
    if not ok:
        print("A4 WARNING: 无配置通过双过滤, 回退为按校准 frac 排序")
        ok = list(rows)
    ok.sort(key=lambda r: (frac_of(r["full_cal"] or r["full"]),
                           r["full"]["r2"]), reverse=True)
    print(f"\n== A4 top ({len(ok)}) ==")
    for r in ok[:10]:
        f = r["full_cal"] or r["full"]
        print(f"  {r['cfg']} frac={frac_of(f):.3f} r2={r['full']['r2']:.4f} "
              f"contrib={r['contrib']}")
    json.dump([r["cfg"] for r in ok[:5]],
              open(TOP5, "w", encoding="utf-8"), ensure_ascii=False)
    return ok[:5]


# ------------------------------------------------------------ Stage B
def stage_b(top5=None):
    if top5 is None:
        top5 = json.load(open(TOP5, encoding="utf-8"))
    results = []
    for cfg in top5:
        for s in SEEDS_B:
            run_once("B", cfg, "full", s)
            run_once("B", cfg, "ent_only", s)
        ev = ensemble_eval(cfg, "B", SEEDS_B)
        results.append(ev)
        print(f"B {cfg['model']:7s} {cfg['target_norm']:7s} K{cfg['K']} "
              f"d{int(cfg['delta'])} sw{cfg.get('seq_weight','-')}: "
              f"sel_frac={sel_frac(ev):.3f} ens frac={frac_of(ev['full']):.3f} "
              f"r2={sel_r2(ev):.4f} contrib={sel_contrib(ev):.3f}",
              flush=True)
    results.sort(key=lambda ev: (
        sel_contrib(ev) >= 0.30 and sel_r2(ev) >= 0.80,
        sel_frac(ev), frac_of(ev["full"]), sel_r2(ev)), reverse=True)
    best = results[0]
    json.dump(dict(cfg=best["cfg"], stage="B", seeds=SEEDS_B,
                   cal_lambda=best["cal_lambda"]),
              open(BEST_CFG, "w", encoding="utf-8"), ensure_ascii=False)
    print("best:", json.dumps(best["cfg"], ensure_ascii=False),
          "sel_frac:", sel_frac(best), "contrib:", sel_contrib(best))
    return best, results


# ------------------------------------------------------------ Stage C
def stage_c(best=None):
    if best is None:
        bc = json.load(open(BEST_CFG, encoding="utf-8"))
        best_cfg = dict(bc["cfg"])
        for s in bc["seeds"]:               # 补齐缺失种子后再评估
            for mode in ("full", "ent_only"):
                run_once(bc["stage"], best_cfg, mode, s)
        best_ev = ensemble_eval(best_cfg, bc["stage"], bc["seeds"],
                                bc.get("cal_lambda", 10.0))
    else:
        best_cfg, best_ev = dict(best["cfg"]), best
    tweaks = [("half_life", 12.0), ("half_life", 24.0),
              ("balanced_sampler", True), ("cal_lambda", 3.0),
              ("cal_lambda", 30.0), ("dropout", 0.0), ("dropout", 0.2),
              ("ent_dropout", 0.2)]
    cur_frac = sel_frac(best_ev)
    print(f"C start: frac={cur_frac:.3f} cfg={best_cfg}")
    for key, val in tweaks:
        cfg = dict(best_cfg)
        cfg[key] = val
        try:
            for s in SEEDS_C:
                run_once("C", cfg, "full", s)
                run_once("C", cfg, "ent_only", s)
            ev = ensemble_eval(cfg, "C", SEEDS_C,
                               cfg.get("cal_lambda", 10.0))
        except Exception as e:
            print(f"  tweak {key}={val} failed: {e}")
            continue
        f = sel_frac(ev)
        ok = (sel_r2(ev) >= 0.80 and sel_contrib(ev) >= 0.30)
        print(f"  tweak {key}={val}: sel_frac={f:.3f} r2={sel_r2(ev):.4f} "
              f"contrib={sel_contrib(ev):.3f} {'ACCEPT' if f > cur_frac and ok else ''}",
              flush=True)
        if f > cur_frac and ok:
            best_cfg, best_ev, cur_frac = cfg, ev, f
    json.dump(dict(cfg=best_cfg, stage="C", seeds=SEEDS_B,
                   cal_lambda=best_cfg.get("cal_lambda", 10.0)),
              open(BEST_CFG, "w", encoding="utf-8"), ensure_ascii=False)
    print("C done. best cal_frac:", cur_frac, "cfg:", best_cfg)
    return best_cfg, best_ev


# ------------------------------------------------------------ Stage D
def stage_d(max_members=5, pool_size=12):
    """跨配置集成的贪心前向选择 (仅种子7预测, 无后平滑——后平滑经核查存在
    前视泄漏: m+1 月预测的输入窗包含 m 月真实熵, 已移除)。
    成员池: 逐国校准后 frac>=0.15 的配置 (排除 gbm); 逐贪心添加使
    集成校准后 frac 最大, 约束集成 pooled R2>=0.80 且贡献>=0.30。
    若优于当前 BEST_CFG, 更新 BEST_CFG 为组合描述"""
    rows = [json.loads(l) for l in open(LOG, encoding="utf-8")
            if '"kind": "config"' in l] if os.path.exists(LOG) else []
    for l in open(LOG, encoding="utf-8") if os.path.exists(LOG) else []:
        if '"kind": "ensemble"' not in l:
            continue
        r = json.loads(l)
        rows.append(dict(cfg=r["cfg"], stage=r["stage"], contrib=r["contrib"],
                         full=r["full"], full_cal=r["full_cal"]))
    uniq = {}
    for r in rows:
        if r["cfg"].get("model") == "gbm":
            continue
        key = json.dumps(r["cfg"], sort_keys=True)
        f = frac_of(r.get("full_cal") or r["full"])
        if key not in uniq or f > frac_of(uniq[key].get("full_cal")
                                        or uniq[key]["full"]):
            uniq[key] = r
    pool = [r for r in uniq.values()
            if frac_of(r.get("full_cal") or r["full"]) >= 0.15]
    pool.sort(key=lambda r: (frac_of(r["full_cal"] or r["full"]),
                             r["full"]["r2"]), reverse=True)
    pool = pool[:pool_size]
    print(f"D: pool={len(pool)} configs")
    if not pool:
        return None

    cache = {}

    def pred(r, mode):
        key = (json.dumps(r["cfg"], sort_keys=True), r["stage"], mode)
        if key not in cache:
            cache[key] = predict_run(cfg_name(r["stage"], r["cfg"], mode, 7))
        return cache[key]

    def eval_sel(sel):
        pf = np.mean(np.stack([pred(r, "full")[0] for r in sel]), axis=0)
        pe = np.mean(np.stack([pred(r, "ent_only")[0] for r in sel]), axis=0)
        D = pred(sel[0], "full")[1]
        y, cs, ts, countries, tr, va = (D["y"], D["cs"], D["ts"],
                                        D["countries"], D["tr"], D["va"])
        m_full = T.summarize(pf[va], y[va], cs[va])
        m_ent = T.summarize(pe[va], y[va], cs[va])
        contrib = (m_ent["mse"] - m_full["mse"]) / m_ent["mse"]
        pf_c = T.rolling_calibrate(pf, y, cs, ts, countries,
                                   lam=ROLL["lam"], win=ROLL["win"])[va]
        m_cal = T.summarize(pf_c, y[va], cs[va])
        frac = frac_of(m_cal)
        r2v, conv = m_full["r2"], round(float(contrib), 4)
        if COUNTS is not None:
            keep = T.filtered_country_set(D, COUNTS, thr=10.0)
            m_cf = T.summarize_on(pf_c, y[va], cs[va], keep)
            m_ff = T.summarize_on(pf[va], y[va], cs[va], keep)
            m_ef = T.summarize_on(pe[va], y[va], cs[va], keep)
            frac = frac_of(m_cf)
            r2v = m_ff["r2"]
            conv = round(float((m_ef["mse"] - m_ff["mse"]) / m_ef["mse"]), 4)
            m_cal = m_cf
        return dict(frac=frac, metrics=m_cal, r2=r2v, contrib=conv,
                    cfgs=[r["cfg"] for r in sel],
                    stages=[r["stage"] for r in sel])

    selected = []
    remaining = list(pool)
    best = None
    while remaining and len(selected) < max_members:
        cand = None
        for r in remaining:
            ev = eval_sel(selected + [r])
            ok = ev["r2"] >= 0.80 and ev["contrib"] >= 0.30
            key = (ok, ev["frac"], ev["r2"])
            if cand is None or key > cand[0]:
                cand = (key, r, ev)
        if cand is None or not cand[0][0]:
            break
        if best is not None and cand[2]["frac"] <= best["frac"]:
            break
        selected.append(cand[1])
        best = cand[2]
        remaining.remove(cand[1])
        print(f"D greedy +{cand[1]['cfg']['model']}/"
              f"{cand[1]['cfg']['target_norm']}/K{cand[1]['cfg']['K']}"
              f"{'/d' if cand[1]['cfg'].get('delta') else ''}: "
              f"frac={best['frac']:.3f} r2={best['r2']:.4f} "
              f"contrib={best['contrib']:.3f}", flush=True)
    if best is None:
        print("D: no ensemble passes filters")
        return None
    log(dict(stage="D", kind="cross_ensemble_greedy", frac=best["frac"],
             metrics=best["metrics"], contrib=best["contrib"],
             cfgs=best["cfgs"], stages=best["stages"]))

    cur = json.load(open(BEST_CFG, encoding="utf-8"))         if os.path.exists(BEST_CFG) else None
    cur_frac = -1.0
    if cur and "cfg" in cur:
        for seeds in (cur.get("seeds"), SEEDS_C):
            if not seeds:
                continue
            try:
                ev = ensemble_eval(cur["cfg"], cur["stage"], seeds,
                                   cur.get("cal_lambda", 10.0))
                cur_frac = sel_frac(ev)
                break
            except Exception:
                continue
    if best["frac"] > cur_frac and best["contrib"] >= 0.30             and best["r2"] >= 0.80:
        json.dump(dict(type="cross", cfgs=best["cfgs"], stages=best["stages"],
                       seed=7, smooth_win=0, cal_lambda=10.0),
                  open(BEST_CFG, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"D: NEW BEST cross-ensemble frac={best['frac']:.3f} "
              f"(was {cur_frac:.3f})")
    else:
        print(f"D: keep current best (frac {cur_frac:.3f} vs "
              f"{best['frac']:.3f})")
    return best


def main():
    global LOG, BEST_CFG, TOP5, ENT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "A2", "A3", "A4", "B", "C",
                                                  "D", "ABC"],
                    default="ABC")
    ap.add_argument("--ent_dir", default=ENT_DIR)
    ap.add_argument("--runs", default=T.RUNS)
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--best_cfg", default=BEST_CFG)
    ap.add_argument("--top5_file", default=TOP5)
    ap.add_argument("--counts_file", default=None)
    a = ap.parse_args()
    global COUNTS_FILE, COUNTS
    ENT_DIR = a.ent_dir
    T.RUNS = a.runs
    LOG, BEST_CFG, TOP5 = a.log, a.best_cfg, a.top5_file
    COUNTS_FILE = a.counts_file
    if a.counts_file:
        COUNTS = T.load_counts(a.counts_file)
    os.makedirs(T.RUNS, exist_ok=True)
    if a.stage in ("A", "ABC"):
        top5 = stage_a()
    elif a.stage == "A2":
        top5 = stage_a2()
    elif a.stage == "A3":
        top5 = stage_a3()
    elif a.stage == "A4":
        top5 = stage_a4()
    else:
        top5 = None
    best = None
    if a.stage in ("B", "ABC"):
        best, _ = stage_b(top5)
    if a.stage in ("C", "ABC"):
        stage_c(best)
    if a.stage in ("D", "ABC"):
        stage_d()
    print("search done.")


if __name__ == "__main__":
    main()
