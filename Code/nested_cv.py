# -*- coding: utf-8 -*-
r"""
巢狀交叉驗證（論文 5.1 節）

整體流程：對每個資料版本、每個比較組、每一個外層折（3 次重複 × 5 折）做三件事
  步驟 1  內層選參：每種 λ 組合在 3 個內層折上各訓練 MAX_EPOCHS 個 epoch，
          記錄每個 epoch 的選參準則，3 折取平均，找出準則最小的「λ 組合 + epoch 數」
          選參準則 = (乾淨 MAE + 單段受訓練用干擾後 MAE) / 2，再取 SBP 與 DBP 平均
  步驟 2  重新訓練：用選好的 λ，在整個外層訓練集上從頭訓練，
          學習率走與內層相同的餘弦軌跡，在選定的 epoch 停下（不看測試集）
  步驟 3  預測：對外層測試集做乾淨預測，存下預測、融合權重、可靠度分數與模型

各比較組要選的參數：B1、B2 只選 epoch；B3 選 λ_r；B4 選 λ_c；B5 選 λ_r 與 λ_c
其他資料版本（M1、M2a、M2b）直接沿用 M3 每一折選出的 λ 與 epoch，只做步驟 2、3

用法：
  1. QUICK_TEST = True，按 F5 試跑（每個版本、每個模型只跑 1 折、3 個 epoch，約幾分鐘）
  2. QUICK_TEST = False，按 F5 正式執行，依 PLAN 依序跑完四個版本（約一晚）
  3. 可隨時中斷；再按 F5 會跳過已完成的折，從中斷處繼續
  執行中可打開 cv\執行紀錄.txt 查看進度與預估剩餘時間

需先執行：preprocess.py、folds.py、baselines.py
輸出：ppgbp_work\cv\<版本>\（試跑在 cv_quick\）
  preds_<模型>.csv       每位受試者在每次重複的預測、融合權重 w_1…、可靠度分數 s_1…
  selection_<模型>.csv   每一折選出的 λ_r、λ_c、epoch 數
  summary.csv            乾淨資料的 MAE、RMSE、偏差，與 B−1、B-demo 並列
  models\                每一折訓練好的模型（干擾測試用）
  parts\                 每一折的暫存結果（中斷續跑用）
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ======================= 設定 =======================
WORK_DIR = r"D:\PPG-BP Database\解壓資料\ppgbp_work"

QUICK_TEST = False                 # True：試跑；False：正式執行

# 執行計畫：依序執行。每一項 = (資料版本, 比較組, 沿用哪個版本的參數；None = 自己做內層選參)
# M3 必須排第一，其他版本才有參數可以沿用。已完成的比較組會自動跳過。
PLAN = [
    ("M3_dedup",      ["B1", "B2", "B3", "B4", "B5", "B6"], None),
    ("M1_all",        ["B2", "B5"],                   "M3_dedup"),
    ("M2a_drop4200",  ["B2", "B5"],                   "M3_dedup"),
    ("M2b_first2100", ["B2", "B5"],                   "M3_dedup"),
]

MAX_EPOCHS = 200                   # 內層每次訓練的最大 epoch 數
LAMBDA_GRID = [0.1, 0.5, 1.0]      # λ_r、λ_c 的候選值
N_JOBS = 6                         # 同時跑幾個外層折；出錯時改成 1
DEVICE_MODE = "cpu"                # "cpu"：建議；"auto"：有 GPU 就用（此時 N_JOBS 請設 1）
SEED = 0
# ====================================================

if QUICK_TEST:
    MAX_EPOCHS = 3
CV_ROOT = os.path.join(WORK_DIR, "cv_quick" if QUICK_TEST else "cv")
LOG_PATH = os.path.join(CV_ROOT, "執行紀錄.txt")


# ---------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------
def log(msg):
    """同時印在畫面上、寫進執行紀錄.txt。"""
    print(msg)
    os.makedirs(CV_ROOT, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(time.strftime("%m/%d %H:%M  ") + msg + "\n")


def hms(seconds):
    """秒數 → 「X 小時 Y 分 Z 秒」，不足 1 小時或 1 分鐘時省略前面的單位。"""
    s = int(round(seconds))
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    if h:
        return f"{h} 小時 {m} 分 {s} 秒"
    return f"{m} 分 {s} 秒" if m else f"{s} 秒"


def lambda_grid(name):
    """此比較組要試的 (λ_r, λ_c) 組合；用不到的設為 None。例如 B5 → 9 種，B2 → 1 種。"""
    from train import CONFIGS
    _, use_rank, use_cons = CONFIGS[name]
    lam_r_list = LAMBDA_GRID if use_rank else [None]
    lam_c_list = LAMBDA_GRID if use_cons else [None]
    return [(a, b) for a in lam_r_list for b in lam_c_list]


def part_path(out_dir, name, r, o):
    """某一折的暫存結果檔；檔案存在就代表這一折已經完成。"""
    return os.path.join(out_dir, "parts", f"{name}_r{r}_o{o}.csv")


def load_reuse(reuse_from, name, r, o):
    """讀出 reuse_from 版本在第 r 次重複、第 o 折選出的 λ 與 epoch。"""
    path = os.path.join(CV_ROOT, reuse_from, f"selection_{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}：請先完成 {reuse_from} 的 {name}")
    sel = pd.read_csv(path)
    row = sel[(sel.repeat == r) & (sel.outer == o)].iloc[0]
    as_float = lambda v: None if pd.isna(v) else float(v)
    return {"lambda_r": as_float(row.lambda_r), "lambda_c": as_float(row.lambda_c), "epochs": int(row.epochs)}


# ---------------------------------------------------------------
# 一個比較組 × 一個外層折（可同時在多個 CPU 核心上執行）
# ---------------------------------------------------------------
def run_one_fold(name, r, o, cfg):
    import sys as _sys
    if cfg["here"] not in _sys.path:
        _sys.path.insert(0, cfg["here"])
    import torch
    import train
    from folds import load_folds, get_split, index_of

    if cfg["device"] == "cpu":
        train.DEVICE = torch.device("cpu")
    if cfg["n_jobs"] > 1:
        torch.set_num_threads(1)          # 多個折同時跑時，每個只用 1 個執行緒，避免互相搶 CPU

    t0 = time.time()
    max_ep = cfg["max_epochs"]

    # 讀資料與這一折的切分
    d = np.load(os.path.join(cfg["work_dir"], f"prep_{cfg['version']}.npz"))
    X, mask, y, sid = d["X"], d["mask"], d["y"].astype(np.float64), d["subject_id"]
    q = d["ssqi_author"] if "ssqi_author" in d.files else np.full(mask.shape, np.nan)   # B6 用的原作者 SSQI
    if name in train.SSQI_SELECT and "ssqi_author" not in d.files:
        raise RuntimeError("prep_*.npz 沒有原作者 SSQI：請先執行第 3 版 preprocess.py，再跑 B6")
    folds = load_folds(os.path.join(cfg["work_dir"], "folds.csv"))
    test_ids, train_ids, inner = get_split(folds, r, o)
    i_te, i_tr = index_of(sid, test_ids), index_of(sid, train_ids)
    seed = cfg["seed"] + 1000 * r + 10 * o    # 同一折的各比較組用相同種子 → 相同的初始權重與干擾

    def fit(rows, epochs, lam_r, lam_c, val_rows=None, t_max=None):
        val = None if val_rows is None else (X[val_rows], mask[val_rows], y[val_rows], q[val_rows])
        return train.train_model(name, X[rows], mask[rows], y[rows], epochs, val=val, seed=seed,
                                 lambda_r=lam_r or 0.0, lambda_c=lam_c or 0.0, verbose=False, t_max=t_max,
                                 ssqi=q[rows])

    # 步驟 1：內層選參（其他版本則直接沿用）
    if cfg["reuse"] is not None:
        lam_r, lam_c, best_ep = cfg["reuse"]["lambda_r"], cfg["reuse"]["lambda_c"], cfg["reuse"]["epochs"]
        best_crit = np.nan
    else:
        best_crit, lam_r, lam_c, best_ep = np.inf, None, None, None
        for a, b in cfg["grid"]:
            curves = []
            for inner_tr, inner_va in inner:
                _, _, hist = fit(index_of(sid, inner_tr), max_ep, a, b, val_rows=index_of(sid, inner_va))
                curves.append([h["criterion"] for h in hist])
            mean_curve = np.mean(curves, axis=0)          # 3 個內層折的平均準則，每個 epoch 一個值
            k = int(np.argmin(mean_curve))
            if mean_curve[k] < best_crit:
                best_crit, lam_r, lam_c, best_ep = float(mean_curve[k]), a, b, k + 1

    # 步驟 2：在整個外層訓練集上重新訓練
    model, norm, _ = fit(i_tr, best_ep, lam_r, lam_c, t_max=max_ep)

    # 步驟 3：預測外層測試集
    if name in train.SSQI_SELECT:                       # B6：只用原作者 SSQI 最高的一段
        X_te, M_te, chosen = train.select_best(X[i_te], mask[i_te], q[i_te])
    else:
        X_te, M_te, chosen = X[i_te], mask[i_te], None
    y_hat, w, s = train.predict(model, norm, X_te, M_te)
    part = pd.DataFrame({"repeat": r, "outer": o, "subject_id": sid[i_te],
                         "sbp_true": y[i_te, 0], "dbp_true": y[i_te, 1],
                         "sbp_pred": y_hat[:, 0], "dbp_pred": y_hat[:, 1]})
    if chosen is not None:
        part["chosen_seg"] = chosen
    for j in range(w.shape[1]):
        part[f"w_{j + 1}"] = np.where(M_te[:, j], w[:, j], np.nan)       # 空位不記錄
        part[f"s_{j + 1}"] = np.where(M_te[:, j], s[:, j], np.nan)
    minutes = (time.time() - t0) / 60
    for key, val in (("lambda_r", lam_r), ("lambda_c", lam_c), ("epochs", best_ep),
                     ("inner_criterion", best_crit), ("minutes", minutes)):
        part[f"sel_{key}"] = val

    # 存檔：先存模型，最後才存暫存結果（暫存檔存在 = 這一折完成）
    out_dir = cfg["out_dir"]
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "fusion": train.CONFIGS[name][0],
                "mu": norm[0], "sd": norm[1], "lambda_r": lam_r, "lambda_c": lam_c,
                "epochs": best_ep, "max_epochs": max_ep, "version": cfg["version"], "test_ids": test_ids},
               os.path.join(out_dir, "models", f"{name}_r{r}_o{o}.pt"))
    part.to_csv(part_path(out_dir, name, r, o), index=False)
    return {"model": name, "repeat": r, "outer": o, "lambda_r": lam_r, "lambda_c": lam_c,
            "epochs": best_ep, "minutes": minutes}


# ---------------------------------------------------------------
# 一個資料版本
# ---------------------------------------------------------------
def run_version(version, models, reuse_from=None):
    """跑完一個資料版本的所有比較組與折，最後彙整。回傳是否全部完成。"""
    from folds import N_REPEATS, N_OUTER
    from baselines import summarize

    out_dir = os.path.join(CV_ROOT, version)
    repeats = [0] if QUICK_TEST else list(range(N_REPEATS))
    outers = [0] if QUICK_TEST else list(range(N_OUTER))
    n_folds = len(repeats) * len(outers)
    cfg_base = dict(here=HERE, work_dir=WORK_DIR, version=version, out_dir=out_dir,
                    max_epochs=MAX_EPOCHS, seed=SEED, device=DEVICE_MODE, n_jobs=N_JOBS)

    # 列出還沒完成的折
    tasks, n_done = [], 0
    for name in models:
        for r in repeats:
            for o in outers:
                if os.path.exists(part_path(out_dir, name, r, o)):
                    n_done += 1
                    continue
                reuse = load_reuse(reuse_from, name, r, o) if reuse_from else None
                tasks.append((name, r, o, dict(cfg_base, grid=lambda_grid(name), reuse=reuse)))

    trainings = {m: n_folds * ((1 if reuse_from else 3 * len(lambda_grid(m))) + 1) for m in models}
    log("=" * 60)
    log(f"版本 {version}{'（試跑）' if QUICK_TEST else ''}；比較組 {'、'.join(models)}；"
        + (f"沿用 {reuse_from} 的參數" if reuse_from else "內層 3 折選參"))
    log("訓練次數：" + "、".join(f"{m} {n}" for m, n in trainings.items())
        + f"；每次最多 {MAX_EPOCHS} epoch；同時執行 {N_JOBS} 個")
    log(f"已完成 {n_done} 折，本次要跑 {len(tasks)} 折")

    # 執行，每完成一折就回報進度與預估剩餘時間
    t0 = time.time()
    if tasks:
        if N_JOBS > 1:
            from joblib import Parallel, delayed
            try:
                results = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(
                    delayed(run_one_fold)(*t) for t in tasks)
            except (TypeError, ValueError):       # 舊版 joblib 不支援逐一回傳：跑完才一次回傳
                results = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(run_one_fold)(*t) for t in tasks)
        else:
            results = (run_one_fold(*t) for t in tasks)
        for k, rec in enumerate(results, 1):
            elapsed = time.time() - t0
            remain = elapsed / k * (len(tasks) - k)
            log(f"[{k}/{len(tasks)}] {rec['model']} 第 {rec['repeat'] + 1} 次重複 第 {rec['outer'] + 1} 折："
                f"λ_r {rec['lambda_r']}、λ_c {rec['lambda_c']}、{rec['epochs']} epoch；"
                f"已用 {hms(elapsed)}，此版本預估還需 {hms(remain)}")
    log(f"{version} 本次耗時 {hms(time.time() - t0)}")

    # 彙整：每個比較組合併 15 折
    rows = []
    for name in models:
        paths = [part_path(out_dir, name, r, o) for r in repeats for o in outers]
        if not all(os.path.exists(p) for p in paths):
            log(f"{name}：尚有未完成的折，暫不彙整")
            continue
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        sel_cols = [c for c in df.columns if c.startswith("sel_")]
        sel = df.groupby(["repeat", "outer"])[sel_cols].first().reset_index()
        sel.columns = [c.replace("sel_", "") for c in sel.columns]
        sel.to_csv(os.path.join(out_dir, f"selection_{name}.csv"), index=False)
        df.drop(columns=sel_cols).to_csv(os.path.join(out_dir, f"preds_{name}.csv"), index=False)
        rows.append(summarize(df, name))
        log(f"{name} 選出的參數：λ_r {sel.lambda_r.value_counts(dropna=False).to_dict()}、"
            f"λ_c {sel.lambda_c.value_counts(dropna=False).to_dict()}、"
            f"epoch 中位數 {sel.epochs.median():.0f}（{sel.epochs.min()}–{sel.epochs.max()}）")

    # 結果表：與 B−1、B-demo 並列
    if rows:
        table = pd.DataFrame(rows)
        base_path = os.path.join(WORK_DIR, "baselines_summary.csv")
        if os.path.exists(base_path):
            table = pd.concat([pd.read_csv(base_path), table], ignore_index=True)
        table.to_csv(os.path.join(out_dir, "summary.csv"), index=False, encoding="utf-8-sig")
        log(f"{version} 乾淨資料結果（3 次重複的平均 ± 標準差，mmHg）"
            + ("；試跑只有 1 折，數字不具意義" if QUICK_TEST else ""))
        log(f"{'':8s}{'SBP MAE':>16s}{'SBP RMSE':>16s}{'DBP MAE':>16s}{'DBP RMSE':>16s}")
        for _, s in table.iterrows():
            sd = lambda c: 0.0 if pd.isna(s[c + "_sd"]) else s[c + "_sd"]
            cells = [f"{s[c]:.2f} ± {sd(c):.2f}" for c in ("sbp_mae", "sbp_rmse", "dbp_mae", "dbp_rmse")]
            log(f"{s['model']:8s}" + "".join(f"{c:>16s}" for c in cells))
    return len(rows) == len(models)


# ---------------------------------------------------------------
# 主程式：依 PLAN 依序執行
# ---------------------------------------------------------------
if __name__ == "__main__":
    t_all = time.time()
    for version, models, reuse_from in PLAN:
        if not run_version(version, models, reuse_from):
            log(f"{version} 未全部完成，停止後續版本；再按 F5 會從中斷處繼續")
            break
    log(f"全部計畫耗時 {hms(time.time() - t_all)}；結果在 {CV_ROOT}")
