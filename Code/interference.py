# -*- coding: utf-8 -*-
r"""
干擾測試（論文 5.3 節；產生 6.3、6.4、6.5 節需要的資料）

載入 nested_cv.py 存下的模型（不重新訓練），對每一折的測試者做三種測試：

  1. 單段干擾：輪流只弄壞一段，9 種干擾條件 × 10 個固定種子
       見過：高斯 20、10 dB；基線漂移 0.3σ、1.0σ
       未見：高斯 0 dB；高頻突發 100、300 ms；削波第 90、80 百分位
     干擾內容只由（種子、受試者、第幾段、干擾條件）決定，與模型無關
     → 每個模型面對完全相同的干擾，比較才公平
  2. 刪除檢驗：拿掉權重最高或最低的一段，看預測變化（權重是否真的反映片段的影響力）
  3. 重複片段測試：輸入 {x1, x1, x2}，和 {x1, x2} 比較預測變化

需先執行：nested_cv.py（正式執行完成）
用法：按 F5。可中斷，再按 F5 會跳過已完成的折。之後執行 stats.py 產生統計結果。
輸出（ppgbp_work\cv\<版本>\）
  interference_<模型>.csv  每列 = 一位受試者 × 一次重複 × 被弄壞的一段 × 一種干擾 × 一個種子
  extra_<模型>.csv         刪除檢驗與重複片段測試，每列 = 一位受試者 × 一次重複
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import skew

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ======================= 設定 =======================
WORK_DIR = r"D:\PPG-BP Database\解壓資料\ppgbp_work"
QUICK_TEST = False                 # True：測試 nested_cv 試跑的結果（cv_quick，種子只用 2 個）

PLAN = [                           # 與 nested_cv.py 相同
    ("M3_dedup",      ["B1", "B2", "B3", "B4", "B5", "B6"]),
    ("M1_all",        ["B2", "B5"]),
    ("M2a_drop4200",  ["B2", "B5"]),
    ("M2b_first2100", ["B2", "B5"]),
]
N_SEEDS = 10                       # 每種干擾條件重複的次數
BATCH = 4096                       # 每次送進模型的集合數
# ====================================================

SSQI_SELECT = {"B6"}               # 只用 SSQI 最高單段的比較組

if QUICK_TEST:
    N_SEEDS = 2
CV_ROOT = os.path.join(WORK_DIR, "cv_quick" if QUICK_TEST else "cv")


def hms(seconds):
    s = int(round(seconds))
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h} 小時 {m} 分 {s} 秒" if h else (f"{m} 分 {s} 秒" if m else f"{s} 秒")


def conditions():
    """9 種干擾條件：(家族, 強度, 家族編號, 強度編號)。"""
    from degrade import FAMILIES
    out = []
    for fi, (fam, (levels, _, _)) in enumerate(FAMILIES.items()):
        for li, lv in enumerate(levels):
            out.append((fam, lv, fi, li))
    return out


def load_model(path):
    """載入 nested_cv.py 存的模型與標準化參數。"""
    import torch
    from model import MultiSegmentBP
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                                  # 舊版 PyTorch 沒有 weights_only 參數
        ck = torch.load(path, map_location="cpu")
    model = MultiSegmentBP(fusion=ck["fusion"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, (np.asarray(ck["mu"]), np.asarray(ck["sd"])), ck


def predict_batches(model, norm, X, mask):
    """分批預測，回傳 (預測 mmHg, 權重, 分數)。一律用 CPU。"""
    import torch
    import train
    from train import predict
    train.DEVICE = torch.device("cpu")
    ys, ws, ss = [], [], []
    for b in range(0, len(X), BATCH):
        y, w, s = predict(model, norm, X[b:b + BATCH], mask[b:b + BATCH])
        ys.append(y), ws.append(w), ss.append(s)
    return np.concatenate(ys), np.concatenate(ws), np.concatenate(ss)


def make_corrupted(X, mask, sid, conds):
    """
    測試集的所有單段干擾版本。每個 (受試者 i, 段 j, 條件 c, 種子 k) 產生一組只有第 j 段被弄壞的集合。
    回傳：X_c (N, S, L)、mask_c (N, S)、索引表 meta (N, 6) = (i, j, 條件編號, k, 家族編號, 強度編號)
    """
    from degrade import degrade
    rows, sets = [], []
    for i in range(len(X)):
        for j in np.flatnonzero(mask[i]):
            for c, (fam, lv, fi, li) in enumerate(conds):
                for k in range(N_SEEDS):
                    rng = np.random.default_rng([k, int(sid[i]), int(j), fi, li])
                    Xi = X[i].copy()
                    Xi[j] = degrade(X[i, j], fam, lv, rng)
                    sets.append(Xi)
                    rows.append((i, j, c, k))
    meta = np.array(rows, dtype=int)
    return np.stack(sets).astype(np.float32), mask[meta[:, 0]], meta


def extra_tests(model, norm, X, mask):
    """刪除檢驗與重複片段測試（只用乾淨資料；只對至少 2 段的人）。"""
    y_full, w, _ = predict_batches(model, norm, X, mask)
    w = np.where(mask, w, np.nan)
    multi = mask.sum(axis=1) >= 2
    m_max, m_min = mask.copy(), mask.copy()
    rows = np.arange(len(X))
    j_max, j_min = np.nanargmax(w, axis=1), np.nanargmin(w, axis=1)
    m_max[rows, j_max] = False
    m_min[rows, j_min] = False
    m_max[~multi], m_min[~multi] = mask[~multi], mask[~multi]           # 只有 1 段的人不刪
    y_dmax, _, _ = predict_batches(model, norm, X, m_max)
    y_dmin, _, _ = predict_batches(model, norm, X, m_min)

    # 重複片段：{x1, x2} 對 {x1, x1, x2}（x1、x2 為前兩段有效片段）
    S = X.shape[1]
    X_pair, X_dup = np.zeros_like(X), np.zeros_like(X)
    m_pair, m_dup = np.zeros_like(mask), np.zeros_like(mask)
    for i in np.flatnonzero(multi):
        a, b = np.flatnonzero(mask[i])[:2]
        X_pair[i, 0], X_pair[i, 1] = X[i, a], X[i, b]
        m_pair[i, :2] = True
        X_dup[i, 0], X_dup[i, 1], X_dup[i, 2] = X[i, a], X[i, a], X[i, b]
        m_dup[i, :3] = True
    idx = np.flatnonzero(multi)
    y_pair = np.full_like(y_full, np.nan)
    y_dup = np.full_like(y_full, np.nan)
    if len(idx) and S >= 3:
        y_pair[idx], _, _ = predict_batches(model, norm, X_pair[idx], m_pair[idx])
        y_dup[idx], _, _ = predict_batches(model, norm, X_dup[idx], m_dup[idx])
    return {"n_seg": mask.sum(axis=1), "sbp_full": y_full[:, 0], "dbp_full": y_full[:, 1],
            "sbp_dropmax": y_dmax[:, 0], "dbp_dropmax": y_dmax[:, 1],
            "sbp_dropmin": y_dmin[:, 0], "dbp_dropmin": y_dmin[:, 1],
            "sbp_pair": y_pair[:, 0], "dbp_pair": y_pair[:, 1],
            "sbp_dup": y_dup[:, 0], "dbp_dup": y_dup[:, 1]}


def run_version(version, models):
    from folds import load_folds, get_split, index_of, N_REPEATS, N_OUTER
    out_dir = os.path.join(CV_ROOT, version)
    part_dir = os.path.join(out_dir, "interference_parts")
    os.makedirs(part_dir, exist_ok=True)
    repeats = [0] if QUICK_TEST else range(N_REPEATS)
    outers = [0] if QUICK_TEST else range(N_OUTER)
    conds = conditions()

    d = np.load(os.path.join(WORK_DIR, f"prep_{version}.npz"))
    X, mask, y, sid = d["X"], d["mask"], d["y"].astype(float), d["subject_id"]
    q = d["ssqi_author"] if "ssqi_author" in d.files else np.full(mask.shape, np.nan)
    if any(m in SSQI_SELECT for m in models) and "ssqi_author" not in d.files:
        raise RuntimeError("prep_*.npz 沒有原作者 SSQI：請先執行第 3 版 preprocess.py，再跑 B6")
    folds = load_folds(os.path.join(WORK_DIR, "folds.csv"))
    print("=" * 60)
    print(f"版本 {version}；比較組 {'、'.join(models)}；{len(conds)} 種干擾 × {N_SEEDS} 個種子")

    t0 = time.time()
    for r in repeats:
        for o in outers:
            todo = [m for m in models if not os.path.exists(os.path.join(part_dir, f"{m}_r{r}_o{o}.csv"))]
            if not todo:
                continue
            te_ids, _, _ = get_split(folds, r, o)
            i_te = index_of(sid, te_ids)
            Xt, Mt, yt, st, qt = X[i_te], mask[i_te], y[i_te], sid[i_te], q[i_te]
            Xc, Mc, meta = make_corrupted(Xt, Mt, st, conds)        # 同一折的所有模型共用
            for m in todo:
                path = os.path.join(out_dir, "models", f"{m}_r{r}_o{o}.pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"找不到模型 {path}：請先完成 nested_cv.py")
                model, norm, ck = load_model(path)
                assert np.array_equal(np.sort(ck["test_ids"]), np.sort(te_ids)), "模型的測試集與 folds.csv 不一致"
                i, j, c, k = meta.T
                if m in SSQI_SELECT:
                    # B6：乾淨資料依原作者 SSQI 選段；受干擾後以重算的 SSQI 重新選段
                    from train import select_best
                    _, _, j_clean = select_best(Xt, Mt, qt)
                    Xs, Ms, j_sel = select_best(Xc, Mc)
                    y_c, _, _ = predict_batches(model, norm, Xs, Ms)
                    w_clean = (np.arange(Mt.shape[1])[None, :] == j_clean[:, None]).astype(float)
                    w_c = (np.arange(Mc.shape[1])[None, :] == j_sel[:, None]).astype(float)
                    s_c = skew(Xc, axis=2, bias=True)          # B6 的「分數」= 重算的 SSQI
                else:
                    _, w_clean, _ = predict_batches(model, norm, Xt, Mt)
                    y_c, w_c, s_c = predict_batches(model, norm, Xc, Mc)
                part = pd.DataFrame({
                    "repeat": r, "outer": o, "subject_id": st[i],
                    "family": [conds[x][0] for x in c], "level": [conds[x][1] for x in c],
                    "seg": j, "seed": k, "sbp_true": yt[i, 0], "dbp_true": yt[i, 1],
                    "sbp_pred": y_c[:, 0], "dbp_pred": y_c[:, 1],
                    "s_seg": s_c[np.arange(len(j)), j], "w_seg": w_c[np.arange(len(j)), j],
                    "w_seg_clean": w_clean[i, j]})
                if m not in SSQI_SELECT:                       # 刪除與重複片段測試不適用於只用一段的 B6
                    ex = pd.DataFrame({"repeat": r, "outer": o, "subject_id": st,
                                       "sbp_true": yt[:, 0], "dbp_true": yt[:, 1], **extra_tests(model, norm, Xt, Mt)})
                    ex.to_csv(os.path.join(part_dir, f"extra_{m}_r{r}_o{o}.csv"), index=False)
                part.to_csv(os.path.join(part_dir, f"{m}_r{r}_o{o}.csv"), index=False, float_format="%.4f")
            print(f"  第 {r + 1} 次重複 第 {o + 1} 折完成（{len(meta)} 組干擾 × {len(todo)} 個模型）；已用 {hms(time.time() - t0)}")

    # 合併
    for m in models:
        names = [(r, o) for r in repeats for o in outers]
        if not all(os.path.exists(os.path.join(part_dir, f"{m}_r{r}_o{o}.csv")) for r, o in names):
            print(f"  {m}：尚有未完成的折")
            continue
        pd.concat([pd.read_csv(os.path.join(part_dir, f"{m}_r{r}_o{o}.csv")) for r, o in names]).to_csv(
            os.path.join(out_dir, f"interference_{m}.csv"), index=False, float_format="%.4f")
        if m not in SSQI_SELECT:
            pd.concat([pd.read_csv(os.path.join(part_dir, f"extra_{m}_r{r}_o{o}.csv")) for r, o in names]).to_csv(
                os.path.join(out_dir, f"extra_{m}.csv"), index=False)
        print(f"  {m}：已輸出 interference_{m}.csv" + ("" if m in SSQI_SELECT else f"、extra_{m}.csv"))
    print(f"{version} 耗時 {hms(time.time() - t0)}")


if __name__ == "__main__":
    t_all = time.time()
    for version, models in PLAN:
        run_version(version, models)
    print(f"\n全部完成，耗時 {hms(time.time() - t_all)}；接著請執行 stats.py")
