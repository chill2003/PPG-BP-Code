# -*- coding: utf-8 -*-
r"""
λ_s 敏感度分析（論文 4.4 節；附錄 C）

問題：分數正則的比重 λ_s 固定為 0.01，換成 0.001 或 0.1，結果會不會明顯改變？
做法：主分析 M3、第 1 次重複的 5 個外層折，B5 沿用巢狀交叉驗證選出的 λ_r、λ_c 與 epoch 數，
      只改 λ_s，在外層訓練集重新訓練，於外層測試集評估：
        乾淨 SBP、DBP MAE；僅測試干擾下的 ΔSBP MAE；排序正確率（見過、未見家族）
      λ_s = 0.01 與主分析設定完全相同，可用來核對（結果應與 nested_cv 的 B5 一致）

需先執行：nested_cv.py（selection_B5.csv）。直接按 F5 執行，約 20–40 分鐘。
輸出：ppgbp_work\stats\lambda_s_sensitivity.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ============ 設定 ============
WORK_DIR = r"D:\PPG-BP Database\解壓資料\ppgbp_work"
VERSION = "M3_dedup"
LAMBDA_S_LIST = [0.001, 0.01, 0.1]
REPEAT = 0                         # 用第 1 次重複的 5 折
MAX_EPOCHS = 200                   # 與 nested_cv.py 相同（決定學習率軌跡）
SEED = 0                           # 與 nested_cv.py 相同
N_SEEDS = 3                        # 干擾測試的種子數（比主分析少，加快速度）
N_JOBS = 6                         # 同時訓練幾個；出錯時改成 1
# ==============================


def run_one(lam_s, o, sel):
    import sys as _sys
    if HERE not in _sys.path:
        _sys.path.insert(0, HERE)
    import torch
    import train
    import interference as itf
    from folds import load_folds, get_split, index_of
    train.DEVICE = torch.device("cpu")
    torch.set_num_threads(1)
    itf.N_SEEDS = N_SEEDS

    d = np.load(os.path.join(WORK_DIR, f"prep_{VERSION}.npz"))
    X, mask, y, sid = d["X"], d["mask"], d["y"].astype(float), d["subject_id"]
    te_ids, tr_ids, _ = get_split(load_folds(os.path.join(WORK_DIR, "folds.csv")), REPEAT, o)
    i_te, i_tr = index_of(sid, te_ids), index_of(sid, tr_ids)
    seed = SEED + 1000 * REPEAT + 10 * o

    model, norm, _ = train.train_model("B5", X[i_tr], mask[i_tr], y[i_tr], int(sel["epochs"]), seed=seed,
                                       lambda_r=float(sel["lambda_r"]), lambda_c=float(sel["lambda_c"]),
                                       lambda_s=lam_s, verbose=False, t_max=MAX_EPOCHS)
    Xt, Mt, yt, st = X[i_te], mask[i_te], y[i_te], sid[i_te]
    y_clean, _, _ = itf.predict_batches(model, norm, Xt, Mt)
    conds = itf.conditions()
    Xc, Mc, meta = itf.make_corrupted(Xt, Mt, st, conds)
    y_c, _, s_c = itf.predict_batches(model, norm, Xc, Mc)
    i, j, c, k = meta.T
    clean = pd.DataFrame({"repeat": REPEAT, "subject_id": st, "sbp_true": yt[:, 0], "dbp_true": yt[:, 1],
                          "sbp_pred": y_clean[:, 0], "dbp_pred": y_clean[:, 1]})
    inter = pd.DataFrame({"repeat": REPEAT, "subject_id": st[i], "family": [conds[x][0] for x in c],
                          "level": [conds[x][1] for x in c], "seg": j, "seed": k,
                          "sbp_true": yt[i, 0], "dbp_true": yt[i, 1], "sbp_pred": y_c[:, 0], "dbp_pred": y_c[:, 1],
                          "s_seg": s_c[np.arange(len(j)), j]})
    return lam_s, o, clean, inter


if __name__ == "__main__":
    from stats import subject_error, subject_delta, ranking, TEST_ONLY

    sel = pd.read_csv(os.path.join(WORK_DIR, "cv", VERSION, "selection_B5.csv"))
    sel = sel[sel.repeat == REPEAT].set_index("outer")
    tasks = [(lam, o, sel.loc[o].to_dict()) for lam in LAMBDA_S_LIST for o in sel.index]
    print(f"λ_s 敏感度：{LAMBDA_S_LIST} × {len(sel)} 折 = {len(tasks)} 次訓練（B5，{VERSION}，第 {REPEAT + 1} 次重複）")
    t0 = time.time()
    if N_JOBS > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(run_one)(*t) for t in tasks)
    else:
        results = [run_one(*t) for t in tasks]

    rows = []
    for lam in LAMBDA_S_LIST:
        clean = pd.concat([r[2] for r in results if r[0] == lam])
        inter = pd.concat([r[3] for r in results if r[0] == lam])
        e = subject_error(clean)
        dlt = subject_delta(inter, clean, sorted(TEST_ONLY))
        rk = ranking(inter)
        rows.append({"lambda_s": lam, "n_subjects": len(e), "clean_sbp_mae": e.sbp.mean(), "clean_dbp_mae": e.dbp.mean(),
                     "test_only_dsbp_mae": dlt.sbp.mean(),
                     "rank_acc_seen": rk["seen"].mean(), "rank_acc_test_only": rk["test_only"].mean()})
    table = pd.DataFrame(rows)
    out_dir = os.path.join(WORK_DIR, "stats")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "lambda_s_sensitivity.csv"), index=False, encoding="utf-8-sig")
    print("\nλ_s 敏感度結果（B5，M3 第 1 次重複 5 折，219 人）")
    print(table.round(3).to_string(index=False))
    print(f"\n耗時 {(time.time() - t0) / 60:.1f} 分鐘；已存：{os.path.join(out_dir, 'lambda_s_sensitivity.csv')}")
