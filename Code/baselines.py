# -*- coding: utf-8 -*-
"""
基準 B−1 與 B-demo（論文 5.2 節）

  B−1   ：每個人都猜「訓練集的平均 SBP、DBP」→ 模型到底比「不學習」好多少
  B-demo：只用年齡、性別、BMI 的嶺迴歸 → PPG 模型是否只是間接估計年齡

兩者都不用 PPG，因此各資料版本結果相同，以主分析 M3 計算。
使用 folds.py 產生的同一份分折（3 次重複 × 5 折）；B-demo 的正則強度 α 由內層 3 折選出。

需先執行 folds.py。直接按 F5 執行。
輸出（WORK_DIR）：
  preds_B-1.csv、preds_B-demo.csv   每位受試者在每次重複的預測（之後做 Wilcoxon 與 bootstrap 用）
  baselines_summary.csv             MAE、RMSE、平均偏差（3 次重複的平均 ± 標準差）
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from folds import load_folds, get_split, index_of, N_REPEATS, N_OUTER

# ============ 設定 ============
WORK_DIR = r"D:\PPG-BP Database\解壓資料\ppgbp_work"
NPZ = os.path.join(WORK_DIR, "prep_M3_dedup.npz")
ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]     # B-demo 的候選 α
# ==============================


def mean_predictor(y_tr, n_te):
    """B−1：訓練集平均。"""
    return np.tile(y_tr.mean(axis=0), (n_te, 1))


def ridge_fit_predict(Z_tr, y_tr, Z_te, alpha):
    """B-demo：特徵標準化（只用訓練集估計）後做嶺迴歸，同時預測 SBP、DBP。"""
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(Z_tr, y_tr)
    return model.predict(Z_te)


def choose_alpha(Z, y, sid, inner):
    """內層 3 折：選 SBP 與 DBP 平均 MAE 最小的 α。"""
    scores = []
    for a in ALPHAS:
        maes = []
        for itr, iva in inner:
            i_tr, i_va = index_of(sid, itr), index_of(sid, iva)
            p = ridge_fit_predict(Z[i_tr], y[i_tr], Z[i_va], a)
            maes.append(np.abs(p - y[i_va]).mean())
        scores.append(np.mean(maes))
    return ALPHAS[int(np.argmin(scores))]


def summarize(pred_df, name):
    """每次重複各算一次 MAE、RMSE、偏差（全部 219 人），再取 3 次的平均 ± 標準差。"""
    rows = []
    for r, g in pred_df.groupby("repeat"):
        rec = {"repeat": r}
        for t in ("sbp", "dbp"):
            e = g[f"{t}_pred"] - g[f"{t}_true"]
            rec[f"{t}_mae"] = e.abs().mean()
            rec[f"{t}_rmse"] = np.sqrt((e ** 2).mean())
            rec[f"{t}_bias"] = e.mean()
        rows.append(rec)
    per = pd.DataFrame(rows)
    out = {"model": name}
    for c in per.columns[1:]:
        out[c] = per[c].mean()
        out[c + "_sd"] = per[c].std(ddof=1)
    return out


if __name__ == "__main__":
    d = np.load(NPZ)
    sid, y = d["subject_id"], d["y"].astype(float)
    Z = d["demo"].astype(float)                       # 年齡、性別（男 = 1）、BMI
    folds = load_folds()
    print(f"資料：{os.path.basename(NPZ)}，{len(sid)} 人；分折：{N_REPEATS} 次 × {N_OUTER} 折")
    assert not np.isnan(Z).any(), "年齡、性別或 BMI 有缺值"

    preds = {"B-1": [], "B-demo": []}
    alphas = []
    for r in range(N_REPEATS):
        for o in range(N_OUTER):
            te_ids, tr_ids, inner = get_split(folds, r, o)
            i_te, i_tr = index_of(sid, te_ids), index_of(sid, tr_ids)
            p1 = mean_predictor(y[i_tr], len(i_te))
            a = choose_alpha(Z, y, sid, inner)
            alphas.append(a)
            p2 = ridge_fit_predict(Z[i_tr], y[i_tr], Z[i_te], a)
            for name, p in (("B-1", p1), ("B-demo", p2)):
                preds[name].append(pd.DataFrame({
                    "repeat": r, "outer": o, "subject_id": sid[i_te],
                    "sbp_true": y[i_te, 0], "dbp_true": y[i_te, 1],
                    "sbp_pred": p[:, 0], "dbp_pred": p[:, 1]}))

    summary = []
    for name, parts in preds.items():
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(WORK_DIR, f"preds_{name}.csv"), index=False)
        summary.append(summarize(df, name))
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(WORK_DIR, "baselines_summary.csv"), index=False, encoding="utf-8-sig")

    print(f"\nB-demo 內層選出的 α：{pd.Series(alphas).value_counts().sort_index().to_dict()}（共 {len(alphas)} 折）")
    print("\n基準結果（3 次重複的平均 ± 標準差，單位 mmHg）")
    print(f"{'':8s}{'SBP MAE':>16s}{'SBP RMSE':>16s}{'SBP 偏差':>14s}{'DBP MAE':>16s}{'DBP RMSE':>16s}{'DBP 偏差':>14s}")
    for _, s in summary.iterrows():
        cells = [f"{s[c]:.2f} ± {s[c + '_sd']:.2f}" for c in
                 ("sbp_mae", "sbp_rmse", "sbp_bias", "dbp_mae", "dbp_rmse", "dbp_bias")]
        print(f"{s['model']:8s}" + "".join(f"{c:>16s}" for c in cells))

    # B-demo 的係數方向（全部資料擬合一次，只用來解讀，不用於評估）
    m = make_pipeline(StandardScaler(), Ridge(alpha=float(pd.Series(alphas).mode()[0]))).fit(Z, y)
    coef = m[-1].coef_
    print("\nB-demo 標準化係數（每增加 1 個標準差，血壓變化 mmHg；僅供解讀）")
    for t, row in zip(("SBP", "DBP"), coef):
        print(f"  {t}：年齡 {row[0]:+.2f}、性別(男) {row[1]:+.2f}、BMI {row[2]:+.2f}")
    print(f"\n輸出：preds_B-1.csv、preds_B-demo.csv、baselines_summary.csv（{WORK_DIR}）")
