# -*- coding: utf-8 -*-
r"""
分折程式（論文 5.1 節）

外層：3 次重複 × 5 折，以受試者為單位切分（同一人不會同時在訓練與測試），依 SBP 分層。
內層：每個外層訓練集再切 3 折，用來選 λ_r、λ_c 與 epoch 數。
受試者 23 與 24 有一段完全相同（23_3 = 24_1），一律分在同一折。

所有資料版本（M1、M2a、M2b、M3）共用同一份分折，版本之間的比較才公平。
直接按 F5 執行：產生 folds.csv 並檢查。其他程式使用方式：
    from folds import load_folds, get_split
    folds = load_folds()
    test_ids, train_ids, inner = get_split(folds, repeat=0, outer=0)

輸出（WORK_DIR）：folds.csv、folds_report.txt；圖：Python\plot\分折SBP分布.png
"""
import os

import numpy as np
import pandas as pd

# ============ 設定 ============
WORK_DIR = r"D:\PPG-BP Database\解壓資料\ppgbp_work"
FIG_DIR = r"D:\PPG-BP Database\Python\plot"
GROUP_SOURCE = "prep_M1_all.npz"   # 用 M1 的 fold_group：其中 23 與 24 同組（最嚴格的限制）
N_REPEATS = 3
N_OUTER = 5
N_INNER = 3
SEED = 2026
# ==============================

FOLDS_CSV = os.path.join(WORK_DIR, "folds.csv")


def stratified_group_folds(groups, sbp, k, rng):
    """
    依 SBP 分層、以 group 為單位分成 k 折。
    做法：每組取平均 SBP 排序，每 k 組一區塊，區塊內隨機分到 k 個折。
    → 每折都含有低、中、高血壓的人，血壓分布相近。回傳每位受試者的折號。
    """
    df = pd.DataFrame({"g": groups, "sbp": sbp})
    g_sbp = df.groupby("g")["sbp"].mean()
    g_sbp = g_sbp.iloc[np.argsort(g_sbp.to_numpy() + rng.uniform(0, 1e-6, len(g_sbp)))]   # 同分時隨機
    names = g_sbp.index.to_numpy()
    g_fold = {}
    for start in range(0, len(names), k):
        block = names[start:start + k]
        labels = rng.permutation(k)[:len(block)]
        g_fold.update(zip(block, labels))
    return np.array([g_fold[g] for g in groups])


def make_folds(subject_id, groups, sbp, seed=SEED):
    """回傳長表：repeat、outer、subject_id、role（"test" 或內層折號 "0"、"1"、"2"）。"""
    rows = []
    for r in range(N_REPEATS):
        rng = np.random.default_rng([seed, r])
        outer = stratified_group_folds(groups, sbp, N_OUTER, rng)
        for o in range(N_OUTER):
            te = outer == o
            tr = ~te
            inner = stratified_group_folds(groups[tr], sbp[tr], N_INNER, np.random.default_rng([seed, r, o]))
            rows += [(r, o, s, "test") for s in subject_id[te]]
            rows += [(r, o, s, str(f)) for s, f in zip(subject_id[tr], inner)]
    return pd.DataFrame(rows, columns=["repeat", "outer", "subject_id", "role"])


def load_folds(path=FOLDS_CSV):
    return pd.read_csv(path, dtype={"role": str})


def get_split(folds, repeat, outer):
    """
    回傳 (測試集受試者編號, 外層訓練集受試者編號, 內層清單)。
    內層清單有 3 項，每項為 (內層訓練編號, 內層驗證編號)。
    """
    f = folds[(folds.repeat == repeat) & (folds.outer == outer)]
    test_ids = f.loc[f.role == "test", "subject_id"].to_numpy()
    tr = f[f.role != "test"]
    inner = [(tr.loc[tr.role != str(k), "subject_id"].to_numpy(), tr.loc[tr.role == str(k), "subject_id"].to_numpy())
             for k in range(N_INNER)]
    return test_ids, tr["subject_id"].to_numpy(), inner


def index_of(subject_id_all, ids):
    """把受試者編號轉成陣列索引（npz 中的列位置）。"""
    pos = {s: i for i, s in enumerate(subject_id_all)}
    return np.array([pos[s] for s in ids])


# ---------- 產生與檢查 ----------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    report = []

    def log(m=""):
        print(m)
        report.append(m)

    d = np.load(os.path.join(WORK_DIR, GROUP_SOURCE))
    sid, groups, sbp = d["subject_id"], d["fold_group"], d["y"][:, 0].astype(float)
    folds = make_folds(sid, groups, sbp)
    folds.to_csv(FOLDS_CSV, index=False)

    log("=" * 60)
    log("分折報告")
    log("=" * 60)
    log(f"{len(sid)} 人、{len(np.unique(groups))} 組；外層 {N_REPEATS} 次 × {N_OUTER} 折，內層 {N_INNER} 折，種子 {SEED}")
    ok_all = True

    # 1. 每次重複中，每人剛好當一次測試
    for r in range(N_REPEATS):
        cnt = folds[(folds.repeat == r) & (folds.role == "test")].subject_id.value_counts()
        ok = len(cnt) == len(sid) and (cnt == 1).all()
        ok_all &= ok
        log(f"1. 第 {r + 1} 次重複：每人剛好當一次測試 {'✓' if ok else '✗'}")

    # 2. 訓練與測試不重疊；內層驗證只來自外層訓練集
    ok = True
    for (r, o), f in folds.groupby(["repeat", "outer"]):
        te, tr, inner = get_split(folds, r, o)
        ok &= len(set(te) & set(tr)) == 0
        for itr, iva in inner:
            ok &= len(set(iva) & set(te)) == 0 and len(set(itr) & set(iva)) == 0
    ok_all &= ok
    log(f"2. 訓練、測試、內層驗證互不重疊 {'✓' if ok else '✗'}")

    # 3. 受試者 23 與 24 永遠同一折
    ok = all(f.set_index("subject_id").loc[[23, 24], "role"].nunique() == 1
             for _, f in folds.groupby(["repeat", "outer"]))
    ok_all &= ok
    log(f"3. 受試者 23 與 24 在每次切分中都同一折（含內層）{'✓' if ok else '✗'}")

    # 4. 各折人數與 SBP 分布相近
    log("\n4. 外層各折人數與 SBP（平均 ± 標準差，mmHg）")
    y_map = dict(zip(sid, sbp))
    for r in range(N_REPEATS):
        parts = []
        for o in range(N_OUTER):
            ids = folds[(folds.repeat == r) & (folds.outer == o) & (folds.role == "test")].subject_id
            v = np.array([y_map[s] for s in ids])
            parts.append(f"{len(v)} 人 {v.mean():.1f}±{v.std():.1f}")
        log(f"   第 {r + 1} 次：" + "；".join(parts))
    log(f"   全體：{sbp.mean():.1f}±{sbp.std():.1f}")

    # 5. 三次重複的切分不同
    a = folds[(folds.repeat == 0) & (folds.role == "test")].set_index("subject_id").outer
    b = folds[(folds.repeat == 1) & (folds.role == "test")].set_index("subject_id").outer
    same = (pd.crosstab(a, b.loc[a.index]).max(axis=1) / pd.Series(a).value_counts().sort_index()).mean()
    ok = same < 0.6
    ok_all &= ok
    log(f"\n5. 第 1、2 次重複的切分不同（同折的人平均只有 {same:.0%} 仍同折）{'✓' if ok else '✗'}")

    log("\n" + ("全部檢查通過" if ok_all else "有檢查未通過，請回報"))
    log(f"分折已存：{FOLDS_CSV}")
    with open(os.path.join(WORK_DIR, "folds_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))

    # 圖：第 1 次重複的各折 SBP 分布
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [[y_map[s] for s in folds[(folds.repeat == 0) & (folds.outer == o) & (folds.role == "test")].subject_id]
            for o in range(N_OUTER)]
    ax.boxplot(data)
    ax.set_xticks(range(1, N_OUTER + 1), [f"第 {o + 1} 折" for o in range(N_OUTER)])
    ax.axhline(sbp.mean(), color="gray", ls="--", lw=1, label=f"全體平均 {sbp.mean():.1f}")
    ax.set_ylabel("SBP (mmHg)")
    ax.set_title("外層分折（第 1 次重複）：各折的 SBP 分布")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "分折SBP分布.png")
    fig.savefig(out, dpi=150)
    print(f"圖已存：{out}")
    plt.show()
