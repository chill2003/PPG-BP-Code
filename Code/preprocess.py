# -*- coding: utf-8 -*-
"""
PPG-BP 前處理（第 3 版，2026-11-16）

需先執行 audit_data.py（第 2 版）。讀取 ppgbp_work 中的 ppgbp_raw.pkl 與 subjects_<版本>.csv，
對每一段 2100 點的片段（含 4200 點拆開的前後半段）：
  1. 在原始 1 kHz 訊號上計算偏態 SSQI（原論文第 6 頁）
  2. 抗混疊濾波後降取樣至 125 Hz，取 262 點
  3. 每段各自 z-score
  4. 另外讀入原作者提供的 SSQI（Table 1.xlsx），供基準 B6 在乾淨資料上選段（論文 3.3 節第 4 點）

用法：在 Spyder 開啟本檔，確認 WORK_DIR，按 F5 執行。

輸出（WORK_DIR）
  prep_M1_all.npz、prep_M2a_drop4200.npz、prep_M2b_first2100.npz、prep_M3_dedup.npz
  prep_report.txt、prep_check.png
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from scipy.stats import skew
import matplotlib.pyplot as plt

# ============ 設定 ============
WORK_DIR = r"D:\PPG-BP Database\解壓資料\ppgbp_work"   # audit_data.py 的輸出資料夾
AUTHOR_XLSX = r"D:\PPG-BP Database\Table 1.xlsx"         # 原作者提供的 SSQI

FS_RAW = 1000          # 原始取樣率 (Hz)
DOWN = 8               # 1000 / 8 = 125 Hz
FS_NEW = FS_RAW // DOWN
RAW_LEN = 2100
NEW_LEN = 262          # 2.1 秒 × 125 Hz 取整
# ==============================

report = []


def log(msg=""):
    print(msg)
    report.append(msg)


def ssqi(x):
    return float(skew(x, bias=True))


def downsample(x):
    """抗混疊 FIR 濾波 + 降取樣；padtype='line' 避免兩端跳動。"""
    return resample_poly(np.asarray(x, dtype=float), up=1, down=DOWN, padtype="line")[:NEW_LEN]


def zscore(x):
    s = x.std()
    if s == 0:
        raise ValueError("訊號為常數，無法 z-score")
    return (x - x.mean()) / s


def preprocess_segment(x_raw):
    """2100 點原始片段 → (125 Hz z-score 訊號, SSQI)。"""
    x = np.asarray(x_raw, dtype=float)
    if len(x) != RAW_LEN:
        raise ValueError(f"片段長度 {len(x)} 不是 {RAW_LEN}")
    return zscore(downsample(x)).astype(np.float32), ssqi(x)


def encode_sex(v):
    s = str(v).strip().lower()
    return 1.0 if s.startswith("m") else 0.0 if s.startswith("f") else np.nan


def load_author_ssqi(path):
    """
    讀 Table 1.xlsx：表格分成左右 4 欄區塊，每區塊為 Num.、subject ID、segment 1、2、3。
    回傳 {(受試者, 片段編號): 作者 SSQI}。
    """
    x = pd.read_excel(path, header=None)
    out = {}
    for b in range(x.shape[1] // 5):
        blk = x.iloc[1:, b * 5:b * 5 + 5].dropna(how="all")
        for _, r in blk.iterrows():
            for k in range(3):
                if pd.notna(r.iloc[2 + k]):
                    out[(int(r.iloc[1]), k + 1)] = float(r.iloc[2 + k])
    return out


def author_value(author, sid, label):
    """
    片段標籤 "1"、"2"、"3" → 作者值。
    4200 點拆開的半段（"1a"、"1b"）沒有對應的作者值 → NaN（B6 改用本研究重算的 SSQI）。
    """
    return author.get((int(sid), int(label)), np.nan) if label.isdigit() else np.nan


def self_test():
    f = lambda t: 2000 + 300 * np.sin(2 * np.pi * 1.2 * t) + 60 * np.sin(2 * np.pi * 8 * t)
    y = downsample(f(np.arange(RAW_LEN) / FS_RAW))
    truth = f(np.arange(NEW_LEN) / FS_NEW)
    err_mid = np.abs(y - truth)[10:-10].max() / 300
    err_edge = np.abs(y - truth).max() / 300
    assert len(y) == NEW_LEN and err_mid < 0.01 and err_edge < 0.1, "降取樣自我檢查失敗"
    return err_mid, err_edge


err_mid, err_edge = self_test()
log("=" * 64)
log("PPG-BP 前處理報告（第 2 版）")
log("=" * 64)
log(f"自我檢查通過：降取樣誤差（相對脈搏振幅）中段 {err_mid:.2%}、含兩端 {err_edge:.2%}")

with open(os.path.join(WORK_DIR, "ppgbp_raw.pkl"), "rb") as f:
    raw = pickle.load(f)
if "pieces" not in raw:
    raise RuntimeError("ppgbp_raw.pkl 是舊版，請先執行第 2 版 audit_data.py")
pieces = raw["pieces"]
summary = pd.read_csv(os.path.join(WORK_DIR, "versions_summary.csv"))
author = load_author_ssqi(AUTHOR_XLSX)
log(f"原作者 SSQI：讀入 {len(author)} 段（Table 1.xlsx）")

cache = {key: preprocess_segment(x) for key, x in pieces.items()}
log(f"\n處理片段：{len(cache)} 段（含長片段的前 2100 點與拆開的半段），"
    f"每段 {RAW_LEN} 點 → {NEW_LEN} 點（{FS_NEW} Hz）")

log("\n--- 各版本輸出 ---")
all_ok = True
for rec in summary.itertuples():
    ver = rec.version
    subj = pd.read_csv(os.path.join(WORK_DIR, f"subjects_{ver}.csv"), dtype={"segments": str})
    labels = [s.split(";") for s in subj["segments"]]
    n, S = len(subj), max(len(l) for l in labels)
    X = np.zeros((n, S, NEW_LEN), dtype=np.float32)
    mask = np.zeros((n, S), dtype=bool)               # True = 此位置有片段
    Q = np.full((n, S), np.nan, dtype=np.float32)     # SSQI
    QA = np.full((n, S), np.nan, dtype=np.float32)    # 原作者 SSQI（B6 用）
    seg_labels = np.full((n, S), "", dtype="U4")      # 片段標籤，例如 "1"、"1a"
    for i, (sid, labs) in enumerate(zip(subj["subject_id"], labels)):
        for j, lab in enumerate(labs):
            X[i, j], Q[i, j] = cache[(int(sid), lab)]
            QA[i, j] = author_value(author, sid, lab)
            mask[i, j] = True
            seg_labels[i, j] = lab
    demo = np.column_stack([subj["age"].to_numpy(float), subj["sex"].map(encode_sex).to_numpy(float),
                            subj["bmi"].to_numpy(float)]).astype(np.float32)
    np.savez_compressed(
        os.path.join(WORK_DIR, f"prep_{ver}.npz"),
        X=X, mask=mask, ssqi=Q, ssqi_author=QA, seg_labels=seg_labels,
        y=subj[["sbp", "dbp"]].to_numpy(np.float32), demo=demo,
        subject_id=subj["subject_id"].to_numpy(), fold_group=subj["fold_group"].to_numpy(),
        hypertension=np.array(subj["hypertension"].fillna("").astype(str).tolist(), dtype="U"),
        fs=FS_NEW, version=ver)
    ok = (n, int(mask.sum())) == (rec.subjects, rec.segments)
    all_ok &= ok
    q = Q[mask]
    log(f"{rec.description}")
    log(f"  {n} 人、{int(mask.sum())} 段，陣列 X {X.shape} {'✓' if ok else '✗ 與稽核結果不符'}；"
        f"SSQI 平均 {q.mean():.3f}、≤ 0 的片段 {int((q <= 0).sum())} 段")
    miss = [f"{s}_{l}" for s, row_l, row_q in zip(subj["subject_id"], seg_labels, QA)
            for l, v in zip(row_l, row_q) if l and np.isnan(v)]
    log(f"  原作者 SSQI：{int(mask.sum()) - len(miss)} 段有值" + (f"；無對應值 {len(miss)} 段（{'、'.join(miss)}）" if miss else ""))

log("\n輸出陣列：X (人數, 最多片段數, 262)、mask、ssqi（本研究重算）、ssqi_author（原作者）、seg_labels、y (SBP, DBP mmHg)、"
    "demo (年齡, 性別 男=1, BMI)、subject_id、fold_group、hypertension")
log("注意：M1 中受試者 231 有 5 段，其他版本最多 3 段；模型以 mask 處理不同片段數")
log("注意：231_1、231_2 的原作者 SSQI 是對完整 4200 點計算的；M2b、M3 只取前 2100 點，仍沿用作者值")
log("完成標準：" + ("全部符合" if all_ok else "有版本人數或片段數不符，請檢查"))

# ---------- 波形對照圖：一般片段 + 4200 點拆開的前後半段 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
show = [sorted(k for k in pieces if k[1].isdigit())[0]]
show += sorted(k for k in pieces if not k[1].isdigit())[:2]          # 第一個長片段的前後半
fig, axes = plt.subplots(len(show), 1, figsize=(9, 2.6 * len(show)), sharex=True)
for ax, key in zip(np.atleast_1d(axes), show):
    x = pieces[key]
    ax.plot(np.arange(RAW_LEN) / FS_RAW, x, lw=0.8, label="1 kHz 原始")
    ax.plot(np.arange(NEW_LEN) / FS_NEW, downsample(x), ".", ms=3, label="125 Hz 降取樣")
    ax.set_title(f"受試者 {key[0]} 片段 {key[1]}（SSQI {cache[key][1]:.2f}）", fontsize=10)
    ax.set_ylabel("ADC 值")
np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
np.atleast_1d(axes)[-1].set_xlabel("時間 (秒)")
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "prep_check.png"), dpi=150)
plt.show()

with open(os.path.join(WORK_DIR, "prep_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"\n完成，輸出在：{WORK_DIR}")
