# -*- coding: utf-8 -*-
"""
PPG-BP 資料讀取與稽核（第 2.1 版，2026-11-11：加入飽和片段規則；主分析為 M3）

用法：在 Spyder 開啟本檔，確認下方路徑，按 F5 執行。原始資料只讀取、不修改。

步驟
  1. 找出長度異常（4200 點）的片段，拆成前後各 2100 點，並以 SSQI、心率、
     接縫跳動判斷兩半是否都像真正的脈搏波
  2. 找出數值完全相同的片段，分成「同一人內重複」與「跨受試者重複」
  3. 對所有片段計算 SSQI，標出 SSQI ≤ 0（不像正常脈搏波）的片段
  4. 依三種方法建立資料版本（方法 2 再分 2a、2b）

輸出（OUT_DIR）
  audit_report.txt          稽核報告
  audit_issues.csv          每一筆異常
  segment_ssqi.csv          每段（含拆開的前後半段）的 SSQI
  long_segments.csv         長度異常片段的前後半比較
  duplicate_groups.csv      相同片段分組
  versions_summary.csv      各版本人數、片段數
  subjects_<版本>.csv        各版本的受試者與使用片段
  ppgbp_raw.pkl             原始訊號與切好的 2100 點片段（前處理直接讀這個檔）
"""
import os
import re
import glob
import pickle
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import skew

# ============ 設定 ============
DATA_DIR = r"D:\PPG-BP Database\解壓資料"                 # 原始資料（遞迴搜尋 .txt 與 .xlsx）
OUT_DIR = r"D:\PPG-BP Database\解壓資料\ppgbp_work"       # 輸出資料夾

EXPECTED_LEN = 2100       # 每段標稱長度（1 kHz × 2.1 秒，原論文第 3 頁）
FS = 1000                 # 標稱取樣率 (Hz)

# 方法 3（刪除相同片段）的規則
DUP_WITHIN = "keep_first"   # 同一人內重複："keep_first" 保留編號最小的一份；"drop_all" 全部刪除
DUP_CROSS = "drop_all"      # 跨受試者重複："drop_all" 全部刪除（無法判斷訊號屬於誰）
M3_LONG = "first"           # 方法 3 中 4200 點片段："first" 只取前 2100 點；"split" 拆兩段；"drop" 刪除

# 訊號飽和：數值觸及 12 位元 AD 的上限 4095 或下限 0，表示感測器量不到訊號（論文 3.2 節）
# 2026-11-11 決定：只刪明顯壞掉的片段；M1 為「完全不刪」的參考，不套用
ADC_MAX = 4095
ADC_MIN = 0
SAT_DROP_IN = {"M2a_drop4200", "M2b_first2100", "M3_dedup"}

# 血壓合理範圍（只標記、不自動排除）
SBP_RANGE = (70, 250)
DBP_RANGE = (40, 150)
# ==============================

VERSIONS = {
    "M1_all":        "方法 1：全部保留（4200 點拆成前後兩段，相同片段照用）",
    "M2a_drop4200":  "方法 2a：刪除 4200 點片段（231_1、231_2）＋刪除飽和片段",
    "M2b_first2100": "方法 2b：4200 點片段只取前 2100 點＋刪除飽和片段",
    "M3_dedup":      "方法 3（主分析）：刪除相同片段＋刪除飽和片段",
}

os.makedirs(OUT_DIR, exist_ok=True)
report, issues = [], []


def log(msg=""):
    print(msg)
    report.append(msg)


def add_issue(sid, seg, kind, detail):
    issues.append({"subject_id": sid, "segment": seg, "type": kind, "detail": detail})


def ssqi(x):
    """偏態 SSQI = 1/N Σ ((x − μ)/σ)^3（原論文第 6 頁）。> 0 表示像正常脈搏波。"""
    return float(skew(np.asarray(x, dtype=float), bias=True))


def heart_rate(x, fs=FS):
    """以自相關估計心率 (bpm)，搜尋 0.3–1.6 秒的週期。"""
    v = np.asarray(x, dtype=float) - np.mean(x)
    ac = np.correlate(v, v, "full")[len(v) - 1:]
    if ac[0] == 0:
        return np.nan
    lo, hi = int(0.3 * fs), min(int(1.6 * fs), len(v) - 1)
    k = lo + int(np.argmax(ac[lo:hi]))
    return 60.0 * fs / k


# ---------- 1. 受試者資訊（Excel）----------
def find_meta_file(data_dir):
    files = [f for f in glob.glob(os.path.join(data_dir, "**", "*.xls*"), recursive=True)
             if not os.path.basename(f).startswith("~$")]
    if not files:
        raise FileNotFoundError("找不到 Excel 檔，請確認 DATA_DIR")
    if len(files) > 1:
        log(f"[注意] 找到 {len(files)} 個 Excel 檔，使用第一個：{os.path.basename(files[0])}")
    return files[0]


def read_meta(path):
    raw = pd.read_excel(path, header=None)
    for i in range(min(10, len(raw))):
        if any("subject" in str(v).lower() for v in raw.iloc[i].tolist()):
            df = raw.iloc[i + 1:].copy()
            df.columns = [str(c).strip() for c in raw.iloc[i].tolist()]
            return df.dropna(how="all").reset_index(drop=True), i
    raise ValueError("Excel 中找不到含有 subject 的標題列，請把前幾列貼給 Claude")


def pick_column(df, *keys):
    for c in df.columns:
        if all(k in c.lower() for k in keys):
            return c
    return None


meta_path = find_meta_file(DATA_DIR)
meta_raw, header_row = read_meta(meta_path)
log("=" * 64)
log("PPG-BP 資料稽核報告（第 2 版）")
log("=" * 64)
log(f"受試者資訊檔：{os.path.basename(meta_path)}（標題在第 {header_row + 1} 列）")

colmap = {k: pick_column(meta_raw, *keys) for k, keys in {
    "subject_id": ("subject",), "sex": ("sex",), "age": ("age",), "height": ("height",),
    "weight": ("weight",), "sbp": ("systolic",), "dbp": ("diastolic",), "hr": ("heart",),
    "bmi": ("bmi",), "hypertension": ("hypertension",), "diabetes": ("diabetes",)}.items()}
for must in ("subject_id", "sbp", "dbp"):
    if colmap[must] is None:
        raise ValueError(f"必要欄位 {must} 找不到，請把 Excel 的標題列貼給 Claude")

meta = pd.DataFrame({k: meta_raw[v] for k, v in colmap.items() if v is not None})
meta["subject_id"] = pd.to_numeric(meta["subject_id"], errors="coerce")
meta = meta.dropna(subset=["subject_id"]).copy()
meta["subject_id"] = meta["subject_id"].astype(int)
for c in ("age", "height", "weight", "sbp", "dbp", "hr", "bmi"):
    if c in meta:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
for sid in meta["subject_id"][meta["subject_id"].duplicated()]:
    add_issue(sid, "", "受試者編號重複", "Excel 中出現多次")
meta = meta.drop_duplicates("subject_id").set_index("subject_id").sort_index()
log(f"Excel 中的受試者數：{len(meta)}")

# ---------- 2. PPG 訊號 ----------
pat = re.compile(r"^(\d+)_(\d+)\.txt$", re.IGNORECASE)
signals, skipped = {}, []
for f in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True)):
    m = pat.match(os.path.basename(f))
    if not m:
        skipped.append(os.path.basename(f))
        continue
    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
        tokens = [t for t in re.split(r"[\s,;]+", fh.read().strip()) if t]
    vals = []
    for t in tokens:
        try:
            vals.append(float(t))
        except ValueError:
            pass
    if len(vals) != len(tokens):
        add_issue(int(m.group(1)), int(m.group(2)), "非數值內容", f"{len(tokens) - len(vals)} 個")
    signals[(int(m.group(1)), int(m.group(2)))] = np.asarray(vals, dtype=float)
log(f"PPG 文字檔：{len(signals)} 個（另有 {len(skipped)} 個檔名不符「編號_片段.txt」，已略過）")

seg_by_subject = defaultdict(list)
for sid, seg in signals:
    seg_by_subject[sid].append(seg)

# ---------- 3. 長度異常片段：拆成前後各 2100 點 ----------
L = EXPECTED_LEN
pieces = {}          # (受試者, 標籤) → 2100 點；標籤 "1" 為原段（長段則為前 2100 點），"1a"、"1b" 為拆開的前後半
long_k = {}          # (受試者, 片段) → 拆成幾段
long_rows = []
log("\n--- 1. 長度異常片段 ---")
for (sid, seg), x in sorted(signals.items()):
    n = len(x)
    if n == L:
        pieces[(sid, str(seg))] = x
        continue
    if n < L:
        add_issue(sid, seg, "長度不足", f"{n} 點，無法使用")
        continue
    add_issue(sid, seg, "長度異常", f"{n} 點（應為 {L}）")
    pieces[(sid, str(seg))] = x[:L]
    if n % L:
        add_issue(sid, seg, "長度不是 2100 的倍數", f"{n} 點，只取前 {L} 點，不拆段")
        continue
    k = n // L
    long_k[(sid, seg)] = k
    d = np.abs(np.diff(x))
    for i in range(k):
        part = x[i * L:(i + 1) * L]
        pieces[(sid, f"{seg}{chr(97 + i)}")] = part
        seam = d[(i + 1) * L - 1] if i < k - 1 else np.nan
        others = np.delete(d, [(j + 1) * L - 1 for j in range(k - 1)])
        long_rows.append({
            "subject_id": sid, "segment": seg, "part": f"{seg}{chr(97 + i)}",
            "points": f"{i * L + 1}–{(i + 1) * L}",
            "ssqi": round(ssqi(part), 3), "heart_rate_bpm": round(heart_rate(part), 1),
            "seam_jump_after": seam, "max_other_jump": others.max(),
        })
    first, second = x[:L], x[L:2 * L]
    seam = d[L - 1]
    log(f"{sid}_{seg}：{n} 點 → 拆成 {k} 段")
    log(f"  前後半完全相同：{'是' if np.array_equal(first, second) else '否'}；"
        f"每點連續重複兩次：{'是' if np.array_equal(x[0::2], x[1::2]) else '否'}")
    log(f"  SSQI 前半 {ssqi(first):.3f}、後半 {ssqi(second):.3f}"
        f"（{'兩半都像正常脈搏波' if min(ssqi(first), ssqi(second)) > 0 else '有一半 SSQI ≤ 0'}）")
    log(f"  估計心率 前半 {heart_rate(first):.0f}、後半 {heart_rate(second):.0f} bpm")
    log(f"  接縫跳動 {seam:.0f}；其他相鄰點最大跳動 {np.delete(d, L - 1).max():.0f}"
        f"（{'接縫明顯，像兩段不同量測接在一起' if seam > np.delete(d, L - 1).max() else '接縫不明顯'}）")
if not long_k:
    log("沒有長度為 2100 倍數的長片段")

# ---------- 4. 相同片段 ----------
log("\n--- 2. 數值完全相同的片段 ---")
hash_map = defaultdict(list)
for key, x in signals.items():
    hash_map[hashlib.md5(np.round(x, 6).tobytes()).hexdigest()].append(key)
dup_groups = [sorted(k) for k in hash_map.values() if len(k) > 1]
dup_rows, drop_dup = [], set()
for gi, grp in enumerate(sorted(dup_groups), start=1):
    kind = "同一人內" if len({s for s, _ in grp}) == 1 else "跨受試者"
    names = "、".join(f"{s}_{g}" for s, g in grp)
    if kind == "同一人內":
        to_drop = grp[1:] if DUP_WITHIN == "keep_first" else grp
    else:
        to_drop = grp if DUP_CROSS == "drop_all" else grp[1:]
    drop_dup.update(to_drop)
    for s, g in grp:
        add_issue(s, g, f"數值完全相同（{kind}）", names)
        dup_rows.append({"group": gi, "kind": kind, "subject_id": s, "segment": g,
                         "ssqi": round(ssqi(signals[(s, g)]), 3),
                         "method3": "刪除" if (s, g) in drop_dup else "保留"})
    log(f"第 {gi} 組（{kind}）：{names} → 方法 3 刪除 "
        + "、".join(f"{s}_{g}" for s, g in to_drop))
if not dup_groups:
    log("沒有相同片段")
cross_pairs = [grp for grp in dup_groups if len({s for s, _ in grp}) > 1]

# ---------- 5. 所有片段的 SSQI ----------
log("\n--- 3. SSQI（是否像正常脈搏波）---")
ssqi_rows = []
for (sid, lab), x in sorted(pieces.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    m_ = re.match(r"(\d+)([a-z]?)$", lab)
    is_long_first = m_.group(2) == "" and (sid, int(m_.group(1))) in long_k
    ssqi_rows.append({"subject_id": sid, "label": lab,
                      "kind": "長片段的前 2100 點" if is_long_first else ("拆開的半段" if m_.group(2) else "一般片段"),
                      "ssqi": round(ssqi(x), 4)})
ssqi_df = pd.DataFrame(ssqi_rows)
normal = ssqi_df[ssqi_df["kind"] == "一般片段"]
log(f"一般片段 {len(normal)} 段：SSQI 平均 {normal['ssqi'].mean():.3f}、"
    f"範圍 {normal['ssqi'].min():.3f} 至 {normal['ssqi'].max():.3f}")
neg = ssqi_df[(ssqi_df["ssqi"] <= 0) & (ssqi_df["kind"] != "長片段的前 2100 點")]
log(f"SSQI ≤ 0 的片段：{len(neg)} 段"
    + ("（與原論文「只存 SSQI > 0」一致）" if neg.empty else "：" + "、".join(
        f"{r.subject_id}_{r.label}（{r.ssqi:.3f}）" for r in neg.itertuples())))
for r in neg.itertuples():
    add_issue(r.subject_id, r.label, "SSQI ≤ 0", f"{r.ssqi:.4f}")

# ---------- 5b. 訊號飽和（明顯壞掉的片段）----------
log("\n--- 3b. 訊號飽和片段（明顯壞掉）---")
saturated = set()
sat_rows = []
for (sid, seg), x in sorted(signals.items()):
    v = np.asarray(x, dtype=float)[:L]
    r_hi = float(np.mean(v >= ADC_MAX))
    r_lo = float(np.mean(v <= ADC_MIN))
    if r_hi > 0 or r_lo > 0:
        saturated.add((sid, seg))
        add_issue(sid, seg, "訊號飽和", f"上限 {r_hi:.0%}、下限 {r_lo:.0%} 時間")
        log(f"{sid}_{seg}：{r_hi:.0%} 時間停在上限 {ADC_MAX}、{r_lo:.0%} 停在下限 {ADC_MIN}，"
            f"SSQI {ssqi(v):.3f} → 方法 2a、2b、3 刪除")
if not saturated:
    log("沒有飽和片段")

# ---------- 6. 受試者資訊檢查 ----------
for sid in sorted(set(meta.index) | set(seg_by_subject)):
    if sid not in meta.index:
        add_issue(sid, "", "缺少受試者資訊", "有訊號但 Excel 中沒有此編號")
        continue
    if len(seg_by_subject.get(sid, [])) != 3:
        add_issue(sid, "", "片段數異常", f"{sorted(seg_by_subject.get(sid, []))}")
    row = meta.loc[sid]
    for c in ("sbp", "dbp", "age", "sex"):
        if c in meta and pd.isna(row[c]):
            add_issue(sid, "", "受試者資訊缺值", c)
    if pd.notna(row["sbp"]) and not SBP_RANGE[0] <= row["sbp"] <= SBP_RANGE[1]:
        add_issue(sid, "", "SBP 超出合理範圍", f"{row['sbp']}")
    if pd.notna(row["dbp"]) and not DBP_RANGE[0] <= row["dbp"] <= DBP_RANGE[1]:
        add_issue(sid, "", "DBP 超出合理範圍", f"{row['dbp']}")

issues_df = pd.DataFrame(issues, columns=["subject_id", "segment", "type", "detail"])
log("\n--- 異常總表 ---")
log(issues_df.groupby("type").size().to_string() if len(issues_df) else "沒有發現異常")
other = issues_df[~issues_df["type"].str.contains("長度異常|數值完全相同|SSQI|訊號飽和")]
if len(other):
    log("[需要決定] 有長度與重複以外的異常，見 audit_issues.csv；程式不會自動排除")


# ---------- 7. 建立資料版本 ----------
def labels_for(version, sid):
    out = []
    for g in sorted(seg_by_subject.get(sid, [])):
        if (sid, str(g)) not in pieces:          # 長度不足等無法使用的片段
            continue
        k = long_k.get((sid, g))
        if version == "M3_dedup" and (sid, g) in drop_dup:
            continue
        if version in SAT_DROP_IN and (sid, g) in saturated:
            continue
        if k:
            mode = {"M1_all": "split", "M2a_drop4200": "drop",
                    "M2b_first2100": "first", "M3_dedup": M3_LONG}[version]
            if mode == "split":
                out += [f"{g}{chr(97 + i)}" for i in range(k)]
            elif mode == "first":
                out.append(str(g))
        else:
            out.append(str(g))
    return out


def fold_groups(version, kept):
    """仍保留的跨受試者重複片段，其受試者必須分在同一折。"""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            a = parent[a]
        return a
    for grp in cross_pairs:
        present = {s for s, g in grp if str(g) in kept.get(s, [])}
        present = sorted(present)
        for s in present[1:]:
            parent[find(s)] = find(present[0])
    return {s: min(t for t in kept if find(t) == find(s)) for s in kept}


log("\n--- 4. 資料版本 ---")
summary = []
for ver, desc in VERSIONS.items():
    kept = {sid: labels_for(ver, sid) for sid in meta.index}
    dropped_subj = [s for s, labs in kept.items() if not labs]
    kept = {s: labs for s, labs in kept.items() if labs}
    groups = fold_groups(ver, kept)
    rows = []
    for sid, labs in kept.items():
        r = meta.loc[sid]
        rows.append({"subject_id": sid, "segments": ";".join(labs), "n_segments": len(labs),
                     "fold_group": groups[sid], "sbp": r["sbp"], "dbp": r["dbp"],
                     "age": r.get("age", np.nan), "sex": r.get("sex", ""),
                     "bmi": r.get("bmi", np.nan), "hypertension": r.get("hypertension", "")})
    v = pd.DataFrame(rows)
    v.to_csv(os.path.join(OUT_DIR, f"subjects_{ver}.csv"), index=False, encoding="utf-8-sig")
    dist = v["n_segments"].value_counts().sort_index()
    same_fold = sorted({tuple(sorted(s for s in kept if groups[s] == g))
                        for g in set(groups.values())} - {(s,) for s in kept})
    summary.append({"version": ver, "description": desc, "subjects": len(v),
                    "segments": int(v["n_segments"].sum()), "max_segments": int(v["n_segments"].max()),
                    "segments_per_subject": "、".join(f"{k} 段 {n} 人" for k, n in dist.items()),
                    "same_fold": "；".join(",".join(map(str, g)) for g in same_fold) or "無",
                    "dropped_subjects": ",".join(map(str, dropped_subj)) or "無"})
    log(f"{desc}")
    log(f"  {len(v)} 人、{int(v['n_segments'].sum())} 段；每人片段數：{summary[-1]['segments_per_subject']}")
    log(f"  必須同折：{summary[-1]['same_fold']}；被排除的受試者：{summary[-1]['dropped_subjects']}")
pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "versions_summary.csv"),
                             index=False, encoding="utf-8-sig")
log("\n比較方式：M2a vs M2b 看 4200 點的處理；M2b vs M3 看相同片段的影響；M1 為完全不刪資料")

# ---------- 8. 存檔 ----------
issues_df.to_csv(os.path.join(OUT_DIR, "audit_issues.csv"), index=False, encoding="utf-8-sig")
ssqi_df.to_csv(os.path.join(OUT_DIR, "segment_ssqi.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(long_rows).to_csv(os.path.join(OUT_DIR, "long_segments.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(dup_rows).to_csv(os.path.join(OUT_DIR, "duplicate_groups.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT_DIR, "ppgbp_raw.pkl"), "wb") as f:
    pickle.dump({"signals": signals, "pieces": pieces, "meta": meta.reset_index(),
                 "expected_len": EXPECTED_LEN, "versions": list(VERSIONS),
                 "dup_groups": dup_groups, "long_k": long_k,
                 "saturated": sorted(saturated)}, f)
with open(os.path.join(OUT_DIR, "audit_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"\n完成，輸出在：{OUT_DIR}")
