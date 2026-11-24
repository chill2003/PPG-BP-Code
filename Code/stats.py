# -*- coding: utf-8 -*-
r"""
統計分析（論文 5.4 節；產生第六節的數字）

原則：分析單位是「受試者」。每位受試者在 3 次重複中的誤差先取平均，得到每人一個數字，
      再做配對比較；信賴區間用受試者層級（群集）bootstrap 2000 次。

  H3（非劣性，乾淨資料）：d_i = B5 的誤差 − B2 的誤差；95% CI 上界 < 1 mmHg → 非劣性成立
  H1（主要假設，優越性）：每人 ΔMAE = 受干擾時的誤差 − 乾淨時的誤差；
       d_i = B5 的 ΔMAE − B2 的 ΔMAE；單尾配對 Wilcoxon（B5 較小）+ bootstrap 95% CI
       主要條件：僅測試的干擾（高斯 0 dB、高頻突發、削波）合併，SBP
  H2：排序正確率（同一片段輕度分數 > 重度分數的比例），B5 對 B2，見過與未見家族分開
  其餘比較皆為探索性，只報告估計值與信賴區間

會用到的檔案（沒有的會自動略過，之後補上再執行即可）：
  乾淨預測：ppgbp_work\preds_B-1.csv、preds_B-demo.csv（baselines.py）
            ppgbp_work\cv\<版本>\preds_<模型>.csv（nested_cv.py）
  干擾測試：ppgbp_work\cv\<版本>\interference_<模型>.csv（11/30 的干擾測試程式產生）
            每列 = 一位受試者、一次重複、被干擾的一段、一種干擾與強度、一個種子，欄位：
            repeat, outer, subject_id, family, level, seg, seed,
            sbp_true, dbp_true, sbp_pred, dbp_pred, s_seg, w_seg, w_seg_clean

直接按 F5 執行。輸出：ppgbp_work\stats\（報告與表格）、Python\plot\（Bland–Altman 圖）
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============ 設定 ============
WORK_DIR = r"D:\PPG-BP Database\解壓資料\ppgbp_work"
FIG_DIR = r"D:\PPG-BP Database\Python\plot"
CV_DIR = os.path.join(WORK_DIR, "cv")               # 試跑結果請改成 "cv_quick"
MAIN_VERSION = "M3_dedup"
OTHER_VERSIONS = ["M1_all", "M2a_drop4200", "M2b_first2100"]
ALL_MODELS = ["B-1", "B-demo", "B0", "B1", "B2", "B3", "B4", "B5", "B6"]
PRIMARY = ("B5", "B2")                              # 主要比較：B5 對 B2（預先指定）
NI_MARGIN = 1.0                                     # 非劣性界限 (mmHg)
N_BOOT = 2000
SEED = 2026
# ==============================

OUT_DIR = os.path.join(WORK_DIR, "stats")
SEEN = {"gauss", "drift"}                           # 訓練時見過的干擾家族
TEST_ONLY = {"gauss_unseen", "burst", "clip"}       # 只在測試時出現
LIGHT_HEAVY = {"gauss": (20, 10), "drift": (0.3, 1.0), "burst": (100, 300), "clip": (90, 80)}
STAGE_ORDER = ["Normal", "Prehypertension", "Stage 1 hypertension", "Stage 2 hypertension"]

report = []


def log(msg=""):
    print(msg)
    report.append(msg)


# ---------- 讀檔 ----------
def preds_path(model, version):
    if model in ("B-1", "B-demo", "B0"):            # 不用 PPG 或不經巢狀 CV 的基準，存在 WORK_DIR
        return os.path.join(WORK_DIR, f"preds_{model}.csv")
    return os.path.join(CV_DIR, version, f"preds_{model}.csv")


def load_clean(version, models=ALL_MODELS):
    out = {}
    for m in models:
        p = preds_path(m, version)
        if os.path.exists(p):
            out[m] = pd.read_csv(p)
    return out


def load_interference(version, models=ALL_MODELS):
    out = {}
    for m in models:
        p = os.path.join(CV_DIR, version, f"interference_{m}.csv")
        if os.path.exists(p):
            out[m] = pd.read_csv(p)
    return out


# ---------- 受試者層級的數字 ----------
def subject_error(df):
    """每人的絕對誤差：先在每次重複算 |預測 − 真實|，再對重複取平均。回傳 index = subject_id。"""
    e = df.assign(sbp=(df.sbp_pred - df.sbp_true).abs(), dbp=(df.dbp_pred - df.dbp_true).abs())
    return e.groupby("subject_id")[["sbp", "dbp"]].mean()


def subject_mean_pred(df):
    """每人的平均預測與真實值（Bland–Altman 用）。"""
    return df.groupby("subject_id")[["sbp_pred", "dbp_pred", "sbp_true", "dbp_true"]].mean()


def bootstrap_ci(d, n_boot=N_BOOT, seed=SEED):
    """受試者層級 bootstrap：每次抽回 n 人，算差值平均；回傳 (平均, 2.5%, 97.5%)。"""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def paired(a, b):
    """兩個模型的每人數字，依受試者對齊後相減（a − b）。"""
    a, b = a.align(b, join="inner")
    return a - b


def fmt_ci(m, lo, hi):
    return f"{m:+.2f}（95% CI {lo:+.2f} 至 {hi:+.2f}）"


# ---------- 6.2：乾淨資料 ----------
def table_clean(clean):
    """各模型 MAE、RMSE、偏差：每次重複算一次（全部受試者），再取 3 次的平均 ± 標準差。"""
    rows = []
    for m, df in clean.items():
        per = []
        for r, g in df.groupby("repeat"):
            rec = {}
            for t in ("sbp", "dbp"):
                e = g[f"{t}_pred"] - g[f"{t}_true"]
                yt, yp = g[f"{t}_true"], g[f"{t}_pred"]
                rec.update({f"{t}_mae": e.abs().mean(), f"{t}_rmse": np.sqrt((e ** 2).mean()), f"{t}_bias": e.mean(),
                            f"{t}_sde": e.std(ddof=1),                                   # 誤差標準差（AAMI 用）
                            f"{t}_r": yt.corr(yp) if yp.std() > 0 else np.nan,           # 預測固定時無相關係數
                            f"{t}_r2": 1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum()})
            per.append(rec)
        per = pd.DataFrame(per)
        row = {"model": m}
        for c in per.columns:
            row[c] = per[c].mean()
            row[c + "_sd"] = per[c].std(ddof=1) if len(per) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def h3_noninferiority(clean):
    a, b = PRIMARY
    if a not in clean or b not in clean:
        log(f"H3：缺少 {a} 或 {b} 的乾淨預測，略過")
        return None
    ea, eb = subject_error(clean[a]), subject_error(clean[b])
    res = {}
    for t, name in (("sbp", "SBP"), ("dbp", "DBP")):
        d = paired(ea[t], eb[t])
        m, lo, hi = bootstrap_ci(d)
        ok = hi < NI_MARGIN
        p = wilcoxon(d).pvalue if (d != 0).any() else np.nan
        res[t] = (m, lo, hi, ok, p, len(d))
        log(f"  {name}：{a} − {b} MAE 差值 {fmt_ci(m, lo, hi)}；上界 {'<' if ok else '≥'} {NI_MARGIN} mmHg → "
            f"非劣性{'成立' if ok else '不成立'}（雙尾 Wilcoxon p = {p:.3g}，探索性；n = {len(d)}）")
    return res


def exploratory_clean(clean, ref="B5"):
    """B5 對其他各模型的乾淨 SBP、DBP MAE 差值（探索性）。"""
    if ref not in clean:
        return pd.DataFrame()
    e_ref = subject_error(clean[ref])
    rows = []
    for m in clean:
        if m == ref:
            continue
        e = subject_error(clean[m])
        row = {"comparison": f"{ref} − {m}"}
        for t in ("sbp", "dbp"):
            mm, lo, hi = bootstrap_ci(paired(e_ref[t], e[t]))
            row.update({f"{t}_diff": mm, f"{t}_lo": lo, f"{t}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def bland_altman(clean, models, version):
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    models = [m for m in models if m in clean]
    if not models:
        return []
    rows = []
    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 7), squeeze=False)
    for j, m in enumerate(models):
        sp = subject_mean_pred(clean[m])
        for i, t in enumerate(("sbp", "dbp")):
            mean = (sp[f"{t}_pred"] + sp[f"{t}_true"]) / 2
            diff = sp[f"{t}_pred"] - sp[f"{t}_true"]
            bias, sd = diff.mean(), diff.std(ddof=1)
            lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
            rows.append({"model": m, "bp": t.upper(), "bias": bias, "loa_low": lo, "loa_high": hi})
            ax = axes[i, j]
            ax.scatter(mean, diff, s=10, alpha=0.6)
            for v, ls in ((bias, "-"), (lo, "--"), (hi, "--")):
                ax.axhline(v, color="#d62728" if ls == "--" else "black", ls=ls, lw=1)
            ax.set_title(f"{m}：{t.upper()}（偏差 {bias:+.1f}，一致性界限 {lo:.1f} 至 {hi:.1f}）", fontsize=9)
            ax.set_xlabel("(預測 + 真實) / 2 (mmHg)")
            ax.set_ylabel("預測 − 真實 (mmHg)")
            ax.grid(alpha=0.3)
    fig.suptitle(f"Bland–Altman 圖（{version}，每人 3 次重複的平均預測）")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"Bland-Altman_{version}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log(f"  圖已存：{out}")
    return rows


def by_stage(clean, models):
    """依血壓分期的 MAE（6.5）。分期取自 prep npz 的 hypertension 欄位。"""
    d = np.load(os.path.join(WORK_DIR, f"prep_{MAIN_VERSION}.npz"))
    stage = pd.Series(d["hypertension"], index=d["subject_id"], name="stage")
    rows = []
    for m in models:
        if m not in clean:
            continue
        e = subject_error(clean[m]).join(stage)
        mp = subject_mean_pred(clean[m]).join(stage)          # 每人 3 次重複的平均預測與真實值
        for s, g in e.groupby("stage"):
            p = mp[mp.stage == s]
            rows.append({"model": m, "stage": s, "n": len(g), "sbp_mae": g.sbp.mean(), "dbp_mae": g.dbp.mean(),
                         "sbp_true_mean": p.sbp_true.mean(), "sbp_pred_mean": p.sbp_pred.mean(),
                         "dbp_true_mean": p.dbp_true.mean(), "dbp_pred_mean": p.dbp_pred.mean()})
    return pd.DataFrame(rows)


def plot_by_stage(st, version):
    """依血壓分期的 SBP、DBP MAE 長條圖（6.5）。"""
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    names = {"Normal": "正常", "Prehypertension": "高血壓前期",
             "Stage 1 hypertension": "第一期", "Stage 2 hypertension": "第二期"}
    stages = [s for s in STAGE_ORDER if s in set(st.stage)]
    models = list(dict.fromkeys(st.model))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / len(models)
    for ax, t in zip(axes, ("sbp", "dbp")):
        for k, m in enumerate(models):
            g = st[st.model == m].set_index("stage").loc[stages]
            xs = np.arange(len(stages)) + (k - (len(models) - 1) / 2) * width
            ax.bar(xs, g[f"{t}_mae"], width, label=m)
        n = st[st.model == models[0]].set_index("stage").loc[stages, "n"]
        ax.set_xticks(np.arange(len(stages)), [f"{names[s]}\n(n = {c})" for s, c in zip(stages, n)])
        ax.set_ylabel(f"{t.upper()} MAE (mmHg)")
        ax.set_title(f"依血壓分期的 {t.upper()} 誤差")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"依血壓分期誤差_{version}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log(f"  圖已存：{out}")


# ---------- 附錄 ----------
def appendix_seen(inter, clean):
    """附錄：每一種干擾條件（家族 × 強度）的 ΔMAE，含訓練時見過的強度。"""
    rows = []
    for m, df in inter.items():
        if m not in clean:
            continue
        e_clean = subject_error(clean[m])
        for (fam, lv), g in df.groupby(["family", "level"]):
            corr = g.assign(sbp=(g.sbp_pred - g.sbp_true).abs(), dbp=(g.dbp_pred - g.dbp_true).abs())
            corr = corr.groupby("subject_id")[["sbp", "dbp"]].mean()
            d = corr - e_clean.loc[corr.index]
            rows.append({"model": m, "family": fam, "level": lv, "seen": fam in SEEN,
                         "dsbp_mae": d.sbp.mean(), "ddbp_mae": d.dbp.mean()})
    return pd.DataFrame(rows)


def appendix_versions_dbp(main_clean, main_inter):
    """附錄：6.6 的 DBP 版本。"""
    rows = []
    for v in [MAIN_VERSION] + OTHER_VERSIONS:
        clean = main_clean if v == MAIN_VERSION else load_clean(v, ["B2", "B5"])
        inter = main_inter if v == MAIN_VERSION else load_interference(v, ["B2", "B5"])
        if not all(m in clean for m in ("B2", "B5")):
            continue
        row = {"version": v, "B2_clean_dbp_mae": subject_error(clean["B2"]).dbp.mean(),
               "B5_clean_dbp_mae": subject_error(clean["B5"]).dbp.mean()}
        if all(m in inter for m in ("B2", "B5")):
            for name, fams in (("gauss0", ["gauss_unseen"]), ("unseen_family", ["burst", "clip"])):
                d = paired(subject_delta(inter["B5"], clean["B5"], fams).dbp,
                           subject_delta(inter["B2"], clean["B2"], fams).dbp)
                mm, lo, hi = bootstrap_ci(d)
                row.update({f"{name}_diff": mm, f"{name}_lo": lo, f"{name}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- 6.3、6.4：干擾測試 ----------
def condition_group(family):
    return "seen" if family in SEEN else "test_only"


def subject_delta(inter, clean_df, families):
    """每人 ΔMAE：受指定干擾時的平均絕對誤差 − 乾淨時的平均絕對誤差。"""
    g = inter[inter.family.isin(families)]
    if g.empty:
        return None
    corr = g.assign(sbp=(g.sbp_pred - g.sbp_true).abs(), dbp=(g.dbp_pred - g.dbp_true).abs())
    corr = corr.groupby("subject_id")[["sbp", "dbp"]].mean()
    return corr - subject_error(clean_df).loc[corr.index]


def table_delta(inter, clean):
    cols = {"gauss_unseen": ["gauss_unseen"], "burst": ["burst"], "clip": ["clip"],
            "test_only": sorted(TEST_ONLY), "seen": sorted(SEEN)}
    rows = []
    for m, df in inter.items():
        if m not in clean:
            continue
        row = {"model": m}
        for name, fams in cols.items():
            dlt = subject_delta(df, clean[m], fams)
            if dlt is not None:
                row[f"{name}_sbp"] = dlt.sbp.mean()
                row[f"{name}_dbp"] = dlt.dbp.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def h1_primary(inter, clean):
    a, b = PRIMARY
    if not all(m in inter and m in clean for m in PRIMARY):
        log(f"H1：缺少 {a} 或 {b} 的干擾測試結果（interference_*.csv），待 interference.py 執行完成後再執行")
        return None
    res = {}
    for label, fams in (("主要：僅測試的干擾合併", TEST_ONLY), ("見過的干擾合併（次要）", SEEN)):
        da, db = subject_delta(inter[a], clean[a], fams), subject_delta(inter[b], clean[b], fams)
        for t, name in (("sbp", "SBP"), ("dbp", "DBP")):
            d = paired(da[t], db[t])
            m, lo, hi = bootstrap_ci(d)
            p = wilcoxon(d, alternative="less").pvalue if (d != 0).any() else np.nan
            res[(label, t)] = (m, lo, hi, p)
            log(f"  {label}，{name}：{a} − {b} 的 ΔMAE 差值 {fmt_ci(m, lo, hi)}；單尾 Wilcoxon p = {p:.3g}")
    return res


def ranking(inter_df):
    """每個 (重複, 受試者, 片段, 種子, 家族)：輕度分數 > 重度分數 → 1。回傳每人的正確率。"""
    rows = []
    for fam, (light, heavy) in LIGHT_HEAVY.items():
        g = inter_df[inter_df.family == fam]
        if g.empty:
            continue
        piv = g.pivot_table(index=["repeat", "subject_id", "seg", "seed"], columns="level", values="s_seg")
        if light not in piv.columns or heavy not in piv.columns:
            continue
        ok = (piv[light] > piv[heavy]).astype(float).rename("correct").reset_index()
        ok["group"] = condition_group(fam)
        rows.append(ok)
    if not rows:
        return None
    r = pd.concat(rows)
    return r.groupby(["group", "subject_id"]).correct.mean().unstack(0)


def table_reliability(inter):
    rows = []
    for m, df in inter.items():
        row = {"model": m}
        rk = ranking(df)
        if rk is not None:
            for g in ("seen", "test_only"):
                if g in rk:
                    row[f"rank_acc_{g}"] = rk[g].mean()
        dw = (df.w_seg - df.w_seg_clean).groupby(df.subject_id).mean()
        row["delta_w"] = dw.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def h2(inter):
    a, b = PRIMARY
    if a not in inter or b not in inter:
        log(f"H2：缺少 {a} 或 {b} 的干擾測試結果，待 interference.py 執行完成後再執行")
        return
    ra, rb = ranking(inter[a]), ranking(inter[b])
    for g, name in (("seen", "見過家族"), ("test_only", "未見家族")):
        if g in ra and g in rb:
            d = paired(ra[g], rb[g])
            m, lo, hi = bootstrap_ci(d)
            p = wilcoxon(d, alternative="greater").pvalue if (d != 0).any() else np.nan
            log(f"  {name}：排序正確率 {a} {ra[g].mean():.1%}、{b} {rb[g].mean():.1%}；"
                f"差值 {fmt_ci(m, lo, hi)}（比例）；單尾 Wilcoxon p = {p:.3g}")


# ---------- 6.4、6.5：刪除檢驗與重複片段測試 ----------
def load_extra(version, models=ALL_MODELS):
    out = {}
    for m in models:
        p = os.path.join(CV_DIR, version, f"extra_{m}.csv")
        if os.path.exists(p):
            out[m] = pd.read_csv(p)
    return out


def table_extra(extra):
    """
    刪除檢驗：拿掉權重最高（或最低）的一段後，每人誤差增加多少（只算至少 2 段的人）。
    若權重有意義，刪掉最高權重段應該比刪掉最低權重段傷害更大。
    重複片段：{x1, x1, x2} 與 {x1, x2} 的預測差異（絕對值）。
    """
    rows = []
    for m, df in extra.items():
        g = df[df.n_seg >= 2]
        row = {"model": m, "n_subjects": g.subject_id.nunique()}
        for t in ("sbp", "dbp"):
            err = lambda c: (g[f"{t}_{c}"] - g[f"{t}_true"]).abs()
            per = pd.DataFrame({"full": err("full"), "dmax": err("dropmax"), "dmin": err("dropmin"),
                                "dup": (g[f"{t}_dup"] - g[f"{t}_pair"]).abs(), "sid": g.subject_id})
            per = per.groupby("sid").mean()
            row[f"{t}_dmae_dropmax"] = (per.dmax - per.full).mean()
            row[f"{t}_dmae_dropmin"] = (per.dmin - per.full).mean()
            row[f"{t}_dup_change"] = per.dup.mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- 6.6：資料版本對照 ----------
def version_table(main_clean, main_inter):
    rows = []
    for v in [MAIN_VERSION] + OTHER_VERSIONS:
        clean = main_clean if v == MAIN_VERSION else load_clean(v, ["B2", "B5"])
        inter = main_inter if v == MAIN_VERSION else load_interference(v, ["B2", "B5"])
        if not all(m in clean for m in ("B2", "B5")):
            rows.append({"version": v, "note": "尚未完成"})
            continue
        row = {"version": v,
               "B2_clean_sbp_mae": subject_error(clean["B2"]).sbp.mean(),
               "B5_clean_sbp_mae": subject_error(clean["B5"]).sbp.mean()}
        if all(m in inter for m in ("B2", "B5")):
            for name, fams in (("gauss0", ["gauss_unseen"]), ("unseen_family", ["burst", "clip"])):
                d = paired(subject_delta(inter["B5"], clean["B5"], fams).sbp,
                           subject_delta(inter["B2"], clean["B2"], fams).sbp)
                m, lo, hi = bootstrap_ci(d)
                row.update({f"{name}_diff": m, f"{name}_lo": lo, f"{name}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def outliers_m1(models=("B2", "B5"), ids=(231, 24)):
    """M1 中受試者 231 與 24 的個別誤差，以及在全體中的百分位（6.6 節要求）。"""
    clean = load_clean("M1_all", list(models))
    for m in models:
        if m not in clean:
            continue
        e = subject_error(clean[m])
        for s in ids:
            if s in e.index:
                pct = (e.sbp < e.sbp[s]).mean()
                log(f"  M1 {m}：受試者 {s} SBP 誤差 {e.sbp[s]:.1f} mmHg（第 {pct:.0%} 百分位）、"
                    f"DBP {e.dbp[s]:.1f} mmHg")


# ---------- 主程式 ----------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.set_option("display.width", 200)
    clean = load_clean(MAIN_VERSION)
    inter = load_interference(MAIN_VERSION)

    log("=" * 70)
    log(f"統計分析報告：主分析 {MAIN_VERSION}")
    log("=" * 70)
    log(f"乾淨預測：{'、'.join(clean) or '無'}")
    log(f"干擾測試：{'、'.join(inter) or '無（待 interference.py）'}")
    missing = [m for m in ALL_MODELS if m not in clean]
    if missing:
        log(f"尚未有結果：{'、'.join(missing)}")

    log("\n【6.2】乾淨資料（3 次重複的平均 ± 標準差，mmHg）")
    t62 = table_clean(clean)
    t62 = t62.set_index("model").loc[[m for m in ALL_MODELS if m in clean]].reset_index()
    t62.to_csv(os.path.join(OUT_DIR, "table_6_2_clean.csv"), index=False, encoding="utf-8-sig")
    log(f"{'':8s}" + "".join(f"{h:>15s}" for h in ("SBP MAE", "SBP RMSE", "SBP 偏差", "DBP MAE", "DBP RMSE", "DBP 偏差")))
    for _, s in t62.iterrows():
        log(f"{s['model']:8s}" + "".join(f"{s[c]:>8.2f} ± {s[c + '_sd']:.2f}" for c in
                                       ("sbp_mae", "sbp_rmse", "sbp_bias", "dbp_mae", "dbp_rmse", "dbp_bias")))

    log("\n【6.2 續】誤差標準差（SDE，AAMI 標準 ≤ 8 mmHg）、Pearson r、R²（每次重複算一次，3 次的平均 ± 標準差）")
    log(f"{'':8s}" + "".join(f"{h:>15s}" for h in ("SBP SDE", "SBP r", "SBP R²", "DBP SDE", "DBP r", "DBP R²")))
    for _, s in t62.iterrows():
        cells = []
        for c in ("sbp_sde", "sbp_r", "sbp_r2", "dbp_sde", "dbp_r", "dbp_r2"):
            cells.append("—" if pd.isna(s[c]) else f"{s[c]:.2f} ± {s[c + '_sd']:.2f}")
        log(f"{s['model']:8s}" + "".join(f"{x:>15s}" for x in cells))

    log("\n【H3】非劣性（乾淨資料，主要比較 B5 − B2）")
    h3_noninferiority(clean)

    log("\n【探索性】B5 與其他模型的乾淨 MAE 差值（負值 = B5 較準）")
    ex = exploratory_clean(clean)
    if not ex.empty:
        ex.to_csv(os.path.join(OUT_DIR, "exploratory_clean.csv"), index=False, encoding="utf-8-sig")
        for _, r in ex.iterrows():
            log(f"  {r.comparison}：SBP {fmt_ci(r.sbp_diff, r.sbp_lo, r.sbp_hi)}；DBP {fmt_ci(r.dbp_diff, r.dbp_lo, r.dbp_hi)}")

    log("\n【Bland–Altman】")
    ba = pd.DataFrame(bland_altman(clean, ["B-demo", "B2", "B5"], MAIN_VERSION))
    if not ba.empty:
        ba.to_csv(os.path.join(OUT_DIR, "bland_altman.csv"), index=False, encoding="utf-8-sig")
        for _, r in ba.iterrows():
            log(f"  {r.model} {r.bp}：偏差 {r.bias:+.2f}，95% 一致性界限 {r.loa_low:.1f} 至 {r.loa_high:.1f} mmHg")

    log("\n【6.5】依血壓分期的 MAE")
    st = by_stage(clean, ["B-demo", "B2", "B5"])
    if not st.empty:
        st.to_csv(os.path.join(OUT_DIR, "by_stage.csv"), index=False, encoding="utf-8-sig")
        for _, r in st.iterrows():
            log(f"  {r.model:7s}{r.stage:24s} n = {r.n:3d}  SBP {r.sbp_mae:5.1f}  DBP {r.dbp_mae:5.1f}"
                f"  ｜平均真實 / 預測 SBP {r.sbp_true_mean:5.1f} / {r.sbp_pred_mean:5.1f}"
                f"，DBP {r.dbp_true_mean:5.1f} / {r.dbp_pred_mean:5.1f}")
        plot_by_stage(st, MAIN_VERSION)

    log("\n【6.3 / H1】單段受干擾時的 ΔMAE")
    if inter:
        t63 = table_delta(inter, clean)
        t63.to_csv(os.path.join(OUT_DIR, "table_6_3_delta.csv"), index=False, encoding="utf-8-sig")
        log(t63.round(2).to_string(index=False))
    h1_primary(inter, clean)

    log("\n【6.4 / H2】可靠度分數")
    if inter:
        t64 = table_reliability(inter)
        t64.to_csv(os.path.join(OUT_DIR, "table_6_4_reliability.csv"), index=False, encoding="utf-8-sig")
        log(t64.round(3).to_string(index=False))
    h2(inter)

    log("\n【6.4、6.5】刪除檢驗與重複片段測試（至少 2 段的受試者）")
    extra = load_extra(MAIN_VERSION)
    if extra:
        te = table_extra(extra)
        te.to_csv(os.path.join(OUT_DIR, "table_extra.csv"), index=False, encoding="utf-8-sig")
        log("  dmae_dropmax / dropmin：刪掉權重最高 / 最低的一段後，MAE 增加量；dup_change：{x1,x1,x2} 與 {x1,x2} 的預測差異")
        log(te.round(2).to_string(index=False))
    else:
        log("  尚無 extra_*.csv，待干擾測試完成")

    log("\n【6.6】資料版本對照（SBP）")
    vt = version_table(clean, inter)
    vt.to_csv(os.path.join(OUT_DIR, "table_6_6_versions.csv"), index=False, encoding="utf-8-sig")
    log(vt.round(2).to_string(index=False))
    outliers_m1()

    log("\n【附錄】各干擾條件的 ΔMAE（含訓練時見過的強度）")
    if inter:
        ap = appendix_seen(inter, clean)
        ap.to_csv(os.path.join(OUT_DIR, "appendix_conditions.csv"), index=False, encoding="utf-8-sig")
        piv = ap.pivot_table(index=["family", "level"], columns="model", values="dsbp_mae")
        log("  ΔSBP MAE（mmHg）")
        log(piv.round(2).to_string())
        piv = ap.pivot_table(index=["family", "level"], columns="model", values="ddbp_mae")
        log("  ΔDBP MAE（mmHg）")
        log(piv.round(2).to_string())
    log("\n【附錄】資料版本對照（DBP）")
    vd = appendix_versions_dbp(clean, inter)
    vd.to_csv(os.path.join(OUT_DIR, "appendix_versions_dbp.csv"), index=False, encoding="utf-8-sig")
    log(vd.round(2).to_string(index=False))

    with open(os.path.join(OUT_DIR, "stats_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n報告與表格已存：{OUT_DIR}")
