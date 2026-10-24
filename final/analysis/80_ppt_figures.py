"""80 - Presentation figures (slides 2-18) and every number quoted on the slides.
One style for all charts: white surface, one y-axis, recessive grid, main series blue, high-risk red, comparison yellow
(team palette #015086 / #d52229 / #ffcf23), context grey, red dashed boxes only as call-outs.
Output: docs/ppt/fig/sXX_*.png (200 dpi) + the numbers printed here (log: outputs/logs/80_ppt_figures.log)."""
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Rectangle
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from common import DATA, ROOT, WORK, parse_num
from cvkit import REPAIRS, T0, t, test_view, tr, y, year

OUT = ROOT / "docs" / "ppt" / "fig"
OUT.mkdir(parents=True, exist_ok=True)
# team palette: blue = main data, red = high risk / call-out boxes, yellow = comparison / low risk,
# light blue (tint of the main blue) = de-emphasised bars, like the dark/light blues of the DAC 2025 winner deck
BLUE, RED, YELLOW, LIGHT, INK, MUTED = "#015086", "#d52229", "#ffcf23", "#A9C1D6", "#1F2933", "#616E7C"
GREY = LIGHT
THR = np.log(0.6)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": "#CBD2D9", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#EEF1F4", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


def pct(v):
    return f"{100 * v:.1f}%"


FT, FE = test_view("v7")
F = FT.reset_index(drop=True)
date = pd.to_datetime([T0 + pd.DateOffset(months=int(k)) for k in range(int(t.max()) + 2)])
tmonth = tr.month.values % 12 + 1                     # calendar month of the TARGET (next month)
dry = F.anom0.values < THR                            # supply this month already below 0.6 x climatology

# ---------------- S2: share of HUCs in water stress, month by month ----------------
share = pd.Series(y).groupby(t).mean()
tgt_date = date[share.index + 1]
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.bar(tgt_date, share.values, width=25, color=BLUE)
for lo, hi, lab in [("2002-03", "2003-01", "Kekeringan 2002"), ("2012-06", "2013-04", "Kekeringan 2012")]:
    ax.add_patch(Rectangle((pd.Timestamp(lo), 0), pd.Timestamp(hi) - pd.Timestamp(lo), 0.46, fill=False, ls="--", ec=RED, lw=1.2))
    ax.text(pd.Timestamp(lo), 0.47, lab, color=RED, fontsize=11)
ax.axhline(share.mean(), color=MUTED, ls=":", lw=1)
ax.text(pd.Timestamp("2009-06"), 0.47, f"garis titik = rata-rata {pct(share.mean())}", color=MUTED, ha="center", fontsize=10)
ax.set_ylabel("Porsi sub-DAS water stress")
ax.set_ylim(0, 0.52)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
save(fig, "s02_share_per_month.png")
print(f"[S2] mean share {share.mean():.3f}; max {share.max():.3f} at {tgt_date[share.values.argmax()]:%Y-%m}; "
      f"min {share.min():.3f} at {tgt_date[share.values.argmin()]:%Y-%m}")

# ---------------- S3: yearly share + heterogeneity across HUCs ----------------
ys = pd.Series(y).groupby(year).mean()
labels = [f"{2000 + k}/{str(2001 + k)[2:]}" for k in ys.index]
S3_HI, S3_MID, S3_LO = "#d52229", "#015086", "#ffcf23"  # tim palette: red = high stress, blue = middle, yellow = low
fig, ax = plt.subplots(1, 2, figsize=(13, 4.9))
cols = [S3_HI if v >= ys.quantile(.8) else S3_LO if v <= ys.quantile(.2) else S3_MID for v in ys.values]
ax[0].bar(labels, ys.values, color=cols)
for i, v in enumerate(ys.values):
    ax[0].text(i, v + 0.005, f"{v:.0%}", ha="center", fontsize=9, color=INK)
ax[0].set_ylabel("Porsi baris water stress"); ax[0].tick_params(axis="x", rotation=60, labelsize=9)
ax[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
hr = pd.Series(y).groupby(tr.id.values).mean()
lo_q, hi_q = hr.quantile(.05), hr.quantile(.95)
cnt, edges_h = np.histogram(hr.values, bins=40)
mid = (edges_h[:-1] + edges_h[1:]) / 2
ax[1].bar(mid, cnt, width=np.diff(edges_h), color=[S3_HI if c > hi_q else S3_LO if c < lo_q else S3_MID for c in mid],
          edgecolor="white", linewidth=0.8)
for q in (lo_q, hi_q):
    ax[1].axvline(q, color=INK, ls="--", lw=1)
ax[1].text(hi_q, cnt.max() * .95, f"  p95 = {hi_q:.0%}", color=INK, fontsize=10)
ax[1].text(lo_q, cnt.max() * .95, f"p5 = {lo_q:.0%}  ", color=INK, fontsize=10, ha="right")
ax[1].set_xlabel("Frekuensi water stress per sub-DAS (2000-2014)"); ax[1].set_ylabel("Jumlah sub-DAS")
ax[1].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
handles = [Rectangle((0, 0), 1, 1, color=c) for c in (S3_HI, S3_MID, S3_LO)]
fig.legend(handles, ["Tinggi (tahun 20% teratas / sub-DAS di atas p95)", "Menengah",
                     "Rendah (tahun 20% terbawah / sub-DAS di bawah p5)"], loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "s03_yearly_and_per_huc.png")
print(f"[S3] yearly share {ys.min():.3f} ({labels[ys.values.argmin()]}) .. {ys.max():.3f} ({labels[ys.values.argmax()]}); "
      f"per-HUC rate p5 {hr.quantile(.05):.3f} median {hr.median():.3f} p95 {hr.quantile(.95):.3f}; HUCs {len(hr)}")

# ---------------- S6: raw rows (dirty values) + missing / sentinel per column family ----------------
raw = pd.read_csv(DATA / "train.csv", dtype=str, nrows=40000)
show = ["id", "origin_id", "month", "q_ws_cum_m3_lag0", "q_ws_cum_m3_lag1", "q_bf_inc_m3_lag0", "clim_median_next", "label"]
v = raw[show]
bad = (v.month.str.contains("[A-Za-z]", na=False) | v.q_ws_cum_m3_lag0.str.contains(",", na=False)
       | v.q_ws_cum_m3_lag1.isin(["-9999", "-9,999"]) | v.q_bf_inc_m3_lag0.isna() | v.clim_median_next.str.contains("e", na=False))
pick = pd.concat([v[bad & v.month.str.contains("[A-Za-z]", na=False)].head(2), v[bad & v.q_ws_cum_m3_lag0.str.contains(",", na=False)].head(2),
                  v[bad & v.q_ws_cum_m3_lag1.isin(["-9999", "-9,999"])].head(2), v[bad & v.q_bf_inc_m3_lag0.isna()].head(1),
                  v[~bad].head(1)]).drop_duplicates()
cells = pick.fillna("NaN").to_numpy()
fig, ax = plt.subplots(figsize=(13, 3.4)); ax.axis("off")
tb = ax.table(cellText=cells, colLabels=show, loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.6)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#E4E7EB")
    if r == 0:
        cell.set_text_props(weight="bold", color=INK); cell.set_facecolor("#F5F7FA")
    else:
        s = str(cells[r - 1, c])
        if s == "NaN" or s in ("-9999", "-9,999") or ("," in s and c >= 3) or (c == 2 and any(ch.isalpha() for ch in s)):
            cell.set_text_props(color=RED, weight="bold")
save(fig, "s06_raw_rows.png")

fam = {"Supply (ws)": "q_ws_cum_m3", "Baseflow": "q_bf_inc_m3", "Quickflow": "q_qf_inc_m3", "Irigasi": "q_ir_wd_inc_m3",
       "Thermoelectric": "q_th_wd_inc_m3", "Public supply": "q_ps_wd_inc_m3"}
rows = []
for k, f in fam.items():
    X = tr[[f"{f}_lag{j}" for j in range(12)]]
    rows.append((k, X.isna().to_numpy().mean(), (X == -9999).to_numpy().mean()))
M = pd.DataFrame(rows, columns=["fam", "nan", "sent"])
fig, ax = plt.subplots(figsize=(7, 4.2))
yy = np.arange(len(M))
ax.barh(yy + 0.2, M.nan * 100, 0.38, color=BLUE, label="Nilai hilang (NaN)")
ax.barh(yy - 0.2, M.sent * 100, 0.38, color=YELLOW, label="Sentinel -9999")
for i, r in M.iterrows():
    ax.text(r.nan * 100 + 0.03, i + 0.2, f"{r.nan:.2%}", va="center", fontsize=9, color=INK)
    ax.text(r.sent * 100 + 0.03, i - 0.2, f"{r.sent:.2%}", va="center", fontsize=9, color=INK)
ax.set_yticks(yy, M.fam); ax.set_xlabel("Persen sel (12 lag)"); ax.set_xlim(0, 2.25)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
ax.grid(axis="y", visible=False)
save(fig, "s06_missing_sentinel.png")
print("[S6] NaN / sentinel share per family:\n" + M.round(4).to_string(index=False))
print(f"[S6] rows with >=1 NaN in lag columns: {tr.filter(like='_lag').isna().any(axis=1).mean():.3f}")

# ---------------- S7: repair ladder ----------------
steps = ["Interpolasi\n(v6)", "+ Nol sisipan\ndulu", "+ Cek bulan\nbadai", "+ Neraca air\njaringan sungai"]
rec, dmg = [61.3, 61.5, 61.5, 84.0], [0.127, 0.106, 0.086, 0.055]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].bar(steps, rec, color=[GREY, GREY, GREY, BLUE])
for i, v_ in enumerate(rec):
    ax[0].text(i, v_ + 1.5, f"{v_:.1f}%", ha="center", color=INK, fontsize=11)
ax[0].set_ylabel("Sel supply korup yang pulih (galat <= 5%)"); ax[0].set_ylim(0, 100)
ax[1].bar(steps, dmg, color=[GREY, GREY, GREY, BLUE])
for i, v_ in enumerate(dmg):
    ax[1].text(i, v_ + 0.004, f"{v_:.3f}%", ha="center", color=INK, fontsize=11)
ax[1].set_ylabel("Sel bersih yang ikut rusak")
save(fig, "s07_repair_ladder.png")

# ---------------- S8: label definition + test rescaling ----------------
long = pd.concat([pd.DataFrame({"id": tr.id.values, "m": t - k, "v": tr[f"q_ws_cum_m3_lag{k}"].abs().where(tr[f"q_ws_cum_m3_lag{k}"] != -9999).values})
                  for k in range(12)]).dropna()
series = long.groupby(["id", "m"]).v.median()
ws_next = series.reindex(pd.MultiIndex.from_arrays([tr.id.values, t + 1])).to_numpy()
ws_now = series.reindex(pd.MultiIndex.from_arrays([tr.id.values, t])).to_numpy()
clim_next = (ws_now + 1) / np.exp(F.margin_raw_persist.values + THR) - 1
ratio = ws_next / clim_next
ok = np.isfinite(ratio) & (clim_next > 0)
edges = np.array([0, .2, .4, .5, .55, .58, .59, .60, .61, .62, .65, .7, .8, 1, 1.5, np.inf])
cut = pd.cut(ratio[ok], edges, right=True)
rate = pd.Series(y[ok]).groupby(cut, observed=True).mean()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
lab = [f"{iv.left:g}-{iv.right:g}" if np.isfinite(iv.right) else f">{iv.left:g}" for iv in rate.index]
ax[0].bar(lab, rate.values, color=[BLUE if iv.right <= 0.6 else GREY for iv in rate.index])
ax[0].axvline(7.5, color=RED, ls="--"); ax[0].text(7.6, 0.9, "ambang 0.6", color=RED, fontsize=10)
ax[0].set_xlabel("Pasokan bulan depan / median klimatologi bulan itu"); ax[0].set_ylabel("Porsi label = 1")
ax[0].tick_params(axis="x", rotation=60, labelsize=9); ax[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
fac = pd.read_csv(WORK / "test_huc_factor.csv").factor
fac = fac[(fac > 0.2) & (fac < 5)]  # drop HUCs whose climatology is also shifted 1000x (fixed separately)
ax[1].hist(np.log2(fac.dropna()), bins=40, color=BLUE, edgecolor="white", linewidth=0.8)
ax[1].set_xticks(np.log2([0.25, 0.5, 1, 2, 4]), ["0.25x", "0.5x", "1x", "2x", "4x"])
ax[1].set_xlabel("Faktor skala acak per sub-DAS di test (skala log)"); ax[1].set_ylabel("Jumlah sub-DAS")
save(fig, "s08_label_and_rescale.png")
acc = ((ratio[ok] <= 0.6) == (y[ok] == 1)).mean()
print(f"[S8] rule 'ratio <= 0.6' matches the label on {acc:.5f} of {ok.sum()} rows; rate in (0.59,0.60] "
      f"{rate.iloc[6]:.3f}, (0.60,0.61] {rate.iloc[7]:.4f}; test factor p5-p95 {fac.quantile(.05):.2f}-{fac.quantile(.95):.2f}")

# ---------------- S10: BQ1 persistence curve ----------------
r_now = np.exp(F.anom0.values)
e2 = np.array([0, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.2, 1.5, 2, 3, np.inf])
c2 = pd.cut(r_now, e2)
g = pd.DataFrame({"c": c2, "y": y}).groupby("c", observed=True).y.agg(["mean", "size"])
lab2 = [f"{iv.left:g}-{iv.right:g}" if np.isfinite(iv.right) else f">{iv.left:g}" for iv in g.index]
fig, ax = plt.subplots(figsize=(12.5, 4.5))
ax.bar(lab2, g["mean"], color=[BLUE if iv.right <= 0.6 else GREY for iv in g.index])
n_dry = sum(iv.right <= 0.6 for iv in g.index)
ax.add_patch(Rectangle((-0.5, 0), n_dry, 0.84, fill=False, ls="--", ec=RED, lw=1.3))
ax.text(n_dry - 0.5, 0.8, "  sudah kering bulan ini", color=RED, fontsize=10.5, va="center")
for i, m_ in enumerate(g["mean"]):
    ax.text(i, m_ + 0.015, f"{m_:.0%}", ha="center", fontsize=10, color=INK)
ax.axhline(y.mean(), color=MUTED, ls=":", lw=1); ax.text(len(g) - 0.5, y.mean() + 0.02, f"rata-rata {y.mean():.0%}", ha="right", color=MUTED, fontsize=10)
ax.set_xlabel("Pasokan BULAN INI / median klimatologi"); ax.set_ylabel("Peluang water stress BULAN DEPAN")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
save(fig, "s10b_bq1_persistence_backup.png")
p_dry, p_wet = y[dry].mean(), y[~dry].mean()
print(f"[S10] P(stress next | dry now) {p_dry:.3f} vs | not dry {p_wet:.3f} (x{p_dry / p_wet:.1f}); share dry now {dry.mean():.3f}; "
      f"lowest bin {g['mean'].iloc[0]:.3f}, highest-ratio bin {g['mean'].iloc[-1]:.3f}; persistence AP {ap(y, -F.anom0.fillna(0)):.3f}")

# ---------------- S11: BQ1 per-year lift and chi-square ----------------
rows = []
for k in range(14):
    m = year == k
    tab = pd.crosstab(dry[m], y[m])
    chi, p, _, _ = chi2_contingency(tab)
    rows.append(dict(yr=f"{2000 + k}/{str(2001 + k)[2:]}", p_dry=y[m & dry].mean(), p_wet=y[m & ~dry].mean(), p=p))
Y1 = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(Y1.yr, Y1.p_dry, marker="o", color=RED, lw=2, label="Bulan ini kering (< 0.6 median)")
ax.plot(Y1.yr, Y1.p_wet, marker="o", color=BLUE, lw=2, label="Bulan ini tidak kering")
for i, r in Y1.iterrows():
    ax.text(i, r.p_dry + 0.03, f"{r.p_dry:.0%}", ha="center", fontsize=8.5, color=INK)
    ax.text(i, r.p_wet - 0.07, f"{r.p_wet:.0%}", ha="center", fontsize=8.5, color=INK)
ax.set_ylim(0, 1); ax.set_ylabel("Peluang water stress bulan depan"); ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax.legend(loc="upper right", fontsize=10)
save(fig, "s11b_bq1_dry_vs_normal_backup.png")
print(f"[S11] years with P(dry)>P(wet): {(Y1.p_dry > Y1.p_wet).sum()}/14; chi-square p max {Y1.p.max():.2e}; "
      f"lift median {(Y1.p_dry / Y1.p_wet).median():.1f}x, min {(Y1.p_dry / Y1.p_wet).min():.1f}x")

# ---------------- S10/S11 main (winner-deck BQ1 layout): per-year Spearman + per-year Mann-Whitney ----------------
# continuous next-month supply ratio (ratio, from the S8 block) vs this month's anomaly, one statistic per hydro year
zn = np.log(np.where(ratio > 0, ratio, np.nan))
a0 = F.anom0.values
good = np.isfinite(zn) & np.isfinite(a0)
yr_prev = pd.Series(y).groupby(year).mean()
wet_years = set(yr_prev[yr_prev < yr_prev.median()].index)
rows = []
for k in range(14):
    m = good & (year == k)
    rho, prho = spearmanr(a0[m], zn[m])
    xn, xd = zn[m & ~dry], zn[m & dry]
    mw = mannwhitneyu(xn, xd, alternative="two-sided")
    r_rb = 2 * mw.statistic / (len(xn) * len(xd)) - 1
    rows.append(dict(k=k, yr=f"{2000 + k}/{str(2001 + k)[2:]}", rho=rho, p_rho=prho, r_rb=r_rb, p_mw=mw.pvalue,
                     prev=yr_prev[k], wet=k in wet_years, n=int(m.sum())))
B1 = pd.DataFrame(rows)
print("[S10] per-year Spearman(this month anomaly, next month supply ratio) and Mann-Whitney dry vs normal:")
print(B1.round(4).to_string(index=False))
for lab, g_ in [("tahun kering (prevalensi >= median)", B1[~B1.wet]), ("tahun basah (prevalensi < median)", B1[B1.wet])]:
    print(f"[S10] {lab}: median rho {g_.rho.median():.3f}, median p {g_.p_rho.median():.2e}, years {len(g_)}; "
          f"median r_rb {g_.r_rb.median():.3f}")
print(f"[S11] years r_rb > 0: {(B1.r_rb > 0).sum()}/14, significant (p<0.05): {(B1.p_mw < 0.05).sum()}/14, "
      f"median r_rb {B1.r_rb.median():.3f}; rho > 0 in {(B1.rho > 0).sum()}/14, median rho {B1.rho.median():.3f}")

def per_year_chart(col, pcol, ylabel, name, box_label):
    fig, ax = plt.subplots(figsize=(11.5, 5))
    ax.plot(B1.yr, B1[col], color=BLUE, lw=2, marker="o", ms=7, label=ylabel.split(" (")[0])
    sig = B1[pcol] < 0.05
    ax.scatter(B1.yr[sig], B1[col][sig], s=90, facecolors="none", edgecolors=RED, linewidths=1.8, zorder=3,
               label="Signifikan (p < 0.05)")
    med = B1[col].median()
    ax.axhline(med, color=INK, ls="--", lw=1.2, label=f"Median = {med:.3f}")
    ax.axhline(0, color=MUTED, lw=0.8)
    for i, v in enumerate(B1[col]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    lo, hi = B1[col].min() - 0.06, B1[col].max() + 0.08
    ax.add_patch(Rectangle((-0.4, lo), 13.8, hi - lo, fill=False, ls="--", ec=RED, lw=1.3))
    ax.text(-0.3, hi + 0.025, box_label, color=RED, fontsize=10.5)
    ax.set_ylim(min(-0.1, lo - 0.05), 1.0)
    ax.set_ylabel(ylabel); ax.set_xlabel("Tahun hidrologi (September - Agustus)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, edgecolor="#E4E7EB")
    save(fig, name)

per_year_chart("rho", "p_rho", "Spearman rho (pasokan bulan ini vs bulan depan)", "s10_bq1_spearman_per_year.png",
               "Selalu berkorelasi positif kuat (rho > 0) di 14 dari 14 tahun")
per_year_chart("r_rb", "p_mw", "Rank-biserial r_rb (normal vs sudah kering)", "s11_bq1_mannwhitney_per_year.png",
               "Sub-DAS yang sudah kering selalu punya pasokan bulan depan lebih rendah")

# ---------------- S12: BQ2 drought duration, diminishing return ----------------
run = F.drought_run.values
g3 = pd.DataFrame({"r": run, "y": y}).groupby("r").y.agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(10.5, 4.4))
ax.plot(g3.index, g3["mean"], marker="o", color=BLUE, lw=2)
for r_, m_ in g3["mean"].items():
    ax.text(r_, m_ + 0.03, f"{m_:.0%}", ha="center", fontsize=9.5, color=INK)
ax.add_patch(Rectangle((5.6, g3["mean"].loc[6] - 0.08), 6.8, 0.2, fill=False, ls="--", ec=RED))
ax.text(6, g3["mean"].loc[6] - 0.2, "kurva melandai: tambahan bulan kering\nhanya menambah sedikit risiko", color=RED, fontsize=10)
ax.set_xticks(range(13)); ax.set_xlabel("Durasi kekeringan beruntun sampai bulan ini (bulan)")
ax.set_ylabel("Peluang water stress bulan depan"); ax.set_ylim(0, 1.05); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
save(fig, "s12b_bq2_probability_backup.png")
inc = g3["mean"].diff()
print("[S12] P(stress next) by drought_run:\n" + g3.assign(increment=inc).round(3).to_string())

# ---------------- S12 main (winner-deck BQ2 layout): median (quantile) regression with linear splines ----------------
# outcome = next month's supply as % of normal (continuous version of the label: label = 1 <=> < 60%), predictor =
# drought duration with a piecewise-linear basis; each coefficient = change of the median per extra month in that segment
okq = np.isfinite(ratio) & (ratio > 0) & np.isfinite(run)
yq = 100 * ratio[okq]
d = run[okq]
SEGS = [("0 -> 1 bulan", 0, 1), ("1 -> 3 bulan", 1, 3), ("3 -> 6 bulan", 3, 6), ("6 -> 12 bulan", 6, 12)]
basis = lambda dd: np.column_stack([np.clip(dd - lo, 0, hi - lo) for _, lo, hi in SEGS])
Xq = sm.add_constant(basis(d))
qr = sm.QuantReg(yq, Xq).fit(q=0.5, max_iter=5000)
coef = pd.DataFrame({"basis": ["Intercept"] + [n for n, _, _ in SEGS], "coef": qr.params, "p_value": qr.pvalues})
print("[S12] median regression of next-month supply (% of normal) on drought duration, linear-spline basis:")
print(coef.round(4).to_string(index=False))
grid = np.arange(13)
fit_med = sm.add_constant(basis(grid), has_constant="add") @ qr.params
print("[S12] fitted median by duration: " + ", ".join(f"{g}:{v:.0f}%" for g, v in zip(grid, fit_med)))
cross = int(grid[np.argmax(fit_med < 60)]) if (fit_med < 60).any() else None
print(f"[S12] fitted median falls below the 60% threshold from duration {cross} months")

rs = np.random.default_rng(0)
pick = rs.choice(len(d), 25000, replace=False)
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.scatter(d[pick] + rs.uniform(-0.28, 0.28, len(pick)), np.clip(yq[pick], 0, 250), s=4, color=LIGHT, alpha=0.35,
           label="Data (sampel 25,000 baris)")
ax.plot(grid, fit_med, color=BLUE, lw=3, marker="o", label="Quantile Regression (tau = 0.5)")
ax.axhline(60, color=RED, ls=":", lw=1.5)
ax.text(12.4, 63, "ambang water stress 60%", color=RED, ha="right", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.add_patch(Rectangle((2.6, fit_med[3:].min() - 12), 9.8, fit_med[3:].max() - fit_med[3:].min() + 24,
                       fill=False, ls="--", ec=RED, lw=1.3))
ax.text(2.7, 78, "Melandai setelah kering > 3 bulan", color=RED, fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.set_xticks(grid)
ax.set_xlim(-0.6, 12.6); ax.set_ylim(0, 250)
ax.set_xlabel("Durasi kekeringan beruntun sampai bulan ini (bulan)")
ax.set_ylabel("Pasokan bulan depan (% dari normal)")
ax.set_title("Quantile Regression with Splines (tau = 0.5)", fontsize=12.5, color=INK)
ax.legend(loc="upper right", fontsize=9.5, frameon=True, edgecolor="#E4E7EB")
save(fig, "s12_bq2_quantile_regression.png")

# ---------------- S13: BQ3 combinations of conditions ----------------
cond = pd.DataFrame({
    "kering": dry,
    "basin kering": F.basin_below0.values >= 0.5,
    "baseflow turun": F.bf_trend1.values < 0,
    "musim menuju basah": F.seas_step.values > 0,
    "tren 3 bln turun": F.ws_trend3.values < 0,
})
key = cond.apply(lambda r: " + ".join([k for k, v_ in r.items() if v_]) or "tidak ada sinyal", axis=1)
G = pd.DataFrame({"k": key.values, "y": y}).groupby("k").y.agg(["mean", "size"])
G["support"] = G["size"] / len(y)
G = G[G.support >= 0.02].sort_values("mean")
top, bot = G.tail(5).iloc[::-1], G.head(5)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
for a, D, c in [(ax[0], top, BLUE), (ax[1], bot, YELLOW)]:
    yy = np.arange(len(D))[::-1]
    a.barh(yy, D["mean"], color=c)
    for i, (k_, r) in zip(yy, D.iterrows()):
        a.text(r["mean"] + 0.01, i, f"{r['mean']:.0%}  (n {r.support:.0%})", va="center", fontsize=9, color=INK)
    a.set_yticks(yy, [k_.replace(" + ", "\n+ ") if k_.count("+") > 1 else k_ for k_ in D.index], fontsize=8.5)
    a.set_xlabel("Peluang water stress bulan depan"); a.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a.grid(axis="y", visible=False); a.set_xlim(0, 1.05)
save(fig, "s13_bq3_combinations.png")
print("[S13] top combinations:\n" + top.round(3).to_string() + "\n[S13] bottom combinations:\n" + bot.round(3).to_string())

# ---------------- S14-S16: BQ4 season ----------------
MN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
pm = pd.Series(y).groupby(tmonth).mean()
rec_m = pd.Series(1 - y[dry]).groupby(tmonth[dry]).mean()
ons_m = pd.Series(y[~dry]).groupby(tmonth[~dry]).mean()
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(MN, pm.values, color=BLUE)
for i, v_ in enumerate(pm.values):
    ax.text(i, v_ + 0.005, f"{v_:.0%}", ha="center", fontsize=10, color=INK)
ax.set_ylim(0, 0.3); ax.set_ylabel("Porsi water stress"); ax.set_xlabel("Bulan target")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
save(fig, "s14_bq4_prevalence_by_month.png")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
hi_rec = rec_m.values >= rec_m.quantile(.75)
ax[0].bar(MN, rec_m.values, color=[BLUE if h else GREY for h in hi_rec])
for i, v_ in enumerate(rec_m.values):
    ax[0].text(i, v_ + 0.01, f"{v_:.0%}", ha="center", fontsize=9.5, color=INK)
ax[0].set_ylabel("Peluang PULIH bulan depan\n(sub-DAS yang sedang kering)"); ax[0].set_ylim(0, 0.7)
ax[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax[0].set_xlabel("Bulan target")
hi_ons = ons_m.values >= ons_m.quantile(.75)
ax[1].bar(MN, ons_m.values, color=[RED if h else GREY for h in hi_ons])
for i, v_ in enumerate(ons_m.values):
    ax[1].text(i, v_ + 0.004, f"{v_:.0%}", ha="center", fontsize=9.5, color=INK)
ax[1].set_ylabel("Peluang MULAI kering bulan depan\n(sub-DAS yang sedang normal)"); ax[1].set_ylim(0, 0.2)
ax[1].set_yticks([0, 0.04, 0.08, 0.12, 0.16, 0.2])
ax[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax[1].set_xlabel("Bulan target")
save(fig, "s15_bq4_recovery_onset.png")
chi_r = chi2_contingency(pd.crosstab(tmonth[dry], y[dry]))
chi_o = chi2_contingency(pd.crosstab(tmonth[~dry], y[~dry]))
chi_p = chi2_contingency(pd.crosstab(tmonth, y))
print(f"[S14] prevalence by target month {pm.min():.3f}..{pm.max():.3f}; chi2 {chi_p[0]:.0f} p {chi_p[1]:.1e}")
print(f"[S15] recovery by month {rec_m.min():.3f} ({MN[rec_m.values.argmin()]}) .. {rec_m.max():.3f} ({MN[rec_m.values.argmax()]}); "
      f"chi2 {chi_r[0]:.0f} p {chi_r[1]:.1e} | onset {ons_m.min():.3f} ({MN[ons_m.values.argmin()]}) .. {ons_m.max():.3f} "
      f"({MN[ons_m.values.argmax()]}); chi2 {chi_o[0]:.0f} p {chi_o[1]:.1e}")
# S16: heatmap recovery by target month x drought duration bucket
bk = pd.cut(run[dry], [0, 1, 3, 6, 12], labels=["1 bln", "2-3 bln", "4-6 bln", "7-12 bln"])
H = pd.DataFrame({"m": tmonth[dry], "b": bk, "r": 1 - y[dry]}).pivot_table(index="b", columns="m", values="r", observed=True)
fig, ax = plt.subplots(figsize=(12, 3.6))
from matplotlib.colors import LinearSegmentedColormap
im = ax.imshow(H.to_numpy(), cmap=LinearSegmentedColormap.from_list("team_blue", ["#FFFFFF", BLUE]), vmin=0, vmax=0.7, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v_ = H.iat[i, j]
        ax.text(j, i, f"{v_:.0%}", ha="center", va="center", fontsize=9.5, color="white" if v_ > 0.45 else INK)
ax.set_xticks(range(12), MN); ax.set_yticks(range(H.shape[0]), H.index); ax.grid(False)
ax.set_xlabel("Bulan target"); ax.set_ylabel("Sudah kering selama")
plt.colorbar(im, ax=ax, label="Peluang pulih", format=lambda v, _: f"{v:.0%}")
save(fig, "s16_bq4_recovery_heatmap.png")
print("[S16] recovery by duration bucket (mean over months): " + H.mean(1).round(3).to_dict().__repr__())

# ---------------- S17: BQ5 water use vs supply ----------------
use = F.use_clim_next.values  # withdrawals / NORMAL supply of the target month (not / current supply: that mixes in the deficit)
okd = np.isfinite(use) & np.isfinite(F.anom0.values)
dq_use = pd.qcut(pd.Series(use[okd]).rank(method="first"), 10, labels=False) + 1
dq_sup = pd.qcut(pd.Series(-F.anom0.values[okd]).rank(method="first"), 10, labels=False) + 1
r_use = pd.Series(y[okd]).groupby(dq_use.values).mean()
r_sup = pd.Series(y[okd]).groupby(dq_sup.values).mean()
fig, ax = plt.subplots(figsize=(10.5, 4.4))
ax.plot(r_sup.index, r_sup.values, marker="o", color=BLUE, lw=2, label="Desil defisit pasokan (anomali bulan ini)")
ax.plot(r_use.index, r_use.values, marker="o", color=YELLOW, lw=2.5, mec=INK, mew=0.8, label="Desil intensitas pemakaian air (pemakaian / pasokan normal)")
ax.text(10.15, r_sup.values[-1], f"{r_sup.values[-1]:.0%}", va="center", fontsize=10, color=INK)
ax.text(10.15, r_use.values[-1], f"{r_use.values[-1]:.0%}", va="center", fontsize=10, color=INK)
ax.set_xticks(range(1, 11)); ax.set_xlabel("Desil (1 = paling rendah, 10 = paling tinggi)")
ax.set_ylabel("Peluang water stress bulan depan"); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax.legend(fontsize=10)
save(fig, "s17b_bq5_deciles_backup.png")
a_use = auc(y[okd], use[okd]); a_sup = auc(y[okd], -F.anom0.values[okd])
rho_use = spearmanr(use[okd], y[okd])[0]
print(f"[S17] AUC water-use intensity {a_use:.3f} vs supply anomaly {a_sup:.3f}; spearman(use, label) {rho_use:.3f}; "
      f"decile range use {r_use.min():.3f}-{r_use.max():.3f}, supply {r_sup.min():.3f}-{r_sup.max():.3f}")

# ---------------- S17 main (winner-deck BQ5 layout): scatter per sub-DAS + concrete-evidence table ----------------
rep6 = REPAIRS["v6"](tr)
use_m = rep6[["q_ir_wd_inc_m3_lag0", "q_th_wd_inc_m3_lag0", "q_ps_wd_inc_m3_lag0"]].abs().sum(1).values
Hs = pd.DataFrame({"id": tr.id.values, "ws": rep6.q_ws_cum_m3_lag0.abs().values, "use": use_m,
                   "ir": rep6.q_ir_wd_inc_m3_lag0.abs().values, "y": y}).groupby("id").agg(
    ws=("ws", "median"), use=("use", "median"), ir=("ir", "median"), rate=("y", "mean"))
n_all = len(Hs)
Hs = Hs[Hs.ws >= 1000]                                  # drop HUCs with ~zero supply (ratio artefact, e.g. 6,891x)
Hs["ratio"] = Hs.use / Hs.ws
n_zero = (Hs.ratio < 1e-4).mean()
P = Hs[Hs.ratio >= 1e-4].copy()
thr10 = Hs.ratio.quantile(0.9)
P["top"] = P.ratio >= thr10
med_rate = Hs.rate.median()
fig, ax = plt.subplots(figsize=(10.5, 6))
ax.scatter(P.ratio[~P.top], P.rate[~P.top], s=16, color=BLUE, alpha=0.45, lw=0, label="Sub-DAS lain")
ax.scatter(P.ratio[P.top], P.rate[P.top], s=26, color=RED, alpha=0.8, lw=0, label="10% sub-DAS paling intensif")
ax.set_xscale("log")
ax.axhline(med_rate, color=MUTED, ls="--", lw=1)
ax.axvline(thr10, color=MUTED, ls="--", lw=1)
ax.text(P.ratio.min(), med_rate + 0.008, f"median frekuensi {med_rate:.0%}", color=MUTED, fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.text(thr10 * 1.15, 0.47, f"batas top 10%\n({thr10:.2f}x pasokan)", color=MUTED, fontsize=10, va="top")
ax.add_patch(Rectangle((thr10, -0.01), P.ratio.max() * 3 - thr10, P.rate.max() + 0.03, fill=False, ls="--", ec=RED, lw=1.3))
ex = P.sort_values("ratio", ascending=False).head(3)
for hid, r in ex.iterrows():
    ax.annotate(f"{hid[:6]}: {r.ratio:,.0f}x, {r.rate:.0%}", (r.ratio, r.rate), xytext=(-150, 12), textcoords="offset points",
                fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlabel("Intensitas pemakaian air = pengambilan air / pasokan (median, skala log)")
ax.set_ylabel("Frekuensi water stress 2000-2014")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}x")
ax.set_ylim(-0.02, 0.5); ax.legend(loc="upper left", fontsize=10)
save(fig, "s17_bq5_scatter_per_huc.png")
top_r, rest_r = Hs.rate[Hs.ratio >= thr10], Hs.rate[Hs.ratio < thr10]
mw = mannwhitneyu(top_r, rest_r)
rho_h = spearmanr(Hs.ratio, Hs.rate)[0]
d_own = pd.DataFrame({"id": tr.id.values, "u": use_m, "y": y})
hi_own = d_own.u > d_own.groupby("id").u.transform("median")
chi_own = chi2_contingency(pd.crosstab(hi_own, d_own.y))
print(f"[S17] HUCs {n_all}, kept (supply >= 1000 m3) {len(Hs)}; use ~0 (<0.01%) {n_zero:.3f}; top-10% threshold {thr10:.3f}x; "
      f"spearman(ratio, rate) {rho_h:.3f}; rate top10 mean {top_r.mean():.3f} median {top_r.median():.3f} vs rest {rest_r.mean():.3f} / "
      f"{rest_r.median():.3f}; Mann-Whitney U {mw.statistic:,.0f} p {mw.pvalue:.2e}")
print(f"[S17] within-HUC: stress rate when use above own median {d_own.y[hi_own].mean():.4f} vs below {d_own.y[~hi_own].mean():.4f}; "
      f"chi2 {chi_own[0]:.1f} p {chi_own[1]:.2e}; irrigation share of use among top10 {(P.ir[P.top] / P.use[P.top]).median():.2f}")

# concrete-evidence table, styled like a notebook df.head() screenshot
T5 = Hs.sort_values("ratio", ascending=False).head(5).reset_index()
T5 = pd.DataFrame({"id": T5.id, "sumber": np.where(T5.ir / T5.use > 0.5, "irigasi", "lainnya"),
                   "pemakaian/pasokan": T5.ratio.map(lambda v: f"{v:,.0f}x"), "frek_water_stress": T5.rate.map(lambda v: f"{v:.1%}")})
fig, ax = plt.subplots(figsize=(8, 2.3)); ax.axis("off")
cells = [[str(i)] + list(r) for i, r in T5.iterrows()]
tb = ax.table(cellText=cells, colLabels=[""] + list(T5.columns), loc="center", cellLoc="right", colLoc="right",
              colWidths=[0.06, 0.26, 0.16, 0.24, 0.28])
tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1, 1.55)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("white")
    cell.set_facecolor("#F5F5F5" if r % 2 == 1 else "white")
    if r == 0 or c == 0:
        cell.set_text_props(weight="bold", color=INK)
    if r > 0 and c == 4 and float(cells[r - 1][4].rstrip("%")) / 100 < med_rate:
        cell.set_text_props(color=RED, weight="bold")
for c in range(5):
    tb[0, c].visible_edges = "B"; tb[0, c].set_edgecolor(INK)
save(fig, "s17_bq5_table_top_users.png")
print("[S17] top-5 most intensive HUCs:\n" + T5.to_string(index=False))

# ---------------- S18: BQ6 importance (one chart, like the winner deck's grouped CatBoost importance) -------------
# bars = 10 meaningful feature groups, colour/legend = 3 categories, importance = mean |SHAP| on the HELD-OUT block of a
# 5-fold contiguous-time CV (purge 12 months), summed across folds (the winner deck sums CatBoost importance across folds)
GROUPS = {
    "Anomali pasokan bulan ini": ("hidro", ["anom0", "margin_anom_persist", "margin_raw_persist"]),
    "Riwayat anomali 12 bulan": ("hidro", [f"anom{k}" for k in range(1, 12)] + ["anom_mean3", "anom_mean6", "anom_mean12",
                                                                                "anom_min6", "anom_std12", "margin_lastyear"]),
    "Durasi kekeringan": ("hidro", ["n_below12", "drought_run"]),
    "Skenario tanpa hujan & komposisi aliran": ("hidro", ["margin_dry", "bf_share0", "qf_share0", "up_share0"]),
    "Tren & resesi aliran": ("hidro", ["ws_trend1", "ws_trend2", "ws_trend3", "bf_trend1", "bf_recession", "qf_mean3_share"]),
    "Kondisi basin & hulu-hilir": ("konteks", ["down_anom0", "up_anom0", "basin_anom0", "basin_below0", "anom0_vs_basin"]),
    "Musim (bulan)": ("konteks", ["month", "m_sin", "m_cos"]),
    "Pola klimatologi": ("konteks", ["seas_step", "clim_cv"]),
    "Karakteristik statis (luas, populasi)": ("konteks", ["log_inc_area", "log_cum_area", "headwater", "log_pop_dens", "pop_growth"]),
    "Pemakaian air": ("pakai", ["use_ratio0", "use_clim_next", "ir_share"]),
}
CAT = {"hidro": ("Kondisi hidrologi sub-DAS", BLUE), "konteks": ("Konteks wilayah & waktu", YELLOW), "pakai": ("Pemakaian air", RED)}
feat = sum([c for _, c in GROUPS.values()], [])
p13 = dict(objective="binary", learning_rate=0.05, num_leaves=53, min_data_in_leaf=91, feature_fraction=0.33, bagging_fraction=0.82,
           bagging_freq=1, lambda_l2=9.27, extra_trees=True, verbose=-1, seed=42, num_threads=20)
edges_cv = np.linspace(0, t.max() + 1, 6).astype(int)
imp = pd.Series(0.0, index=feat)
fold_ap = []
for f_i in range(5):
    lo, hi = edges_cv[f_i], edges_cv[f_i + 1]
    va = (t >= lo) & (t < hi)
    fit = (t < lo - 12) | (t >= hi + 12)
    mdl = lgb.train(p13, lgb.Dataset(F.loc[fit, feat].to_numpy("float32"), y[fit]), 400)
    fold_ap.append(ap(y[va], mdl.predict(F.loc[va, feat].to_numpy("float32"))))
    Sv = F.loc[va, feat].sample(8000, random_state=f_i).to_numpy("float32")
    imp += np.abs(mdl.predict(Sv, pred_contrib=True)[:, :-1]).mean(0)
gi = pd.Series({g_: imp[c].sum() for g_, (_, c) in GROUPS.items()})
gi = (gi / gi.sum()).sort_values()
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.barh(gi.index, gi.values, height=0.6, color=[CAT[GROUPS[k_][0]][1] for k_ in gi.index])
for i, v_ in enumerate(gi.values):
    ax.text(v_ + 0.004, i, f"{100 * v_:.1f}", va="center", fontsize=10.5, color=INK)
ax.set_title("Feature importance (LightGBM, SHAP, dikelompokkan)", loc="left", fontsize=13, color=INK, pad=12)
ax.set_xlabel("Kontribusi (% rata-rata |SHAP| pada data validasi, 5 fold)")
ax.grid(axis="y", visible=False); ax.set_xlim(0, gi.max() * 1.15)
ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}")
ax.legend([Rectangle((0, 0), 1, 1, color=c) for _, c in CAT.values()], [n for n, _ in CAT.values()],
          loc="lower right", fontsize=10.5, frameon=True, edgecolor="#E4E7EB")
save(fig, "s18_bq6_feature_importance.png")
old = OUT / "s18_bq6_importance_and_alerts.png"
if old.exists():
    old.unlink()
print(f"[S18] 5-fold time-CV AP of the explained LightGBM: {np.mean(fold_ap):.4f} (" + ", ".join(f"{a:.3f}" for a in fold_ap) + ")")
print("[S18] category shares: " + ", ".join(f"{CAT[c][0]} {gi[[k for k, (cc, _) in GROUPS.items() if cc == c]].sum():.3f}" for c in CAT))

# early-warning capacity (numbers for the slide text, no chart): alert budget on the LOYO + basin-holdout OOF of 63
O = pd.read_parquet(WORK / "63_oof_s42.parquet")
mem = ["lgb V8", "lgb V13 tuned", "lgb dart", "xgboost", "catboost"]
ens = np.full(len(tr), np.nan)
for k in range(14):
    m = year == k
    ens[m] = O.loc[m, mem].rank(pct=True).mean(1).values
rows = []
okm = ~np.isnan(ens)
for q in (0.05, 0.10, 0.20, 0.30):
    flag, fp = np.zeros(len(tr), bool), np.zeros(len(tr), bool)
    for tt in np.unique(t[okm]):
        m = np.flatnonzero((t == tt) & okm)
        n = int(round(q * len(m)))
        flag[m[np.argsort(-ens[m])[:n]]] = True
        fp[m[np.argsort(F.anom0.fillna(0).values[m])[:n]]] = True
    rows.append(dict(q=q, recall=y[flag].sum() / y[okm].sum(), precision=y[flag].mean(),
                     recall_persist=y[fp].sum() / y[okm].sum(), precision_persist=y[fp].mean()))
A_ = pd.DataFrame(rows)
print("[S18] grouped importance:\n" + gi.iloc[::-1].round(3).to_string())
print("[S18] alert budget (validation LOYO + basin holdout):\n" + A_.round(3).to_string(index=False))
yrs_ap = np.array([ap(y[year == k], ens[year == k]) for k in range(14)])
print(f"[S18] validation AP (LOYO + basin holdout, all5 blend) {yrs_ap.mean():.4f}; persistence "
      f"{np.mean([ap(y[year == k], -F.anom0.fillna(0).values[year == k]) for k in range(14)]):.4f}")
