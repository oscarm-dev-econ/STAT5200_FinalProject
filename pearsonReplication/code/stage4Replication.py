# Pearson (2024) replication - ACS random forests and Table 4.
# Trains 100-tree classifiers separately for 2005-2017 using IPUMS ACS, predicts noncitizen childhood arrival probabilities in NCVS, and replicates Table 4.
# Outputs: prediction/sample benchmarks and Table 4 coefficients, clustered SEs, means, and Ns in the terminal.
# Saves: data_processed/pearson_ncvsperson_rf_predictions.pkl and data_processed/pearson_table4_with_predictions.pkl. No CSV or Excel files.

from pathlib import Path
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier

root = Path(os.environ.get("PEARSON_ROOT", r"C:\Users\Oscar\Dropbox\STAT5200_FinalProject\pearsonReplicationV2"))
acs_file = root / "data_raw" / "IPUMS_ACS" / "usa_00007.dta"
ncvs_dir = root / "data_raw" / "NCVS_files"
processed = root / "data_processed"
base_file = processed / "pearson_table2_baseline.pkl"
pred_file = processed / "pearson_ncvsperson_rf_predictions.pkl"
table4_file = processed / "pearson_table4_with_predictions.pkl"
retrain_forests = False
processed.mkdir(parents=True, exist_ok=True)

features = ["age", "female", "married", "divorced", "separated", "widowed", "somecollege", "profschool", "associate", "bachelor", "masters", "doctorate", "black", "pacisland", "asia", "whiteblack", "whiteai", "whiteasian", "whitehawaiian", "races2_3", "races4_6", "midwest", "south", "west", "notmsa", "centralmsa", "notcentralmsa", "own", "rent"] + [f"income{i}" for i in range(1, 15)] + ["nevermarried", "hs", "white", "northeast", "uknownownship", "incomeunk"]

if pred_file.exists() and not retrain_forests:
    predictions = pd.read_pickle(pred_file)
else:
    acs_cols = ["year", "age", "sex", "marst", "educ", "educd", "hispan", "raced", "region", "metro", "ownershp", "hhincome", "citizen", "birthyr", "yrimmig"]
    acs_parts = []
    rng = np.random.default_rng(1234)

    for chunk in pd.read_stata(acs_file, columns=acs_cols, chunksize=150_000, convert_categoricals=False):
        chunk.columns = chunk.columns.str.lower()
        a = chunk.loc[(chunk["hispan"] > 0) & (chunk["educd"] >= 62) & (chunk["educ"] != 999) & chunk["year"].between(2005, 2017)].copy()
        if len(a) == 0:
            continue

        f = pd.DataFrame(index=a.index)
        r, e, m, t, inc = a["raced"], a["educd"], a["marst"], a["metro"], a["hhincome"]
        f["year"], f["age"] = a["year"].astype(int), a["age"]
        f["DACA_eligible"] = ((a["citizen"] == 3) & a["yrimmig"].between(1, 2007) & (a["yrimmig"] <= a["birthyr"] + 16)).astype(int)
        f["female"], f["married"] = (a["sex"] == 2).astype(int), m.isin([1, 2]).astype(int)
        for name, code in {"divorced": 4, "separated": 3, "widowed": 5, "nevermarried": 6}.items():
            f[name] = (m == code).astype(int)
        for name, code in {"profschool": 115, "associate": 81, "bachelor": 101, "masters": 114, "doctorate": 116}.items():
            f[name] = (e == code).astype(int)
        f["somecollege"], f["hs"] = e.between(65, 71).astype(int), e.isin([62, 63, 64]).astype(int)
        for name, code in {"white": 100, "black": 200, "whiteblack": 801, "whiteai": 802, "whitehawaiian": 821}.items():
            f[name] = (r == code).astype(int)
        f["pacisland"], f["asia"], f["whiteasian"] = r.between(680, 699).astype(int), r.between(400, 679).astype(int), r.between(810, 818).astype(int)
        f["races2_3"], f["races4_6"] = (r.between(822, 949) | (r == 300)).astype(int), r.between(950, 990).astype(int)
        f["northeast"], f["midwest"] = a["region"].isin([11, 12]).astype(int), a["region"].isin([21, 22]).astype(int)
        f["south"], f["west"] = a["region"].isin([31, 32, 33]).astype(int), a["region"].isin([41, 42]).astype(int)
        f["notmsa"], f["centralmsa"], f["notcentralmsa"] = (t == 1).astype(int), t.isin([2, 4]).astype(int), (t == 3).astype(int)
        f["own"], f["rent"] = (a["ownershp"] == 1).astype(int), (a["ownershp"] == 2).astype(int)
        f["uknownownship"], f["incomeunk"] = (a["ownershp"] == 0).astype(int), (inc == 9999999).astype(int)
        income_limits = [(-np.inf, 4999), (5000, 7499), (7500, 9999), (10000, 12499), (12500, 14999), (15000, 17499), (17500, 19999), (20000, 24999), (25000, 29999), (30000, 34999), (35000, 39999), (40000, 49999), (50000, 74999), (75000, np.inf)]
        for i, (lo, hi) in enumerate(income_limits, 1):
            f[f"income{i}"] = (inc.between(lo, hi) & (inc != 9999999)).astype(int)
        f["train"] = rng.random(len(f)) < 0.9
        acs_parts.append(f[["year", "DACA_eligible", "train"] + features].reset_index(drop=True))

    acs = pd.concat(acs_parts, ignore_index=True)
    if len(acs) != 2_424_297:
        raise ValueError(f"ACS sample has {len(acs):,} observations, not Pearson's 2,424,297.")
    del acs_parts

    hh_cols = ["IDHH", "YEARQ", "V2014", "V2026", "V2129", "V2127B"]
    person_cols = ["IDHH", "IDPER", "YEARQ", "V3014", "V3015", "V3017", "V3020", "V3023A", "V3024"]
    ncvs_parts = []

    for year in range(2005, 2018):
        folder = ncvs_dir / str(year)
        hh = pd.read_stata(folder / f"{year}_hhd.dta", columns=hh_cols, convert_categoricals=False).drop_duplicates(["IDHH", "YEARQ"])
        p = pd.read_stata(folder / f"{year}_person.dta", columns=person_cols, convert_categoricals=False).drop_duplicates(["IDPER", "YEARQ"])
        p = p.merge(hh, on=["IDHH", "YEARQ"], how="left", validate="m:1")
        p = p.loc[(p["V3024"] == 1) & (p["V3020"] > 12) & ~p["V3020"].isin([27, 98, 99])].copy()
        if len(p) == 0:
            continue

        f = p[["IDPER", "YEARQ"]].copy()
        r, e, m, t, inc = p["V3023A"], p["V3020"], p["V3015"], p["V2129"], p["V2026"]
        f["year"], f["age"] = year, p["V3014"]
        f["female"], f["married"] = (p["V3017"] == 2).astype(int), (m == 1).astype(int)
        for name, code in {"divorced": 3, "separated": 4, "widowed": 2, "nevermarried": 5}.items():
            f[name] = (m == code).astype(int)
        for name, code in {"profschool": 44, "associate": 41, "bachelor": 42, "masters": 43, "doctorate": 45}.items():
            f[name] = (e == code).astype(int)
        f["somecollege"], f["hs"] = (e == 40).astype(int), (e == 28).astype(int)
        for name, code in {"white": 1, "black": 2, "whiteblack": 6, "whiteai": 7, "whiteasian": 8, "whitehawaiian": 9}.items():
            f[name] = (r == code).astype(int)
        f["asia"], f["pacisland"] = (r == 4).astype(int), (r == 5).astype(int)
        f["races2_3"], f["races4_6"] = (r.between(10, 19) | (r == 3)).astype(int), (r == 20).astype(int)
        f["northeast"], f["midwest"] = (p["V2127B"] == 1).astype(int), (p["V2127B"] == 2).astype(int)
        f["south"], f["west"] = (p["V2127B"] == 3).astype(int), (p["V2127B"] == 4).astype(int)
        f["notmsa"], f["centralmsa"], f["notcentralmsa"] = (t == 3).astype(int), (t == 1).astype(int), (t == 2).astype(int)
        f["own"], f["rent"] = (p["V2014"] == 1).astype(int), p["V2014"].isin([2, 3]).astype(int)
        f["uknownownship"], f["incomeunk"] = (p["V2014"].isna() | (p["V2014"] >= 8)).astype(int), (inc.isna() | (inc >= 98)).astype(int)
        for i in range(1, 15):
            f[f"income{i}"] = (inc == i).astype(int)
        f["income14"] = ((inc == 14) | (inc.between(15, 18) & (p["YEARQ"] > 2017) & (p["YEARQ"] < 2022))).astype(int)
        ncvs_parts.append(f[["IDPER", "YEARQ", "year"] + features].reset_index(drop=True))

    ncvs = pd.concat(ncvs_parts, ignore_index=True).drop_duplicates(["IDPER", "YEARQ"], keep="first")
    del ncvs_parts
    print(f"NCVS Hispanic HS+ person-surveys: {len(ncvs):,} (Pearson: 186,810)")
    print(f"ACS Hispanic HS+ persons: {len(acs):,} (Pearson: 2,424,297)")

    print("Training 13 annual random forests...")
    predicted_parts = []
    for year in range(2005, 2018):
        a = acs.loc[acs["year"] == year]
        n = ncvs.loc[ncvs["year"] == year, ["IDPER", "YEARQ"] + features].copy()
        train = a.loc[a["train"]].dropna(subset=["DACA_eligible"] + features)
        n["probDE_RF100"] = np.nan
        valid = n[features].notna().all(axis=1)
        if len(train) and valid.any():
            rf = RandomForestClassifier(n_estimators=100, criterion="entropy", max_features="sqrt", bootstrap=True, min_samples_leaf=1, random_state=1234, n_jobs=4)
            rf.fit(train[features].to_numpy(dtype=np.float32), train["DACA_eligible"].to_numpy(dtype=np.int8))
            if 1 in rf.classes_:
                n.loc[valid, "probDE_RF100"] = rf.predict_proba(n.loc[valid, features].to_numpy(dtype=np.float32))[:, list(rf.classes_).index(1)]
            else:
                n.loc[valid, "probDE_RF100"] = 0.0
        predicted_parts.append(n[["IDPER", "YEARQ", "probDE_RF100"]])

    predictions = pd.concat(predicted_parts, ignore_index=True).drop_duplicates(["IDPER", "YEARQ"])
    predictions.to_pickle(pred_file)

baseline = pd.read_pickle(base_file)
if len(baseline) != 5364:
    raise ValueError(f"Stage 1 baseline has {len(baseline):,} observations; expected 5,364.")

valid_reference = predictions.loc[(predictions["YEARQ"] > 2005) & (predictions["YEARQ"] < 2018), "probDE_RF100"].dropna()
if valid_reference.empty:
    raise ValueError("No NCVS predictions in YEARQ 2005-2017. Check YEARQ codes and annual NCVS files.")
p50, p80 = np.quantile(valid_reference, [0.50, 0.80], method="averaged_inverted_cdf")
base = baseline.merge(predictions, on=["IDPER", "YEARQ"], how="left", validate="m:1")
base["likely_DE"] = (base["probDE_RF100"] > 0).astype(int)
base["top50"] = (base["probDE_RF100"] > p50).astype(int)
base["top20"] = (base["probDE_RF100"] > p80).astype(int)
base.to_pickle(table4_file)

print(f"Table 4 baseline: {len(base):,} incidents; {base['probDE_RF100'].notna().sum():,} with predictions")
print(f"\nNCVS prediction percentiles: median={p50:.4f}, 80th={p80:.4f} (Pearson: 0.010, 0.086)")
print("\nTABLE 4: EFFECTS OF DACA ON REPORTING")
print(f"{'Sample':<19}{'Beta':>10}{'SE':>10}{'Mean':>9}{'N':>9}{'Pub. beta':>12}{'Pub. N':>9}")

specs = [("All", base.index == base.index, .083, 5364), ("Nonzero probability", base["likely_DE"] == 1, .136, 3696), ("Below median", base["top50"] == 0, -.068, 2377), ("Above median", base["top50"] == 1, .134, 2986), ("Top quintile", base["top20"] == 1, .145, 1211)]
controls = ["agesq", "female", "married", "midwest", "south", "west", "notmsa", "centralmsa", "notcentralmsa"]

for label, mask, published_b, published_n in specs:
    sample = base.loc[mask].dropna(subset=["report", "interaction", "year_incident", "age_DACA", "region"] + controls).copy()
    while len(sample):
        keep = (sample.groupby("age_DACA")["age_DACA"].transform("size") > 1) & (sample.groupby("year_incident")["year_incident"].transform("size") > 1)
        if keep.all():
            break
        sample = sample.loc[keep].copy()
    if len(sample) < 2:
        raise ValueError(f"Not enough observations for {label}.")
    sample["cluster"] = sample["region"].astype(int).astype(str) + "_" + sample["year_incident"].astype(int).astype(str)
    formula = "report ~ interaction + " + " + ".join(controls + ["C(year_incident)", "C(age_DACA)"])
    fit = smf.ols(formula, data=sample).fit(cov_type="cluster", cov_kwds={"groups": sample["cluster"], "use_correction": True, "df_correction": True}, use_t=True)
    print(f"{label:<19}{fit.params['interaction']:>10.4f}{fit.bse['interaction']:>10.4f}{sample['report'].mean():>9.3f}{len(sample):>9,}{published_b:>12.3f}{published_n:>9,}")
