# Pearson (2024) replication of Table 3, Table 1, Figure 1, and Figure 2.
# Outputs: Table 3 and Table 1 estimates printed in the terminal; two PNGs in outputs/figures/.
# Also creates data_processed/pearson_stage2_extended_reporting.pkl if that cache does not exist.

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\Oscar\Dropbox\STAT5200_FinalProject\pearsonReplicationV2")
NCVS = ROOT / "data_raw" / "NCVS_files"
PROCESSED = ROOT / "data_processed"
FIGURES = ROOT / "outputs" / "figures"
BASELINE = PROCESSED / "pearson_table2_baseline.pkl"
EXTENDED = PROCESSED / "pearson_stage2_extended_reporting.pkl"
DEMO = ["agesq", "female", "married"]
GEO = ["midwest", "south", "west", "notmsa", "centralmsa", "notcentralmsa"]

base = pd.read_pickle(BASELINE)
table3 = []

for name, variable, value in [("Women", "female", 1), ("Men", "female", 0), ("HS only", "post_2ndary_deg", 0), ("Postsecondary", "post_2ndary_deg", 1), ("Doesn't own home", "own", 0), ("Owns home", "own", 1)]:
    data = base.loc[base[variable] == value].dropna(subset=["report", "interaction", "year_incident", "age_DACA", "region"] + DEMO + GEO).copy()
    while True:
        keep = (data.groupby("age_DACA")["age_DACA"].transform("size") > 1) & (data.groupby("year_incident")["year_incident"].transform("size") > 1)
        if keep.all():
            break
        data = data.loc[keep].copy()
    clusters = data["region"].astype(int).astype(str) + "_" + data["year_incident"].astype(int).astype(str)
    formula = "report ~ interaction + " + " + ".join(DEMO + GEO + ["C(age_DACA)", "C(year_incident)"])
    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": True, "df_correction": True}, use_t=True)
    table3.append((name, model.params["interaction"], model.bse["interaction"], data["report"].mean(), int(model.nobs)))

print("\nTABLE 3: HETEROGENEOUS EFFECTS")
print(f"{'Group':<20}{'Beta':>10}{'SE':>10}{'Mean':>10}{'N':>9}")
for name, beta, se, mean, n in table3:
    print(f"{name:<20}{beta:>10.4f}{se:>10.4f}{mean:>10.3f}{n:>9,}")

if EXTENDED.exists():
    extended = pd.read_pickle(EXTENDED)
else:
    household_columns = ["IDHH", "YEARQ", "V2127B", "V2129"]
    person_columns = ["IDHH", "IDPER", "YEARQ", "V3014", "V3015", "V3017", "V3020", "V3023A", "V3024", "V3025", "V3026", "V3027"]
    incident_columns = ["IDHH", "IDPER", "YEARQ", "V4015", "V4399"]
    yearly = []

    for year in range(2005, 2018):
        files = {}
        for kind in ["hhd", "person", "incident"]:
            nested = NCVS / str(year) / f"{year}_{kind}.dta"
            files[kind] = nested if nested.exists() else NCVS / f"{year}_{kind}.dta"

        household = pd.read_stata(files["hhd"], columns=household_columns, convert_categoricals=False).drop_duplicates(["IDHH", "YEARQ"])
        person = pd.read_stata(files["person"], columns=person_columns, convert_categoricals=False).drop_duplicates(["IDPER", "YEARQ"])
        incidents = pd.read_stata(files["incident"], columns=incident_columns, convert_categoricals=False)
        person = person.merge(household, on=["IDHH", "YEARQ"], how="left", validate="m:1")
        incidents = incidents.merge(person.drop(columns="IDHH"), on=["IDPER", "YEARQ"], how="left", validate="m:1")

        age, edu = pd.to_numeric(incidents["V3014"], errors="coerce"), pd.to_numeric(incidents["V3020"], errors="coerce")
        hisp, race = pd.to_numeric(incidents["V3024"], errors="coerce"), pd.to_numeric(incidents["V3023A"], errors="coerce")
        incident_year = pd.to_numeric(incidents["V4015"], errors="coerce")
        keep = age.between(22, 50) & incident_year.between(2005, 2016) & (edu > 12) & ~edu.isin([27, 98, 99]) & ((hisp == 1) | ((hisp == 2) & (race == 1)))
        yearly.append(incidents.loc[keep].copy())

    extended = pd.concat(yearly, ignore_index=True)
    extended["age"] = pd.to_numeric(extended["V3014"], errors="coerce")
    extended["agesq"] = extended["age"] ** 2
    extended["year_incident"] = pd.to_numeric(extended["V4015"], errors="coerce")
    extended["region"] = pd.to_numeric(extended["V2127B"], errors="coerce")
    extended["hispanic"] = (extended["V3024"] == 1).astype(int)
    extended["white_nonhispanic"] = ((extended["V3024"] == 2) & (extended["V3023A"] == 1)).astype(int)
    extended["female"] = (extended["V3017"] == 2).astype(int)
    extended["married"] = (extended["V3015"] == 1).astype(int)
    extended["midwest"] = (extended["region"] == 2).astype(int)
    extended["south"] = (extended["region"] == 3).astype(int)
    extended["west"] = (extended["region"] == 4).astype(int)
    extended["notmsa"] = (extended["V2129"] == 3).astype(int)
    extended["centralmsa"] = (extended["V2129"] == 1).astype(int)
    extended["notcentralmsa"] = (extended["V2129"] == 2).astype(int)
    date = pd.to_datetime({"year": extended["V3027"], "month": extended["V3025"], "day": extended["V3026"]}, errors="coerce")
    extended["age_DACA"] = np.floor(extended["age"] - (date - pd.Timestamp("2012-06-15")).dt.days / 365 + 0.5)
    extended["age_fake_DACA"] = np.floor(extended["age"] - (date - pd.Timestamp("2008-06-15")).dt.days / 365 + 0.5)
    extended["treat"] = np.where(extended["age_DACA"].isna(), np.nan, (extended["age_DACA"] < 31).astype(int))
    extended["fake_treat"] = (extended["age_fake_DACA"] <= 30).astype(int)
    extended["afterplacebo"] = (extended["year_incident"] >= 2008).astype(int)
    extended["interaction_placebo"] = extended["treat"] * extended["afterplacebo"]
    extended["interaction_fake"] = extended["fake_treat"] * extended["afterplacebo"]
    extended["report"] = pd.to_numeric(extended["V4399"], errors="coerce").replace({2: 0, 3: np.nan, 8: np.nan})
    columns = ["age", "agesq", "year_incident", "region", "hispanic", "white_nonhispanic", "female", "married", "age_DACA", "age_fake_DACA", "treat", "fake_treat", "afterplacebo", "interaction_placebo", "interaction_fake", "report"] + GEO
    extended = extended[list(dict.fromkeys(columns))].copy()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    extended.to_pickle(EXTENDED)

placebo = extended.loc[(extended["hispanic"] == 1) & extended["year_incident"].isin([2005, 2006, 2007, 2009, 2010, 2011])].copy()
table1 = []

for name, interaction, controls, cohort in [("(1) After placebo", "interaction_placebo", [], "age_DACA"), ("(2) + controls", "interaction_placebo", DEMO + GEO, "age_DACA"), ("(3) Fake eligibility", "interaction_fake", [], "age_fake_DACA"), ("(4) + controls", "interaction_fake", DEMO + GEO, "age_fake_DACA")]:
    data = placebo.dropna(subset=["report", interaction, "year_incident", cohort, "region"] + controls).copy()
    while True:
        keep = (data.groupby(cohort)[cohort].transform("size") > 1) & (data.groupby("year_incident")["year_incident"].transform("size") > 1)
        if keep.all():
            break
        data = data.loc[keep].copy()
    clusters = data["region"].astype(int).astype(str) + "_" + data["year_incident"].astype(int).astype(str)
    formula = "report ~ " + " + ".join([interaction] + controls + [f"C({cohort})", "C(year_incident)"])
    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": True, "df_correction": True}, use_t=True)
    table1.append((name, model.params[interaction], model.bse[interaction], data["report"].mean(), int(model.nobs)))

print("\nTABLE 1: 2008 PLACEBO TESTS")
print(f"{'Specification':<24}{'Beta':>10}{'SE':>10}{'Mean':>10}{'N':>9}")
for name, beta, se, mean, n in table1:
    print(f"{name:<24}{beta:>10.4f}{se:>10.4f}{mean:>10.3f}{n:>9,}")

FIGURES.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)

for ax, title, group in [(axes[0], "(a) Hispanic victims", "hispanic"), (axes[1], "(b) White non-Hispanic victims", "white_nonhispanic")]:
    data = extended.loc[(extended[group] == 1) & extended["treat"].isin([0, 1])]
    rates = data.groupby(["year_incident", "treat"])["report"].mean().unstack("treat").sort_index().rolling(2, min_periods=2).mean()
    years = np.arange(2006, 2017)
    for eligibility, label in [(1.0, "Age-eligible"), (0.0, "Age-ineligible")]:
        ax.plot(years, rates.reindex(years)[eligibility], marker="o", linewidth=1.5, markersize=3.5, label=label)
    ax.axvline(2011.5, linestyle="--", linewidth=1, color="black")
    ax.set(title=title, xlabel="Year of incident", ylim=(0.30, 0.50))
    ax.set_xticks(years[::2])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)

axes[0].set_ylabel("Share of incidents reported to the police")
fig.tight_layout()
fig.savefig(FIGURES / "figure1_reporting_trends.png", dpi=220, bbox_inches="tight")
plt.close(fig)

es = extended.loc[(extended["hispanic"] == 1) & extended["year_incident"].between(2007, 2016)].dropna(subset=["report", "treat", "age_DACA", "region", "year_incident"] + DEMO + GEO).copy()
years = list(range(2007, 2017))
terms = []
for year in years:
    if year != 2011:
        name = f"eligible_{year}"
        es[name] = es["treat"] * (es["year_incident"] == year).astype(int)
        terms.append(name)

es = es.loc[es.groupby("age_DACA")["age_DACA"].transform("size") > 1].copy()
clusters = es["region"].astype(int).astype(str) + "_" + es["year_incident"].astype(int).astype(str)
formula = "report ~ " + " + ".join(terms + DEMO + GEO + ["C(age_DACA)", "C(year_incident)"])
model = smf.ols(formula, data=es).fit(cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": True, "df_correction": True}, use_t=True)

betas, low, high = [], [], []
for year in years:
    if year == 2011:
        betas.append(0.0)
        low.append(0.0)
        high.append(0.0)
    else:
        betas.append(model.params[f"eligible_{year}"])
        ci = model.conf_int().loc[f"eligible_{year}"]
        low.append(ci.iloc[0])
        high.append(ci.iloc[1])

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(years, betas, yerr=[np.array(betas) - low, np.array(high) - betas], fmt="o", capsize=3, markersize=4, linewidth=1.2)
ax.axhline(0, linewidth=1, color="black", linestyle="--")
ax.axvline(2011.5, linewidth=1, color="black", linestyle="--")
ax.set(xticks=years, xlabel="Year of incident", ylabel="Coefficient", title="Effect of DACA age eligibility on reporting (2011 omitted)")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(FIGURES / "figure2_event_study.png", dpi=220, bbox_inches="tight")
plt.close(fig)
