# Pearson (2024): replicate Figure 3, placebo effects on reporting among three groups unlikely to benefit directly from DACA.
# Outputs: coefficients, clustered standard errors, reporting means and sample sizes in the terminal.
# Saves outputs/figures/figure3_reporting_placebos.png; no CSV, Excel or intermediate datasets.

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\Oscar\Dropbox\STAT5200_FinalProject\pearsonReplicationV2")
NCVS = ROOT / "data_raw" / "NCVS_files"
FIGURES = ROOT / "outputs" / "figures"

hhd_cols = ["IDHH", "YEARQ", "V2127B", "V2129"]
person_cols = ["IDHH", "IDPER", "YEARQ", "V3014", "V3015", "V3017", "V3020", "V3023A", "V3024", "V3025", "V3026", "V3027"]
incident_cols = ["IDHH", "IDPER", "YEARQ", "V4015", "V4399"]
yearly = []

for year in range(2007, 2018):
    folder = NCVS / str(year)
    if not folder.exists():
        folder = NCVS
    hhd = pd.read_stata(folder / f"{year}_hhd.dta", columns=hhd_cols, convert_categoricals=False).drop_duplicates(["IDHH", "YEARQ"])
    person = pd.read_stata(folder / f"{year}_person.dta", columns=person_cols, convert_categoricals=False).drop_duplicates(["IDPER", "YEARQ"])
    inc = pd.read_stata(folder / f"{year}_incident.dta", columns=incident_cols, convert_categoricals=False)
    person = person.merge(hhd, on=["IDHH", "YEARQ"], how="left", validate="m:1")
    inc = inc.merge(person.drop(columns="IDHH"), on=["IDPER", "YEARQ"], how="left", validate="m:1")

    age = pd.to_numeric(inc["V3014"], errors="coerce")
    inc_year = pd.to_numeric(inc["V4015"], errors="coerce")
    hispanic = pd.to_numeric(inc["V3024"], errors="coerce")
    race = pd.to_numeric(inc["V3023A"], errors="coerce")
    edu = pd.to_numeric(inc["V3020"], errors="coerce")
    hsplus = (edu > 12) & ~edu.isin([27, 98, 99])
    eligible_group = ((hispanic == 1) & ~hsplus) | ((hispanic == 2) & hsplus & race.isin([1, 2]))
    keep = age.between(22, 50) & inc_year.between(2007, 2016) & (inc_year != 2012) & eligible_group
    yearly.append(inc.loc[keep].copy())

df = pd.concat(yearly, ignore_index=True)
df["age"] = pd.to_numeric(df["V3014"], errors="coerce")
df["year_incident"] = pd.to_numeric(df["V4015"], errors="coerce")
df["region"] = pd.to_numeric(df["V2127B"], errors="coerce")
df["report"] = pd.to_numeric(df["V4399"], errors="coerce").replace({2: 0, 3: np.nan, 8: np.nan})
df["agesq"] = df["age"] ** 2
df["female"] = (df["V3017"] == 2).astype(int)
df["married"] = (df["V3015"] == 1).astype(int)

for name, code in [("midwest", 2), ("south", 3), ("west", 4)]:
    df[name] = (df["region"] == code).astype(int)
for name, code in [("notmsa", 3), ("centralmsa", 1), ("notcentralmsa", 2)]:
    df[name] = (df["V2129"] == code).astype(int)

date = pd.to_datetime({"year": df["V3027"], "month": df["V3025"], "day": df["V3026"]}, errors="coerce")
df["age_DACA"] = np.floor(df["age"] - (date - pd.Timestamp("2012-06-15")).dt.days / 365 + 0.5)
df["treat"] = np.where(df["age_DACA"].isna(), np.nan, (df["age_DACA"] <= 30).astype(int))
df["interaction"] = df["treat"] * (df["year_incident"] >= 2013).astype(int)

hispanic = pd.to_numeric(df["V3024"], errors="coerce")
race = pd.to_numeric(df["V3023A"], errors="coerce")
edu = pd.to_numeric(df["V3020"], errors="coerce")
hsplus = (edu > 12) & ~edu.isin([27, 98, 99])
groups = [("Hispanics without HS degree", (hispanic == 1) & ~hsplus), ("Black non-Hispanics", (hispanic == 2) & (race == 2) & hsplus), ("White non-Hispanics", (hispanic == 2) & (race == 1) & hsplus)]
controls = ["agesq", "female", "married", "midwest", "south", "west", "notmsa", "centralmsa", "notcentralmsa"]
formula = "report ~ interaction + " + " + ".join(controls + ["C(age_DACA)", "C(year_incident)"])
results = []

for label, condition in groups:
    data = df.loc[condition].dropna(subset=["report", "interaction", "age_DACA", "year_incident", "region"] + controls).copy()
    while True:
        keep = (data.groupby("age_DACA")["age_DACA"].transform("size") > 1) & (data.groupby("year_incident")["year_incident"].transform("size") > 1)
        if keep.all():
            break
        data = data.loc[keep].copy()
    clusters = data["region"].astype(int).astype(str) + "_" + data["year_incident"].astype(int).astype(str)
    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": True, "df_correction": True}, use_t=True)
    ci = model.conf_int().loc["interaction"]
    results.append((label, model.params["interaction"], model.bse["interaction"], data["report"].mean(), int(model.nobs), ci.iloc[0], ci.iloc[1]))

print("\nFIGURE 3: PLACEBO EFFECTS ON CRIME REPORTING")
print(f"{'Group':<31}{'Beta':>10}{'SE':>10}{'Mean':>10}{'N':>10}")
for label, beta, se, mean, n, low, high in results:
    print(f"{label:<31}{beta:>10.4f}{se:>10.4f}{mean:>10.3f}{n:>10,}")

FIGURES.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 4.2))
for y, (label, beta, se, mean, n, low, high) in zip([2, 1, 0], results):
    ax.errorbar(beta, y, xerr=[[beta - low], [high - beta]], fmt="o", capsize=4, color="0.25", ecolor="0.4", markersize=6)
ax.axvline(0, linestyle="--", color="black", linewidth=1)
ax.set_yticks([2, 1, 0], [item[0] for item in results])
ax.set(xlim=(-0.25, 0.40), xlabel="Effect of DACA eligibility on reporting probability")
ax.set_xticks(np.arange(-0.25, 0.401, 0.05))
ax.grid(axis="x", alpha=0.15)
fig.tight_layout()
fig.savefig(FIGURES / "figure3_reporting_placebos.png", dpi=220, bbox_inches="tight")
plt.close(fig)
