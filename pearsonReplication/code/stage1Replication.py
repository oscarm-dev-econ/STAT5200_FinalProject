# Pearson (2024) replication - Stage 1
# Reads annual NCVS files (2005-2017) and checks the IPUMS ACS sample.
# Constructs the baseline incident sample and replicates Table 2 (columns 1-6).
# Outputs: ACS and NCVS sample sizes, mean reporting rate, unadjusted DiD, and Table 2 results in the terminal.
# Saves: data_processed/pearson_table2_baseline.pkl for later replication scripts.

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

root = Path(r"C:\Users\Oscar\Dropbox\STAT5200_FinalProject\pearsonReplicationV2")
ncvs_folder = root / "data_raw" / "NCVS_files"
acs_file = root / "data_raw" / "IPUMS_ACS" / "usa_00007.dta"
output_folder = root / "data_processed"

hhd_cols = ["IDHH", "YEARQ", "V2014", "V2026", "V2129", "V2127B"]
person_cols = ["IDHH", "IDPER", "YEARQ", "V3014", "V3015", "V3017", "V3020", "V3023A", "V3024", "V3025", "V3026", "V3027"]
incident_cols = ["IDHH", "IDPER", "YEARQ", "V4015", "V4399", "V4529", "V4321", "V4364", "V4288", "V4323", "V4111", "V4127", "WGTVICCY"]
acs_required = ["year", "age", "sex", "marst", "educ", "educd", "hispan", "raced", "region", "metro", "ownershp", "hhincome", "citizen", "birthyr", "yrimmig", "perwt"]

with pd.io.stata.StataReader(acs_file, convert_categoricals=False) as reader:
    acs_columns = {name.lower(): name for name in reader.variable_labels()}

missing = set(acs_required) - set(acs_columns)
if missing:
    raise ValueError(f"Missing ACS variables: {sorted(missing)}")

acs_n = 0
acs_cols = [acs_columns[name] for name in ["year", "hispan", "educd", "educ"]]
for chunk in pd.read_stata(acs_file, columns=acs_cols, chunksize=150_000, convert_categoricals=False):
    chunk.columns = [col.lower() for col in chunk.columns]
    acs_n += ((chunk["hispan"] > 0) & (chunk["educd"] >= 62) & (chunk["educ"] != 999) & chunk["year"].between(2005, 2017)).sum()

annual_data = []
for year in range(2005, 2018):
    folder = ncvs_folder / str(year)
    hhd = pd.read_stata(folder / f"{year}_hhd.dta", columns=hhd_cols, convert_categoricals=False)
    person = pd.read_stata(folder / f"{year}_person.dta", columns=person_cols, convert_categoricals=False)
    incident = pd.read_stata(folder / f"{year}_incident.dta", columns=incident_cols, convert_categoricals=False)

    hhd = hhd.drop_duplicates(["IDHH", "YEARQ"], keep="first")
    person = person.drop_duplicates(["IDPER", "YEARQ"], keep="first")
    person = person.merge(hhd, on=["IDHH", "YEARQ"], how="left", validate="m:1")
    incident = incident.merge(person.drop(columns="IDHH"), on=["IDPER", "YEARQ"], how="left", validate="m:1")
    incident["source_survey_year"] = year
    annual_data.append(incident)

df = pd.concat(annual_data, ignore_index=True)
del annual_data

for name, source in {"age": "V3014", "year_incident": "V4015", "hispanic": "V3024", "education_code": "V3020", "region": "V2127B"}.items():
    df[name] = pd.to_numeric(df[source], errors="coerce")

df["hs_plus"] = ((df["education_code"] > 12) & ~df["education_code"].isin([27, 98, 99])).astype(int)
interview_date = pd.to_datetime({"year": df["V3027"], "month": df["V3025"], "day": df["V3026"]}, errors="coerce")
df["age_DACA"] = np.floor(df["age"] - (interview_date - pd.Timestamp("2012-06-15")).dt.days / 365 + 0.5)
df["treat"] = np.where(df["age_DACA"].isna(), np.nan, (df["age_DACA"] < 31).astype(int))
df["report"] = pd.to_numeric(df["V4399"], errors="coerce").replace({2: 0, 3: np.nan, 8: np.nan})

df["female"] = (df["V3017"] == 2).astype(int)
df["married"] = (df["V3015"] == 1).astype(int)
df["agesq"] = df["age"] ** 2

df["midwest"] = (df["region"] == 2).astype(int)
df["south"] = (df["region"] == 3).astype(int)
df["west"] = (df["region"] == 4).astype(int)
df["notmsa"] = (df["V2129"] == 3).astype(int)
df["centralmsa"] = (df["V2129"] == 1).astype(int)
df["notcentralmsa"] = (df["V2129"] == 2).astype(int)

df["own"] = (df["V2014"] == 1).astype(int)
df["post_2ndary_deg"] = df["education_code"].isin([41, 42, 43, 44, 45]).astype(int)
income = pd.to_numeric(df["V2026"], errors="coerce")
df["incomeunk"] = ((income >= 98) | income.isna()).astype(int)
for category in range(1, 15):
    df[f"income{category}"] = (income == category).astype(int)

crime = pd.to_numeric(df["V4529"], errors="coerce")
df["sex_rape"] = crime.isin([1, 2, 3, 4, 15, 16, 18, 19]).astype(int)
df["robbery"] = crime.between(5, 10).astype(int)
df["assault"] = (crime.between(11, 14) | crime.isin([17, 20])).astype(int)
df["theft"] = (crime.isin([21, 22, 23]) | (crime >= 40) | crime.isna()).astype(int)
df["burglary"] = crime.between(31, 33).astype(int)

cash = pd.to_numeric(df["V4321"], errors="coerce").replace({99999: np.nan})
property_value = pd.to_numeric(df["V4364"], errors="coerce").replace({99998: np.nan, 99999: np.nan})
cash = cash.mask(df["V4288"] == 2, 0)
property_value = property_value.mask((df["V4288"] == 2) | (df["V4323"] == 1), 0)
total_value = cash.add(property_value, fill_value=0)
total_value.loc[cash.isna() & property_value.isna()] = np.nan
df["value_ind"] = (total_value.notna() & (total_value > 0)).astype(int)
df["lnvalue"] = 0.0
df.loc[df["value_ind"] == 1, "lnvalue"] = np.log(total_value.loc[df["value_ind"] == 1])
df["injuries_yes"] = (df["V4111"] == 1).astype(int)
df["med_care"] = (df["V4127"] == 1).astype(int)

base = df.loc[df["age"].between(22, 50) & (df["hispanic"] == 1) & (df["hs_plus"] == 1) & df["year_incident"].between(2007, 2016) & (df["year_incident"] != 2012) & df["report"].notna()].copy()
base = base.dropna(subset=["age_DACA", "treat", "region", "year_incident"])
base["after"] = (base["year_incident"] >= 2013).astype(int)
base["interaction"] = base["treat"] * base["after"]

output_folder.mkdir(parents=True, exist_ok=True)
base.to_pickle(output_folder / "pearson_table2_baseline.pkl")

means = base.groupby(["treat", "after"])["report"].mean()
simple_did = (means[1.0, 1] - means[1.0, 0]) - (means[0.0, 1] - means[0.0, 0])

print("Pearson (2024) - Table 2")
print(f"ACS Hispanic HS+ sample: {acs_n:,}")
print(f"NCVS sample: {len(base):,} | Mean reporting: {base['report'].mean():.3f} | Unadjusted DiD: {simple_did:.4f}")

age_controls = ["agesq", "female", "married"]
geo_controls = ["midwest", "south", "west", "notmsa", "centralmsa", "notcentralmsa"]
socio_controls = ["incomeunk"] + [f"income{i}" for i in range(1, 15)] + ["own", "post_2ndary_deg"]
crime_controls = ["sex_rape", "robbery", "assault", "theft", "burglary"]
severity_controls = ["value_ind", "lnvalue", "injuries_yes", "med_care"]
models = [[], age_controls, age_controls + geo_controls, age_controls + geo_controls + socio_controls, age_controls + geo_controls + socio_controls + crime_controls, age_controls + geo_controls + socio_controls + crime_controls + severity_controls]

print(f"\n{'Column':<10}{'N':>8}{'Coef.':>12}{'Cluster SE':>14}")
for i, controls in enumerate(models, 1):
    sample = base.dropna(subset=["report", "interaction", "age_DACA", "year_incident", "region"] + controls).copy()
    sample["cluster"] = sample["region"].astype(int).astype(str) + "_" + sample["year_incident"].astype(int).astype(str)
    formula = "report ~ interaction + " + " + ".join(controls + ["C(age_DACA)", "C(year_incident)"])
    fit = smf.ols(formula, data=sample).fit(cov_type="cluster", cov_kwds={"groups": sample["cluster"], "use_correction": True, "df_correction": True}, use_t=True)
    print(f"{i:<10}{int(fit.nobs):>8,}{fit.params['interaction']:>12.4f}{fit.bse['interaction']:>14.4f}")
