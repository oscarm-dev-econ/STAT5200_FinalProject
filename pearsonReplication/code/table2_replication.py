
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

data_dir = Path(__file__).resolve().parent.parent / "data_raw" / "NCVS_files"

household_cols = ["IDHH", "YEARQ", "V2014", "V2026", "V2127B", "V2129"]
person_cols = ["IDPER", "IDHH", "YEARQ", "V3014", "V3015", "V3017", "V3020", "V3024", "V3025", "V3026", "V3027"]
incident_cols = ["IDPER", "YEARQ", "V4015", "V4399", "V4529", "V4321", "V4364", "V4288", "V4323", "V4111", "V4127"]

published_betas = [0.086, 0.084, 0.083, 0.089, 0.108, 0.105]
published_ses = [0.038, 0.044, 0.044, 0.046, 0.045, 0.044]

print("=" * 76)
print("PEARSON (2024) - TABLE 2 REPLICATION")
print("=" * 76)
print("Data directory:", data_dir)

all_incidents = []

for year in range(2007, 2018):
    folder = data_dir / str(year)
    household_file = folder / f"{year}_hhd.dta"
    person_file = folder / f"{year}_person.dta"
    incident_file = folder / f"{year}_incident.dta"

    for file in [household_file, person_file, incident_file]:
        if not file.is_file():
            raise FileNotFoundError(f"Missing NCVS file: {file}")

    households = pd.read_stata(household_file, columns=household_cols, convert_categoricals=False)
    people = pd.read_stata(person_file, columns=person_cols, convert_categoricals=False)
    incidents = pd.read_stata(incident_file, columns=incident_cols, convert_categoricals=False)

    households = households.drop_duplicates(["IDHH", "YEARQ"])
    people = people.drop_duplicates(["IDPER", "YEARQ"])
    people = people.merge(households, on=["IDHH", "YEARQ"], how="left", validate="many_to_one")
    incidents = incidents.merge(people, on=["IDPER", "YEARQ"], how="left", validate="many_to_one")

    all_incidents.append(incidents)
    print(f"{year}: {len(incidents):,} incidents")

data = pd.concat(all_incidents, ignore_index=True)
del all_incidents

age = pd.to_numeric(data["V3014"], errors="coerce")
education = pd.to_numeric(data["V3020"], errors="coerce")
hispanic = pd.to_numeric(data["V3024"], errors="coerce")
incident_year = pd.to_numeric(data["V4015"], errors="coerce")

high_school = ((education > 12) | education.isna()) & ~education.isin([27, 98, 99])
keep = (age.between(22, 50) & (hispanic == 1) & high_school & incident_year.between(2007, 2016) & (incident_year != 2012))

data = data.loc[keep].copy()
print(f"\nObservations after sample restrictions: {len(data):,}")

data["age"] = age.loc[keep]
data["agesq"] = data["age"] ** 2
data["year_incident"] = incident_year.loc[keep].astype(int)
data["female"] = (data["V3017"] == 2).astype(int)
data["married"] = (data["V3015"] == 1).astype(int)

interview_date = pd.to_datetime({
    "year": pd.to_numeric(data["V3027"], errors="coerce"),
    "month": pd.to_numeric(data["V3025"], errors="coerce"),
    "day": pd.to_numeric(data["V3026"], errors="coerce")
}, errors="coerce")

years_since_daca = (interview_date - pd.Timestamp("2012-06-15")).dt.days / 365.0
data["age_DACA"] = np.floor(data["age"] - years_since_daca + 0.5)
data["eligible"] = (data["age_DACA"] < 31).astype(float)
data.loc[data["age_DACA"].isna(), "eligible"] = np.nan
data["eligible_after"] = data["eligible"] * (data["year_incident"] >= 2013).astype(int)

data["report"] = pd.to_numeric(data["V4399"], errors="coerce").map({1: 1.0, 2: 0.0})
data["region"] = pd.to_numeric(data["V2127B"], errors="coerce")

for name, code in [("midwest", 2), ("south", 3), ("west", 4)]:
    data[name] = (data["region"] == code).astype(int)

msa = pd.to_numeric(data["V2129"], errors="coerce")
data["centralmsa"] = (msa == 1).astype(int)
data["notcentralmsa"] = (msa == 2).astype(int)
data["notmsa"] = (msa == 3).astype(int)

income = pd.to_numeric(data["V2026"], errors="coerce")
data["incomeunk"] = ((income >= 98) | income.isna()).astype(int)
for k in range(1, 15):
    data[f"income{k}"] = (income == k).astype(int)

data["own"] = (data["V2014"] == 1).astype(int)
data["post_2ndary_deg"] = education.loc[keep].isin([41, 42, 43, 44, 45]).astype(int)

crime = pd.to_numeric(data["V4529"], errors="coerce")
data["sex_rape"] = crime.isin([1, 2, 3, 4, 15, 16, 18, 19]).astype(int)
data["robbery"] = crime.between(5, 10).astype(int)
data["assault"] = (crime.between(11, 14) | crime.isin([17, 20])).astype(int)
data["theft"] = (crime.isin([21, 22, 23]) | (crime >= 40) | crime.isna()).astype(int)
data["burglary"] = crime.between(31, 33).astype(int)

taken = pd.to_numeric(data["V4288"], errors="coerce")
only_cash = pd.to_numeric(data["V4323"], errors="coerce")
cash = pd.to_numeric(data["V4321"], errors="coerce").replace(99999, np.nan)
property_value = pd.to_numeric(data["V4364"], errors="coerce").replace({99998: np.nan, 99999: np.nan})
cash.loc[taken == 2] = 0
property_value.loc[(taken == 2) | (only_cash == 1)] = 0
total_value = cash.add(property_value, fill_value=0)

data["value_ind"] = (total_value > 0).astype(int)
data["lnvalue"] = 0.0
positive = total_value > 0
data.loc[positive, "lnvalue"] = np.log(total_value.loc[positive])
data["injuries_yes"] = (data["V4111"] == 1).astype(int)
data["med_care"] = (data["V4127"] == 1).astype(int)

demographic = ["agesq", "female", "married"]
geographic = ["midwest", "south", "west", "notmsa", "centralmsa", "notcentralmsa"]
socioeconomic = ["incomeunk"] + [f"income{k}" for k in range(1, 15)] + ["own", "post_2ndary_deg"]
crime_types = ["sex_rape", "robbery", "assault", "theft", "burglary"]
crime_severity = ["value_ind", "lnvalue", "injuries_yes", "med_care"]

specifications = [[], demographic, demographic + geographic, demographic + geographic + socioeconomic, demographic + geographic + socioeconomic + crime_types, demographic + geographic + socioeconomic + crime_types + crime_severity]

results = []

for col, controls in enumerate(specifications, start=1):
    needed = ["report", "eligible_after", "year_incident", "age_DACA", "region"] + controls
    sample = data.dropna(subset=needed).copy()

    formula = "report ~ eligible_after + C(year_incident) + C(age_DACA)"
    if controls:
        formula += " + " + " + ".join(controls)

    clusters = pd.factorize(pd.MultiIndex.from_frame(sample[["region", "year_incident"]]))[0]
    model = smf.ols(formula, data=sample).fit(
        cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": True, "df_correction": True})

    results.append({
        "column": col, "beta_python": model.params["eligible_after"], "se_python": model.bse["eligible_after"],
        "beta_published": published_betas[col - 1], "se_published": published_ses[col - 1],
        "N": int(model.nobs), "mean_report": sample["report"].mean()
    })
    print(f"Finished column {col}/6")

results = pd.DataFrame(results)
results["beta_difference"] = results["beta_python"] - results["beta_published"]

print("\n" + "=" * 76)
print("TABLE 2: REPLICATION VS. PUBLISHED RESULTS")
print("=" * 76)
output_cols = ["column", "beta_python", "se_python", "beta_published", "se_published", "beta_difference", "N", "mean_report"]
print(results[output_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nPublished sample: 5,364 observations; reporting mean: 0.408")
print("No output files were created.")
