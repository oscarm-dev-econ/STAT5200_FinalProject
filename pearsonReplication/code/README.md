# Pearson (2024) — Replication Scripts

This folder contains four Python scripts used to replicate results from Thomas Pearson (2024), *"The Impact of Immigration Status on Crime Reporting: Evidence from DACA,"* published in the Journal of Urban Economics.

### stage1Replication.py — Baseline Difference-in-Differences (Table 2)

Reads the National Crime Victimization Survey (NCVS) files from 2005–2017 and checks the IPUMS American Community Survey (ACS) sample. Constructs the baseline incident-level dataset and estimates all six difference-in-differences specifications in Table 2.

**Outputs:** ACS and NCVS sample sizes, mean crime reporting rate, unadjusted difference-in-differences estimate, and regression coefficients and clustered standard errors for Table 2.

**Saved file:** `data_processed/pearson_table2_baseline.pkl`

### stage2Replication.py — Heterogeneity, Placebo Tests, and Event Study (Tables 1 and 3; Figures 1 and 2)

Uses the baseline dataset and additional NCVS observations to replicate heterogeneous treatment effects by gender, education, and home ownership (Table 3), placebo tests using a hypothetical 2008 DACA implementation (Table 1), reporting trends by age eligibility (Figure 1), and event-study estimates (Figure 2).

**Outputs:** Table 1 and Table 3 regression results in the terminal, along with two figures saved as PNG files in `outputs/figures/`.

**Saved file:** `data_processed/pearson_stage2_extended_reporting.pkl`

### stage3Replication.py — Placebo Tests Across Demographic Groups (Figure 3)

Estimates the effect of DACA eligibility on crime reporting among three groups unlikely to benefit directly from the policy: Hispanics without a high school degree, Black non-Hispanics, and White non-Hispanics.

**Outputs:** Regression coefficients, clustered standard errors, mean reporting rates, and sample sizes for each group, along with Figure 3 saved as `outputs/figures/figure3_reporting_placebos.png`.

### stage4Replication.py — Random Forest Predictions and Heterogeneous Effects (Table 4)

Uses IPUMS ACS data from 2005–2017 to train separate annual random forest classifiers with 100 trees each. Applies these models to NCVS respondents to estimate their probability of being noncitizen childhood arrivals. Uses the resulting predictions to estimate the heterogeneous effects of DACA eligibility on crime reporting reported in Table 4.

**Outputs:** Prediction benchmarks, probability cutoffs, and Table 4 regression coefficients, clustered standard errors, reporting means, and sample sizes.

**Saved files:**
- `data_processed/pearson_ncvsperson_rf_predictions.pkl`
- `data_processed/pearson_table4_with_predictions.pkl`

### Replication Notes
The scripts require the original NCVS and IPUMS ACS data, which are not included in this repository because of data redistribution restrictions.
