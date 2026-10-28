# Pearson (2024) Replication

Replication of "The Impact of Immigration Status on Crime Reporting: Evidence from DACA," published in the Journal of Urban Economics (2024).

This replication is part of the STAT 5200 final project at the University of Pennsylvania.

The objective is to reproduce the paper's main difference-in-differences estimates and examine the relationship between DACA eligibility and crime reporting.

Original paper DOI: 10.1016/j.jue.2024.103670

## Data

The analysis uses NCVS household, person, and incident files
from ICPSR for 2007–2017, using the original 2016 release.

The raw data are not included in this repository. Users must
download the files separately and place them in the year-specific
folders under data_raw/NCVS_files/.

## Replication Results

| Column | Published Estimate | Replicated Estimate | Replicated SE |
|--------|-------------------:|--------------------:|--------------:|
| 1 | 0.086 | 0.0854 | 0.0378 |
| 2 | 0.084 | 0.0801 | 0.0418 |
| 3 | 0.083 | 0.0797 | 0.0418 |
| 4 | 0.089 | 0.0853 | 0.0437 |
| 5 | 0.108 | 0.1064 | 0.0431 |
| 6 | 0.105 | 0.1050 | 0.0419 |

The published regressions use 5,364 observations, while the
replicated regressions use 5,232 observations.

The estimates are close to those reported in the paper, but
the replication is not exact. The difference in sample size
remains unresolved.

## Running the Code

Install the dependencies:

pip install pandas numpy statsmodels

Set the data directory in table2_replication.py to the local
location of the NCVS files, then run:

python code/table2_replication.py

The script prints the regression results directly in the terminal
