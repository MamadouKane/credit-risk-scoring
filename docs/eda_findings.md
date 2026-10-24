# EDA findings

Findings from `notebooks/01_eda_application.ipynb` and `notebooks/02_eda_history.ipynb`. Each one changes a modelling decision. Numbers are computed on `application_train` (307,511 applications) unless stated otherwise.

## Application table

1. **Finding:** 8.07 % of applications are bads (24,825 out of 307,511); approving everyone gives 91.93 % accuracy and catches no defaulter.
   **Decision:** never report accuracy. Report Gini, KS, PR-AUC and Brier score, and stratify every split on `TARGET`.
2. **Finding:** 18.0 % of clients have `DAYS_EMPLOYED = 365243` (55,352 pensioners and 22 unemployed), and they default less (5.40 % vs 8.66 %).
   **Decision:** replace the placeholder by NaN and add a flag `DAYS_EMPLOYED_ANOM` in `data.clean`.
3. **Finding:** missingness predicts default: no building data means 9.15 % default vs 6.96 %, no `EXT_SOURCE_3` means 9.31 % vs 7.77 %.
   **Decision:** do not drop columns because they are mostly empty and do not impute blindly; missing values get their own bin in the scorecard.
4. **Finding:** the three external scores are the strongest and monotonic signals (IV 0.329, 0.306, 0.151): default falls from 20.0 % in the lowest `EXT_SOURCE_3` decile to 3.2 % in the highest.
   **Decision:** they are the backbone of the scorecard, binned with a monotonic trend.
5. **Finding:** strong redundancy: `AMT_GOODS_PRICE` vs `AMT_CREDIT` at 0.985 (Spearman), and each building feature exists three times (`_AVG` vs `_MEDI` at 0.995).
   **Decision:** keep only `_AVG` building features, use the ratio `CREDIT_GOODS_RATIO` instead of both amounts, and drop one variable of each pair above 0.8 in the scorecard.

## Credit history tables

6. **Finding:** `bureau` and `installments_payments` are one-to-many: median 4 external credits and 25 instalments per applicant.
   **Decision:** aggregate per `SK_ID_CURR` first, then left-join one row per applicant, and check that the row count stays at 307,511.
7. **Finding:** the two "no history" groups go in opposite directions: no Credit Bureau record means 10.12 % default (44,020 applicants), no previous Home Credit loan means 5.98 % (15,868 applicants).
   **Decision:** keep missing aggregates as NaN and add the flags `HAS_BUREAU` and `HAS_INSTALMENTS`; never fill them with 0.
8. **Finding:** past behaviour is monotonic with risk: more than 20 % late instalments means 11.99 % default (vs 6.80 % for none), more than 10 % underpaid means 10.79 % (vs 7.10 %), any overdue external credit means 16.20 % (3,334 applicants).
   **Decision:** build `LATE_SHARE`, `MAX_DAYS_LATE`, `UNDERPAID_SHARE`, `N_ACTIVE` and `ANY_OVERDUE` in `features.py`; they also give the plainest reasons for a refusal.
9. **Finding:** the history is dated before the application, except `DAYS_CREDIT_UPDATE` (17 records updated after it); 146 bureau end dates are more than 40 years old while the oldest credit is 8.0 years old.
   **Decision:** exclude `DAYS_CREDIT_UPDATE` from the features and set impossible end dates to NaN.
10. **Finding:** numeric variables are stable between train and test (all PSI below 0.1), but revolving loans fall from 9.52 % to 0.90 % of applications (PSI 0.21).
    **Decision:** monitor categorical variables as well as numeric ones in the drift report, and check performance per product.
