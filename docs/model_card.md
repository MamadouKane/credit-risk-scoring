# Model card: application scorecard (champion)

## Intended use

**Purpose:** estimate the probability of early payment difficulty of a consumer-loan applicant at origination, and support an approve or reject decision with a cost-based threshold and plain-language reasons.

**Users:** credit analysts and risk teams, as decision support with human review of rejections.

**Out of scope:** automatic decisions without human oversight; regulatory capital (IRB) or IFRS 9 provisions (the target is not the regulatory default definition); pricing; any population other than the Home Credit applicants it was trained on.

## Data

- Home Credit Default Risk (Kaggle): 307,511 labelled applications, plus `bureau` (1.7M external credits) and `installments_payments` (13.6M instalments).
- Split: 70 % train (215,257), 15 % calibration (46,127), 15 % test (46,127), stratified on the target, seed 42. The test set was used once.
- Target: `TARGET = 1` for payment difficulties on the first instalments (8.07 %). This is an early-payment-difficulty proxy, not the 90-days-past-due default definition.
- Accepted applicants only: no reject inference.

## Model

- Optimal binning with monotonic risk per variable, minimum 5 % of rows per bin, missing values in their own bin.
- Weight of Evidence transformation, logistic regression on 20 variables, all coefficients negative.
- Points: 600 at odds 50:1, 20 points to double the odds. Grades A to E are the quintiles of the training scores.
- Calibration: sigmoid, fitted on the calibration set (no change in practice: the logistic model is already calibrated).
- Decision: approve if calibrated PD < 0.0846, the threshold that minimises LGD x EAD on approved defaulters plus margin x EAD on rejected good clients (LGD 45 %, margin 5 %, EAD = credit amount, all assumed).

## Performance (test set)

| Metric | Value |
|---|---|
| AUC / Gini | 0.748 / 0.495 |
| KS | 0.373 |
| PR-AUC | 0.225 |
| Brier score | 0.0686 |
| Approval rate | 66.6 % |
| Default rate among approved | 3.94 % |
| Expected cost per 1,000 applications vs approve-all | -27.6 % |

Challenger (XGBoost, calibrated): Gini 0.555, KS 0.422, Brier 0.0667. Not deployed: its decisions cannot be broken down into points per variable.

## Fairness

Gender is excluded from the features. Approval ratio men vs women: 0.86. Approval rate by age band: 27.6 % (20 to 25) to 84.3 % (55 to 70), ratio 0.33, although age is not a feature. Default rates among approved applicants are similar across groups (3.5 % to 4.9 %). The age gap requires a documented business justification and human review before any use.

## Monitoring

PSI per scorecard variable and on the product mix, reading: below 0.1 stable, 0.1 to 0.25 to watch, above 0.25 significant shift. Retraining triggers: a variable above 0.25, or the observed default rate per grade leaving its expected range once outcomes are known.

## Limitations and risks

- External scores `EXT_SOURCE_*` carry most of the signal and are not documented.
- No out-of-time validation possible (relative days only).
- Cost assumptions drive the threshold more than the model (see the sensitivity table in the README).
- High-risk system under the EU AI Act (Annex III): requires risk management, data governance, logging, human oversight and transparency before any production use.
