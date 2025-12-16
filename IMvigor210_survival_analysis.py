# ==============================================================================
# IMvigor210_survival_analysis.py
# Multivariate Cox Regression Analysis for IMvigor210 Cohort
# Adjusting for ECOG, PD-L1 IC, TMB, and Metastatic Disease Status
# ==============================================================================

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Load and Merge Data
# ==============================================================================

clinical = pd.read_csv("data/IMvigor210_clinical.csv")
predictions = pd.read_csv("results/IMvigor210_predicted_labels.csv")

# Merge data
df = clinical.merge(predictions, on="sample_id", how="left")

print("=== Original Data ===")
print(f"Total samples: {len(df)}")

# ==============================================================================
# 2. Variable Preprocessing
# ==============================================================================

df = df.rename(columns={
    'Baseline ECOG Score': 'ECOG',
    'IC Level': 'IC_Level',
    'FMOne mutation burden per MB': 'TMB',
    'Met Disease Status': 'Met_Status',
    'os': 'OS_time',
    'censOS': 'OS_status'
})

# HotHigh binary (1 = HotHigh, 0 = Others)
df['HotHigh'] = (df['predicted_label'] == 'HotHigh').astype(int)

# TMB numeric
df['TMB'] = pd.to_numeric(df['TMB'], errors='coerce')

# ECOG dummy variables (reference: ECOG 0)
df['ECOG_1'] = (df['ECOG'] == 1).astype(int)
df['ECOG_2'] = (df['ECOG'] == 2).astype(int)

# IC Level dummy variables (reference: IC0)
df['IC_1'] = (df['IC_Level'] == 'IC1').astype(int)
df['IC_2plus'] = (df['IC_Level'] == 'IC2+').astype(int)

# Met Status dummy variables (reference: NA/LN Only)
df['Met_Liver'] = (df['Met_Status'] == 'Liver').astype(int)
df['Met_Visceral'] = (df['Met_Status'] == 'Visceral').astype(int)

# ==============================================================================
# 3. Define Cohorts
# ==============================================================================

# Full cohort (OS available samples)
full_cohort = df[df['OS_time'].notna() & df['OS_status'].notna()].copy()

print(f"\n=== Full Cohort (OS available) ===")
print(f"N = {len(full_cohort)}")
print(f"HotHigh: {full_cohort['HotHigh'].sum()}")
print(f"Others: {len(full_cohort) - full_cohort['HotHigh'].sum()}")

# TMB-high cohort: TMB >= 10 (for survival analysis)
tmb_high_cohort = df[df['TMB'] >= 10].copy()

print(f"\n=== TMB-high Cohort (TMB >= 10) ===")
print(f"N = {len(tmb_high_cohort)}")
print(f"HotHigh: {tmb_high_cohort['HotHigh'].sum()}")
print(f"Others: {len(tmb_high_cohort) - tmb_high_cohort['HotHigh'].sum()}")

# ==============================================================================
# 4. TMB-high Cohort: Univariate Cox Regression
# ==============================================================================

print("\n")
print("=" * 70)
print(f"     TMB-high Cohort (N = {len(tmb_high_cohort)}) - Cox Regression")
print("=" * 70)

cox_vars_uni = ['OS_time', 'OS_status', 'HotHigh']

# Univariate Cox
print("\n--- Univariate Cox (HotHigh only) ---")
tmb_uni = tmb_high_cohort[cox_vars_uni].dropna()
print(f"N = {len(tmb_uni)}, HotHigh = {tmb_uni['HotHigh'].sum()}, Others = {len(tmb_uni) - tmb_uni['HotHigh'].sum()}")

cph_uni_tmb = CoxPHFitter()
cph_uni_tmb.fit(tmb_uni, duration_col='OS_time', event_col='OS_status')
print(cph_uni_tmb.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

# Log-rank test
hothigh = tmb_uni[tmb_uni['HotHigh'] == 1]
others = tmb_uni[tmb_uni['HotHigh'] == 0]
lr_result = logrank_test(hothigh['OS_time'], others['OS_time'],
                          event_observed_A=hothigh['OS_status'],
                          event_observed_B=others['OS_status'])
print(f"\nLog-rank p-value: {lr_result.p_value:.4f}")

# ==============================================================================
# 5. TMB-high Cohort: Multivariate Cox Regression
# ==============================================================================

cox_vars_multi_tmb = ['OS_time', 'OS_status', 'HotHigh', 'ECOG_1', 'ECOG_2',
                       'IC_1', 'IC_2plus', 'Met_Liver', 'Met_Visceral']

print("\n--- Multivariate Cox (HotHigh + ECOG + IC + Met Status) ---")
tmb_multi = tmb_high_cohort[cox_vars_multi_tmb].dropna()
print(f"N = {len(tmb_multi)} (complete cases)")

cph_multi_tmb = CoxPHFitter()
cph_multi_tmb.fit(tmb_multi, duration_col='OS_time', event_col='OS_status')
print(cph_multi_tmb.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

# ==============================================================================
# 6. Full Cohort: Cox Regression (Reference)
# ==============================================================================

print("\n")
print("=" * 70)
print(f"     Full Cohort (N = {len(full_cohort)}) - Cox Regression")
print("=" * 70)

cox_vars_multi_full = ['OS_time', 'OS_status', 'HotHigh', 'ECOG_1', 'ECOG_2',
                        'IC_1', 'IC_2plus', 'TMB', 'Met_Liver', 'Met_Visceral']

# Univariate
print("\n--- Univariate Cox (HotHigh only) ---")
full_uni = full_cohort[cox_vars_uni].dropna()
cph_uni_full = CoxPHFitter()
cph_uni_full.fit(full_uni, duration_col='OS_time', event_col='OS_status')
print(cph_uni_full.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

# Multivariate
print("\n--- Multivariate Cox (HotHigh + ECOG + IC + TMB + Met Status) ---")
full_multi = full_cohort[cox_vars_multi_full].dropna()
print(f"N = {len(full_multi)} (complete cases)")

cph_multi_full = CoxPHFitter()
cph_multi_full.fit(full_multi, duration_col='OS_time', event_col='OS_status')
print(cph_multi_full.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

# ==============================================================================
# 7. Summary Table
# ==============================================================================

print("\n")
print("=" * 70)
print("                    Summary Table - HotHigh")
print("=" * 70)

def extract_hothigh_results(cph, cohort_name, model_name, n):
    row = cph.summary.loc['HotHigh']
    return {
        'Cohort': cohort_name,
        'N': n,
        'Model': model_name,
        'HR': round(row['exp(coef)'], 2),
        'CI_lower': round(row['exp(coef) lower 95%'], 2),
        'CI_upper': round(row['exp(coef) upper 95%'], 2),
        'p_value': round(row['p'], 4)
    }

results = []
results.append(extract_hothigh_results(cph_uni_full, 'Full cohort', 'Univariate', len(full_uni)))
results.append(extract_hothigh_results(cph_multi_full, 'Full cohort', 'Multivariate', len(full_multi)))
results.append(extract_hothigh_results(cph_uni_tmb, 'TMB-high', 'Univariate', len(tmb_uni)))
results.append(extract_hothigh_results(cph_multi_tmb, 'TMB-high', 'Multivariate', len(tmb_multi)))

summary_df = pd.DataFrame(results)
summary_df['HR (95% CI)'] = summary_df.apply(
    lambda x: f"{x['HR']:.2f} ({x['CI_lower']:.2f}-{x['CI_upper']:.2f})", axis=1
)

print(summary_df[['Cohort', 'N', 'Model', 'HR (95% CI)', 'p_value']].to_string(index=False))

# Save results
summary_df.to_csv("results/multivariate_cox_summary.csv", index=False)
print("\nSaved: results/multivariate_cox_summary.csv")

# ==============================================================================
# 8. Missing Data Check
# ==============================================================================

print("\n=== Missing Data (TMB-high cohort) ===")
print(f"  ECOG missing: {tmb_high_cohort['ECOG'].isna().sum()}")
print(f"  IC_Level missing: {tmb_high_cohort['IC_Level'].isna().sum()}")
print(f"  Met_Status missing: {tmb_high_cohort['Met_Status'].isna().sum()}")
print(f"  OS_time missing: {tmb_high_cohort['OS_time'].isna().sum()}")
print(f"  OS_status missing: {tmb_high_cohort['OS_status'].isna().sum()}")

# ==============================================================================
# 9. Full Multivariate Results (for Forest Plot)
# ==============================================================================

print("\n")
print("=" * 70)
print("     TMB-high Multivariate Full Results (Forest Plot)")
print("=" * 70)

mv_results = cph_multi_tmb.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].copy()
mv_results.columns = ['HR', 'CI_lower', 'CI_upper', 'p_value']
mv_results['HR (95% CI)'] = mv_results.apply(
    lambda x: f"{x['HR']:.2f} ({x['CI_lower']:.2f}-{x['CI_upper']:.2f})", axis=1
)
mv_results['p_value'] = mv_results['p_value'].round(4)

print(mv_results[['HR (95% CI)', 'p_value']])

# Save full multivariate results
mv_results.to_csv("results/multivariate_cox_full_results.csv")
print("\nSaved: results/multivariate_cox_full_results.csv")

print("\n" + "=" * 70)
print("Analysis Complete!")
print("=" * 70)
