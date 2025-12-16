# ==============================================================================
# classifier_validation.py
# 20-Gene HotHigh Classifier: 5-Fold Cross-Validation and External Validation
# ==============================================================================

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 20-Gene Signature
# ==============================================================================

GENE_SET_20 = [
    'CTSS', 'STAT1', 'CD74', 'HLA-A', 'HLA-E', 'HLA-C', 'HLA-B', 'CFB',
    'HLA-DRA', 'HLA-DRB1', 'HLA-DQA1', 'TAP1', 'SOD2', 'IDO1', 'IFITM1',
    'SERPING1', 'C1S', 'B2M', 'RNF213', 'IFI30'
]

# ==============================================================================
# Bootstrap AUC 95% CI Function
# ==============================================================================

def bootstrap_auc_ci(y_true, y_pred_proba, n_bootstrap=2000, ci=0.95, random_state=42):
    """
    Calculate AUC 95% CI using bootstrap resampling
    
    Parameters:
    - n_bootstrap: number of iterations (default 2000)
    - ci: confidence level (default 0.95 = 95%)
    """
    n = len(y_true)
    aucs = []

    np.random.seed(random_state)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc = roc_auc_score(y_true[idx], y_pred_proba[idx])
        aucs.append(auc)

    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)

    return lower, upper

# ==============================================================================
# 1. Load Internal Cohort (TCGA dMMR/MSI-H, n=259)
# ==============================================================================

print("=" * 70)
print("STEP 1: Internal Cohort - 5-Fold Cross-Validation")
print("=" * 70)

internal_df = pd.read_csv('data/TCGA_expression_with_labels.csv')

available_genes_int = [g for g in GENE_SET_20 if g in internal_df.columns]
print(f"Available genes: {len(available_genes_int)}/{len(GENE_SET_20)}")

X_raw = internal_df[available_genes_int]
y = (internal_df['combined_label'] == 'HotHigh').astype(int)

print(f"\nInternal Cohort:")
print(f"  Samples: {len(X_raw)}")
print(f"  HotHigh: {y.sum()} ({y.sum() / len(y) * 100:.2f}%)")
print(f"  Others: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")

# ==============================================================================
# 2. 5-Fold Stratified Cross-Validation
# ==============================================================================

print(f"\n{'=' * 70}")
print("5-Fold Cross-Validation")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {
    'fold': [], 'auc': [], 'accuracy': [],
    'sensitivity': [], 'specificity': [], 'ppv': [], 'npv': []
}

y_pred_proba_cv = np.zeros(len(y))
y_pred_cv = np.zeros(len(y))

fold_num = 1
for train_idx, val_idx in skf.split(X_raw, y):
    print(f"\n--- Fold {fold_num}/5 ---")

    X_train_fold = X_raw.iloc[train_idx]
    X_val_fold = X_raw.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    # Z-score normalization (using training set statistics)
    train_means = X_train_fold.mean()
    train_stds = X_train_fold.std()
    X_train_zscore = (X_train_fold - train_means) / train_stds
    X_val_zscore = (X_val_fold - train_means) / train_stds

    # Train model
    model_fold = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', C=1.0)
    model_fold.fit(X_train_zscore, y_train_fold)

    # Predict
    y_pred_proba_fold = model_fold.predict_proba(X_val_zscore)[:, 1]
    y_pred_fold = model_fold.predict(X_val_zscore)

    # Store OOF predictions
    y_pred_proba_cv[val_idx] = y_pred_proba_fold
    y_pred_cv[val_idx] = y_pred_fold

    # Calculate metrics
    auc_fold = roc_auc_score(y_val_fold, y_pred_proba_fold)
    cm_fold = confusion_matrix(y_val_fold, y_pred_fold)

    accuracy_fold = (cm_fold[0, 0] + cm_fold[1, 1]) / cm_fold.sum()
    sensitivity_fold = cm_fold[1, 1] / (cm_fold[1, 0] + cm_fold[1, 1]) if (cm_fold[1, 0] + cm_fold[1, 1]) > 0 else 0
    specificity_fold = cm_fold[0, 0] / (cm_fold[0, 0] + cm_fold[0, 1]) if (cm_fold[0, 0] + cm_fold[0, 1]) > 0 else 0
    ppv_fold = cm_fold[1, 1] / (cm_fold[0, 1] + cm_fold[1, 1]) if (cm_fold[0, 1] + cm_fold[1, 1]) > 0 else 0
    npv_fold = cm_fold[0, 0] / (cm_fold[0, 0] + cm_fold[1, 0]) if (cm_fold[0, 0] + cm_fold[1, 0]) > 0 else 0

    cv_results['fold'].append(fold_num)
    cv_results['auc'].append(auc_fold)
    cv_results['accuracy'].append(accuracy_fold)
    cv_results['sensitivity'].append(sensitivity_fold)
    cv_results['specificity'].append(specificity_fold)
    cv_results['ppv'].append(ppv_fold)
    cv_results['npv'].append(npv_fold)

    print(f"  AUC: {auc_fold:.4f}")
    print(f"  Accuracy: {accuracy_fold:.4f}")
    print(f"  Sensitivity: {sensitivity_fold:.4f}")
    print(f"  Specificity: {specificity_fold:.4f}")

    fold_num += 1

# ==============================================================================
# 3. Cross-Validation Summary
# ==============================================================================

print(f"\n{'=' * 70}")
print("Cross-Validation Summary (5-Fold)")
print("=" * 70)

cv_df = pd.DataFrame(cv_results)
print(cv_df.to_string(index=False))

print(f"\nMean Performance:")
for metric in ['auc', 'accuracy', 'sensitivity', 'specificity', 'ppv', 'npv']:
    mean_val = cv_df[metric].mean()
    std_val = cv_df[metric].std()
    print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

# ==============================================================================
# 4. Overall Out-of-Fold Performance
# ==============================================================================

print(f"\n{'=' * 70}")
print("Overall Out-of-Fold (OOF) Performance")
print("=" * 70)

auc_cv_overall = roc_auc_score(y, y_pred_proba_cv)
cm_cv_overall = confusion_matrix(y, y_pred_cv)

print(f"Overall AUC: {auc_cv_overall:.4f}")
print(f"\nConfusion Matrix:")
print(cm_cv_overall)

accuracy_cv = (cm_cv_overall[0, 0] + cm_cv_overall[1, 1]) / cm_cv_overall.sum()
sensitivity_cv = cm_cv_overall[1, 1] / (cm_cv_overall[1, 0] + cm_cv_overall[1, 1])
specificity_cv = cm_cv_overall[0, 0] / (cm_cv_overall[0, 0] + cm_cv_overall[0, 1])
ppv_cv = cm_cv_overall[1, 1] / (cm_cv_overall[0, 1] + cm_cv_overall[1, 1])
npv_cv = cm_cv_overall[0, 0] / (cm_cv_overall[0, 0] + cm_cv_overall[1, 0])

print(f"\nOverall Accuracy:    {accuracy_cv:.4f} ({accuracy_cv * 100:.2f}%)")
print(f"Overall Sensitivity: {sensitivity_cv:.4f} ({sensitivity_cv * 100:.2f}%)")
print(f"Overall Specificity: {specificity_cv:.4f} ({specificity_cv * 100:.2f}%)")
print(f"Overall PPV:         {ppv_cv:.4f} ({ppv_cv * 100:.2f}%)")
print(f"Overall NPV:         {npv_cv:.4f} ({npv_cv * 100:.2f}%)")

# Internal AUC 95% CI
print(f"\n--- Internal AUC 95% CI (Bootstrap, n=2000) ---")
ci_lower_int, ci_upper_int = bootstrap_auc_ci(y.values, y_pred_proba_cv, n_bootstrap=2000)
print(f"AUC = {auc_cv_overall:.3f} (95% CI: {ci_lower_int:.3f}-{ci_upper_int:.3f})")

# ==============================================================================
# 5. Train Final Model (Full Internal Data)
# ==============================================================================

print(f"\n{'=' * 70}")
print("Final Model Training")
print("=" * 70)

X_train_zscore = X_raw.apply(zscore, axis=0)
final_model = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', C=1.0)
final_model.fit(X_train_zscore, y)

print("Final model trained!")

coefficients = pd.DataFrame({
    'gene': available_genes_int,
    'coefficient': final_model.coef_[0],
    'abs_coefficient': np.abs(final_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print(f"\nTop 10 Features:")
print(coefficients.head(10).to_string(index=False))
print(f"\nModel Intercept: {final_model.intercept_[0]:.4f}")

# ==============================================================================
# 6. External Validation (GSE39582, n=77)
# ==============================================================================

print(f"\n{'=' * 70}")
print("STEP 2: External Cohort Validation")
print("=" * 70)

external_df = pd.read_csv('data/GEO_expression_with_labels.csv')

available_genes_ext = [g for g in available_genes_int if g in external_df.columns]
print(f"Available genes: {len(available_genes_ext)}/{len(available_genes_int)}")

X_test_raw = external_df[available_genes_ext]
X_test_zscore = X_test_raw.apply(zscore, axis=0)
y_test = (external_df['label'] == 'HotHigh').astype(int)

print(f"\nExternal Cohort:")
print(f"  Samples: {len(X_test_zscore)}")
print(f"  HotHigh: {y_test.sum()} ({y_test.sum() / len(y_test) * 100:.2f}%)")
print(f"  Others: {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.2f}%)")

# Predict
y_pred_proba_test = final_model.predict_proba(X_test_zscore)[:, 1]
y_pred_test = final_model.predict(X_test_zscore)

auc_test = roc_auc_score(y_test, y_pred_proba_test)
cm_test = confusion_matrix(y_test, y_pred_test)

print(f"\n{'=' * 70}")
print("External Cohort Performance")
print("=" * 70)
print(f"AUC: {auc_test:.4f}")
print(f"\nConfusion Matrix:")
print(cm_test)

accuracy_test = (cm_test[0, 0] + cm_test[1, 1]) / cm_test.sum()
sensitivity_test = cm_test[1, 1] / (cm_test[1, 0] + cm_test[1, 1]) if (cm_test[1, 0] + cm_test[1, 1]) > 0 else 0
specificity_test = cm_test[0, 0] / (cm_test[0, 0] + cm_test[0, 1]) if (cm_test[0, 0] + cm_test[0, 1]) > 0 else 0
ppv_test = cm_test[1, 1] / (cm_test[0, 1] + cm_test[1, 1]) if (cm_test[0, 1] + cm_test[1, 1]) > 0 else 0
npv_test = cm_test[0, 0] / (cm_test[0, 0] + cm_test[1, 0]) if (cm_test[0, 0] + cm_test[1, 0]) > 0 else 0

print(f"\nAccuracy:    {accuracy_test:.4f} ({accuracy_test * 100:.2f}%)")
print(f"Sensitivity: {sensitivity_test:.4f} ({sensitivity_test * 100:.2f}%)")
print(f"Specificity: {specificity_test:.4f} ({specificity_test * 100:.2f}%)")
print(f"PPV:         {ppv_test:.4f} ({ppv_test * 100:.2f}%)")
print(f"NPV:         {npv_test:.4f} ({npv_test * 100:.2f}%)")

# External AUC 95% CI
print(f"\n--- External AUC 95% CI (Bootstrap, n=2000) ---")
ci_lower_ext, ci_upper_ext = bootstrap_auc_ci(y_test.values, y_pred_proba_test, n_bootstrap=2000)
print(f"AUC = {auc_test:.3f} (95% CI: {ci_lower_ext:.3f}-{ci_upper_ext:.3f})")

# ==============================================================================
# 7. Summary
# ==============================================================================

print(f"\n{'=' * 70}")
print("AUC 95% CI SUMMARY")
print("=" * 70)
print(f"Internal (TCGA, n={len(y)}):     AUC = {auc_cv_overall:.3f} (95% CI: {ci_lower_int:.3f}-{ci_upper_int:.3f})")
print(f"External (GSE39582, n={len(y_test)}): AUC = {auc_test:.3f} (95% CI: {ci_lower_ext:.3f}-{ci_upper_ext:.3f})")

# ==============================================================================
# 8. Visualization
# ==============================================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# A. ROC Curves
fpr_cv, tpr_cv, _ = roc_curve(y, y_pred_proba_cv)
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)

ax_a = fig.add_subplot(gs[0, :2])
ax_a.plot(fpr_cv, tpr_cv, color='#1f77b4', lw=3,
          label=f'Internal CV (AUC = {auc_cv_overall:.3f}, 95% CI: {ci_lower_int:.3f}-{ci_upper_int:.3f})')
ax_a.plot(fpr_test, tpr_test, color='#ff7f0e', lw=3,
          label=f'External (AUC = {auc_test:.3f}, 95% CI: {ci_lower_ext:.3f}-{ci_upper_ext:.3f})')
ax_a.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random', alpha=0.5)
ax_a.fill_between(fpr_cv, tpr_cv, alpha=0.2, color='#1f77b4')
ax_a.fill_between(fpr_test, tpr_test, alpha=0.2, color='#ff7f0e')
ax_a.set_xlim([0.0, 1.0])
ax_a.set_ylim([0.0, 1.05])
ax_a.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax_a.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
ax_a.set_title('A. Receiver Operating Characteristic Curves', fontsize=13, fontweight='bold', loc='left')
ax_a.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)
ax_a.grid(alpha=0.3)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# B. Performance Summary
ax_b = fig.add_subplot(gs[0, 2])
metrics_short = ['AUC', 'Acc', 'Sens', 'Spec']
internal_short = [auc_cv_overall, accuracy_cv, sensitivity_cv, specificity_cv]
external_short = [auc_test, accuracy_test, sensitivity_test, specificity_test]

x_pos = np.arange(len(metrics_short))
width = 0.35
bars1 = ax_b.bar(x_pos - width / 2, internal_short, width, label='Internal CV',
                 color='#1f77b4', alpha=0.8, edgecolor='black')
bars2 = ax_b.bar(x_pos + width / 2, external_short, width, label='External',
                 color='#ff7f0e', alpha=0.8, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                  f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax_b.set_ylabel('Score', fontsize=11, fontweight='bold')
ax_b.set_title('B. Performance Summary', fontsize=13, fontweight='bold', loc='left')
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(metrics_short, fontsize=10)
ax_b.set_ylim([0, 1.15])
ax_b.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax_b.legend(fontsize=9, loc='lower left')
ax_b.grid(axis='y', alpha=0.3)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# C. Internal CV Confusion Matrix
ax_c = fig.add_subplot(gs[1, 0])
sns.heatmap(cm_cv_overall, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Others', 'HotHigh'], yticklabels=['Others', 'HotHigh'],
            ax=ax_c, square=True, linewidths=2, linecolor='black',
            annot_kws={'size': 14, 'weight': 'bold'})
ax_c.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax_c.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax_c.set_title(f'C. Internal CV (n={len(y)})', fontsize=13, fontweight='bold', loc='left')

# D. External Confusion Matrix
ax_d = fig.add_subplot(gs[1, 1])
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', cbar=False,
            xticklabels=['Others', 'HotHigh'], yticklabels=['Others', 'HotHigh'],
            ax=ax_d, square=True, linewidths=2, linecolor='black',
            annot_kws={'size': 14, 'weight': 'bold'})
ax_d.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax_d.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax_d.set_title(f'D. External (n={len(y_test)})', fontsize=13, fontweight='bold', loc='left')

# E. Top 10 Features
ax_e = fig.add_subplot(gs[1, 2])
top10 = coefficients.head(10).sort_values('coefficient', ascending=False)
colors_top10 = ['#d62728' if coef > 0 else '#1f77b4' for coef in top10['coefficient']]
ax_e.barh(top10['gene'], top10['coefficient'], color=colors_top10, alpha=0.8, edgecolor='black')
ax_e.axvline(0, color='black', linewidth=2)
ax_e.set_xlabel('Coefficient', fontsize=11, fontweight='bold')
ax_e.set_title('E. Top 10 Features', fontsize=13, fontweight='bold', loc='left')
ax_e.grid(axis='x', alpha=0.3)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)
ax_e.invert_yaxis()

plt.suptitle('HotHigh Classifier: 5-Fold CV and External Validation',
             fontsize=15, fontweight='bold', y=0.995)

plt.savefig('results/Figure_validation.png', dpi=300, bbox_inches='tight')
plt.savefig('results/Figure_validation.tiff', dpi=300, bbox_inches='tight')
plt.savefig('results/Figure_validation.pdf', bbox_inches='tight')
print(f"\nFigure saved!")

# ==============================================================================
# 9. Save Results
# ==============================================================================

cv_df.to_csv('results/CV_fold_results.csv', index=False, float_format='%.4f')

results_summary = pd.DataFrame({
    'Cohort': ['Internal (5-Fold CV)', 'External (GSE39582)'],
    'N': [len(y), len(y_test)],
    'HotHigh_N': [y.sum(), y_test.sum()],
    'AUC': [auc_cv_overall, auc_test],
    'AUC_CI_lower': [ci_lower_int, ci_lower_ext],
    'AUC_CI_upper': [ci_upper_int, ci_upper_ext],
    'Accuracy': [accuracy_cv, accuracy_test],
    'Sensitivity': [sensitivity_cv, sensitivity_test],
    'Specificity': [specificity_cv, specificity_test],
    'PPV': [ppv_cv, ppv_test],
    'NPV': [npv_cv, npv_test]
})

results_summary.to_csv('results/model_performance.csv', index=False, float_format='%.4f')
coefficients.to_csv('results/gene_coefficients.csv', index=False, float_format='%.6f')

print(f"\n{'=' * 70}")
print("Analysis Complete!")
print("=" * 70)
print("\nSaved files:")
print("  - results/Figure_validation.png/tiff/pdf")
print("  - results/CV_fold_results.csv")
print("  - results/model_performance.csv")
print("  - results/gene_coefficients.csv")
