# ==============================================================================
# IMvigor210_prediction.py
# Apply trained HotHigh classifier to IMvigor210 cohort
# ==============================================================================

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# ==============================================================================
# 20-Gene Signature
# ==============================================================================

GENE_SET_20 = [
    'CTSS', 'STAT1', 'CD74', 'HLA-A', 'HLA-E', 'HLA-C', 'HLA-B', 'CFB',
    'HLA-DRA', 'HLA-DRB1', 'HLA-DQA1', 'TAP1', 'SOD2', 'IDO1', 'IFITM1',
    'SERPING1', 'C1S', 'B2M', 'RNF213', 'IFI30'
]

# ==============================================================================
# 1. Train Model on Internal Cohort (TCGA)
# ==============================================================================

print("=" * 70)
print("STEP 1: Train Model on Internal Cohort")
print("=" * 70)

internal_df = pd.read_csv('data/TCGA_expression_with_labels.csv')

available_genes_int = [g for g in GENE_SET_20 if g in internal_df.columns]
print(f"Available genes: {len(available_genes_int)}/{len(GENE_SET_20)}")

# Z-score normalization within cohort
X_train_raw = internal_df[available_genes_int]
X_train_zscore = X_train_raw.apply(zscore, axis=0)

y_train = (internal_df['combined_label'] == 'HotHigh').astype(int)

print(f"\nInternal Cohort:")
print(f"  Samples: {len(X_train_zscore)}")
print(f"  HotHigh: {y_train.sum()} ({y_train.sum() / len(y_train) * 100:.2f}%)")
print(f"  Others: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)")

# ==============================================================================
# 2. Train Logistic Regression
# ==============================================================================

print(f"\n{'=' * 70}")
print("Training Logistic Regression Model...")
print("=" * 70)

model = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', C=1.0)
model.fit(X_train_zscore, y_train)

print("Model trained!")

# Coefficients
coefficients = pd.DataFrame({
    'gene': available_genes_int,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print(f"\nTop 10 Features:")
print(coefficients.head(10).to_string(index=False))

# Internal performance
y_pred_proba_train = model.predict_proba(X_train_zscore)[:, 1]
y_pred_train = model.predict(X_train_zscore)

auc_train = roc_auc_score(y_train, y_pred_proba_train)
cm_train = confusion_matrix(y_train, y_pred_train)

print(f"\n{'=' * 70}")
print("Internal Cohort Performance")
print("=" * 70)
print(f"AUC: {auc_train:.4f}")
print(f"Confusion Matrix:\n{cm_train}")

accuracy_train = (cm_train[0, 0] + cm_train[1, 1]) / cm_train.sum()
sensitivity_train = cm_train[1, 1] / (cm_train[1, 0] + cm_train[1, 1])
specificity_train = cm_train[0, 0] / (cm_train[0, 0] + cm_train[0, 1])

print(f"Accuracy:    {accuracy_train:.4f} ({accuracy_train * 100:.2f}%)")
print(f"Sensitivity: {sensitivity_train:.4f} ({sensitivity_train * 100:.2f}%)")
print(f"Specificity: {specificity_train:.4f} ({specificity_train * 100:.2f}%)")

# ==============================================================================
# 3. Predict on New Data (IMvigor210)
# ==============================================================================

print(f"\n{'=' * 70}")
print("STEP 2: Predict on IMvigor210 Cohort")
print("=" * 70)

# Load new data (gene x sample format)
new_data_raw = pd.read_csv('data/IMvigor210_expression_20genes.csv')

print(f"Original data shape: {new_data_raw.shape}")

# Transpose if needed (rows=genes -> rows=samples)
if new_data_raw.columns[0] in ['gene', 'Gene', 'gene_name', 'Gene_name']:
    new_data_df = new_data_raw.set_index(new_data_raw.columns[0]).T
else:
    new_data_df = new_data_raw.set_index(new_data_raw.columns[0]).T

print(f"After transpose: {new_data_df.shape}")
print(f"  (Samples={new_data_df.shape[0]}, Genes={new_data_df.shape[1]})")

# Store sample IDs
sample_ids = new_data_df.index

# Extract required genes
available_genes_new = [g for g in available_genes_int if g in new_data_df.columns]
print(f"Available genes in new data: {len(available_genes_new)}/{len(available_genes_int)}")

if len(available_genes_new) < len(available_genes_int):
    missing = [g for g in available_genes_int if g not in new_data_df.columns]
    print(f"Missing genes: {missing}")

# Z-score normalization within new cohort
X_new_raw = new_data_df[available_genes_new]
X_new_zscore = X_new_raw.apply(zscore, axis=0)

print(f"\nNew data samples: {len(X_new_zscore)}")

# Predict
y_pred_proba_new = model.predict_proba(X_new_zscore)[:, 1]
y_pred_new = (y_pred_proba_new >= 0.5).astype(int)
y_pred_label_new = ['HotHigh' if pred == 1 else 'Others' for pred in y_pred_new]

# Create results dataframe
results_df = pd.DataFrame({
    'sample_id': sample_ids,
    'predicted_probability': y_pred_proba_new,
    'predicted_label': y_pred_label_new,
    'predicted_binary': y_pred_new
})

# ==============================================================================
# 4. Prediction Summary
# ==============================================================================

print(f"\n{'=' * 70}")
print("Prediction Summary")
print("=" * 70)
print(f"Total samples: {len(results_df)}")
print(f"HotHigh: {(results_df['predicted_binary'] == 1).sum()} ({(results_df['predicted_binary'] == 1).sum() / len(results_df) * 100:.2f}%)")
print(f"Others: {(results_df['predicted_binary'] == 0).sum()} ({(results_df['predicted_binary'] == 0).sum() / len(results_df) * 100:.2f}%)")

print(f"\nProbability distribution:")
print(f"  Mean: {y_pred_proba_new.mean():.4f}")
print(f"  Median: {np.median(y_pred_proba_new):.4f}")
print(f"  Min: {y_pred_proba_new.min():.4f}")
print(f"  Max: {y_pred_proba_new.max():.4f}")

# Save results
results_df.to_csv('results/IMvigor210_predicted_labels.csv', index=False)
print(f"\nSaved: results/IMvigor210_predicted_labels.csv")

# First 10 samples
print(f"\nFirst 10 predictions:")
print(results_df.head(10).to_string(index=False))

# ==============================================================================
# 5. Visualization
# ==============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Probability distribution
axes[0].hist(y_pred_proba_new, bins=30, alpha=0.7, color='green', edgecolor='black')
axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
axes[0].set_xlabel('Predicted Probability', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Predicted Probability Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Label distribution
label_counts = results_df['predicted_label'].value_counts()
axes[1].bar(label_counts.index, label_counts.values, color=['steelblue', 'coral'], edgecolor='black')
axes[1].set_xlabel('Predicted Label', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title('Predicted Label Distribution', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

for i, (label, count) in enumerate(label_counts.items()):
    axes[1].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('results/IMvigor210_prediction_results.png', dpi=300)
print(f"\nSaved: results/IMvigor210_prediction_results.png")

# ==============================================================================
# 6. Performance Evaluation (if labels available)
# ==============================================================================

if 'label' in new_data_raw.columns or 'label' in new_data_df.columns:
    print(f"\n{'=' * 70}")
    print("Labels found! Evaluating performance...")
    print("=" * 70)

    if 'label' in new_data_df.columns:
        y_true = (new_data_df['label'] == 'HotHigh').astype(int)
    else:
        y_true = (new_data_raw.set_index(new_data_raw.columns[0]).T['label'] == 'HotHigh').astype(int)

    auc = roc_auc_score(y_true, y_pred_proba_new)
    cm = confusion_matrix(y_true, y_pred_new)

    print(f"AUC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)

    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0

    print(f"\nAccuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Sensitivity: {sensitivity:.4f} ({sensitivity * 100:.2f}%)")
    print(f"Specificity: {specificity:.4f} ({specificity * 100:.2f}%)")
    print(f"Precision:   {precision:.4f} ({precision * 100:.2f}%)")

print(f"\n{'=' * 70}")
print("Prediction Complete!")
print("=" * 70)
