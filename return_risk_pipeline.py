"""
Return-Risk Scoring Pipeline
Builds ML models to predict order returns and saves the final artifact.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, classification_report, confusion_matrix)
from sklearn.inspection import permutation_importance

# ============================================================
# Load and prepare data
# ============================================================
df = pd.read_csv('orders_dataset.csv')

feature_cols = [
    'product_category', 'price_inr', 'discount_pct', 'payment_method',
    'customer_tenure_days', 'num_previous_orders', 'num_previous_returns',
    'delivery_distance_km', 'delivery_days', 'is_weekend_order', 'rating_given'
]
X = df[feature_cols].copy()
y = df['returned'].copy()

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train return rate: {y_train.mean():.4f}, Test return rate: {y_test.mean():.4f}")

# ============================================================
# Task 4: Preprocessing Pipeline (no leakage)
# ============================================================
numerical_features = [
    'price_inr', 'discount_pct', 'customer_tenure_days',
    'num_previous_orders', 'num_previous_returns',
    'delivery_distance_km', 'delivery_days',
    'is_weekend_order', 'rating_given'
]
categorical_features = ['product_category', 'payment_method']

# ColumnTransformer with imputation, encoding, scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

# Fit on training only, transform both
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\nPreprocessed shape: train={X_train_processed.shape}, test={X_test_processed.shape}")

# ============================================================
# Task 5: Baseline (DummyClassifier)
# ============================================================
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train_processed, y_train)
y_dummy_pred = dummy.predict(X_test_processed)

dummy_acc = accuracy_score(y_test, y_dummy_pred)
dummy_f1 = f1_score(y_test, y_dummy_pred, pos_label=1)

print(f"\n=== DummyClassifier Baseline ===")
print(f"Accuracy: {dummy_acc:.4f}")
print(f"F1 (returned=1): {dummy_f1:.4f}")

# ============================================================
# Task 5: Logistic Regression with threshold sweep
# ============================================================
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)
y_lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
y_lr_pred = lr_pipeline.predict(X_test)

lr_acc = accuracy_score(y_test, y_lr_pred)
lr_f1 = f1_score(y_test, y_lr_pred, pos_label=1)
lr_recall = recall_score(y_test, y_lr_pred, pos_label=1)
lr_precision = precision_score(y_test, y_lr_pred, pos_label=1)
lr_roc_auc = roc_auc_score(y_test, y_lr_proba)

print(f"\n=== Logistic Regression (default threshold=0.5) ===")
print(f"Accuracy: {lr_acc:.4f}")
print(f"F1: {lr_f1:.4f}")
print(f"Recall: {lr_recall:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"ROC-AUC: {lr_roc_auc:.4f}")

# Threshold sweep
thresholds = np.arange(0.1, 0.91, 0.02)
f1_scores = []
recall_scores = []
precision_scores = []

for t in thresholds:
    y_pred_t = (y_lr_proba >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_t, pos_label=1))
    recall_scores.append(recall_score(y_test, y_pred_t, pos_label=1))
    precision_scores.append(precision_score(y_test, y_pred_t, pos_label=1))

best_idx = np.argmax(f1_scores)
t_star_lr = thresholds[best_idx]
best_f1_lr = f1_scores[best_idx]

print(f"\n=== LR Threshold Sweep ===")
print(f"Best threshold: {t_star_lr:.2f}")
print(f"Best F1: {best_f1_lr:.4f}")
print(f"Recall at best threshold: {recall_scores[best_idx]:.4f}")
print(f"Precision at best threshold: {precision_scores[best_idx]:.4f}")

# Plot F1 vs threshold
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, f1_scores, 'b-', label='F1', linewidth=2)
ax.plot(thresholds, recall_scores, 'r--', label='Recall', alpha=0.7)
ax.plot(thresholds, precision_scores, 'g--', label='Precision', alpha=0.7)
ax.axvline(t_star_lr, color='k', linestyle=':', label=f'Best t={t_star_lr:.2f}')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Logistic Regression: F1/Recall/Precision vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lr_threshold_sweep.png', dpi=150)
plt.close()
print("Saved lr_threshold_sweep.png")

# ============================================================
# Task 6: Random Forest with GridSearchCV
# ============================================================
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [6, 10, None]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    rf_pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_rf_proba = best_rf.predict_proba(X_test)[:, 1]

print(f"\n=== Random Forest GridSearchCV ===")
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# Test set ROC-AUC for best
rf_roc_auc_test = roc_auc_score(y_test, y_rf_proba)
print(f"Test ROC-AUC: {rf_roc_auc_test:.4f}")

y_rf_pred = best_rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_rf_pred)
rf_f1 = f1_score(y_test, y_rf_pred, pos_label=1)
rf_recall = recall_score(y_test, y_rf_pred, pos_label=1)
rf_precision = precision_score(y_test, y_rf_pred, pos_label=1)

print(f"Test Accuracy (@0.5 threshold): {rf_acc:.4f}")
print(f"Test F1 (@0.5 threshold): {rf_f1:.4f}")

# ============================================================
# Task 6 continued: Threshold sweep on Random Forest
# ============================================================
thresholds_rf = np.arange(0.1, 0.91, 0.02)
f1_scores_rf = []
recall_scores_rf = []
precision_scores_rf = []

for t in thresholds_rf:
    y_pred_t = (y_rf_proba >= t).astype(int)
    f1_scores_rf.append(f1_score(y_test, y_pred_t, pos_label=1))
    recall_scores_rf.append(recall_score(y_test, y_pred_t, pos_label=1))
    precision_scores_rf.append(precision_score(y_test, y_pred_t, pos_label=1))

best_idx_rf = np.argmax(f1_scores_rf)
t_star_rf = thresholds_rf[best_idx_rf]
best_f1_rf = f1_scores_rf[best_idx_rf]

print(f"\n=== RF Threshold Sweep ===")
print(f"Best threshold (t*_rf): {t_star_rf:.2f}")
print(f"Best F1: {best_f1_rf:.4f}")
print(f"Recall at best threshold: {recall_scores_rf[best_idx_rf]:.4f}")
print(f"Precision at best threshold: {precision_scores_rf[best_idx_rf]:.4f}")

# Plot RF threshold sweep
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds_rf, f1_scores_rf, 'b-', label='F1', linewidth=2)
ax.plot(thresholds_rf, recall_scores_rf, 'r--', label='Recall', alpha=0.7)
ax.plot(thresholds_rf, precision_scores_rf, 'g--', label='Precision', alpha=0.7)
ax.axvline(t_star_rf, color='k', linestyle=':', label=f'Best t*={t_star_rf:.2f}')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Random Forest: F1/Recall/Precision vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_threshold_sweep.png', dpi=150)
plt.close()
print("Saved rf_threshold_sweep.png")

# ============================================================
# Task 7: Model Explanation
# ============================================================
# Get feature names from preprocessor
best_rf_preprocessor = best_rf.named_steps['preprocessor']
best_rf_classifier = best_rf.named_steps['classifier']

# Get feature names
feature_names = []
for name, transformer, cols in best_rf_preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(cols)
    elif name == 'cat':
        ohe = transformer.named_steps['onehot']
        cat_names = ohe.get_feature_names_out(cols).tolist()
        feature_names.extend(cat_names)

# Impurity-based feature importances
importances = best_rf_classifier.feature_importances_
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False)

print(f"\n=== Top 10 Impurity-Based Feature Importances ===")
print(imp_df.head(10).to_string())

top5_impurity = imp_df.head(5)['feature'].tolist()

# Permutation importance - uses original columns (pipeline level)
print(f"\n=== Permutation Importance (All Features) ===")
perm_result = permutation_importance(
    best_rf, X_test, y_test,
    scoring='roc_auc',
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_imp = pd.Series(perm_result.importances_mean, index=X_test.columns)
perm_imp = perm_imp.sort_values(ascending=False)
print(perm_imp.round(6).to_string())

# Compare top-5 impurity features with their permutation counterparts
print(f"\n=== Impurity vs Permutation Comparison (Top 5 by Impurity) ===")

# For each top-5 impurity feature, map to original column name
comparison_rows = []
for _, row in imp_df.head(5).iterrows():
    feat = row['feature']
    # Map one-hot encoded back to original column
    orig_col = None
    for col in X_test.columns:
        if feat == col or feat.startswith(col + '_'):
            orig_col = col
            break
    perm_val = perm_imp.get(orig_col, 0) if orig_col else 0
    comparison_rows.append({
        'Feature (impurity)': feat,
        'Original Column': orig_col,
        'Impurity_Importance': row['importance'],
        'Permutation_Importance': perm_val
    })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.round(6).to_string())

# Identify which top-5 lose importance under permutation
print(f"\n=== Features Losing Most Importance Under Permutation ===")
imp_ranked = imp_df.head(5)['feature'].tolist()
for _, row in comparison_df.iterrows():
    feat = row['Feature (impurity)']
    imp_val = row['Impurity_Importance']
    perm_val = row['Permutation_Importance']
    ratio = perm_val / imp_val if imp_val > 0 else 0
    print(f"  {feat}: impurity={imp_val:.6f}, perm={perm_val:.6f}, ratio={ratio:.2%}")

# ============================================================
# Task 8: Subgroup Analysis
# ============================================================
y_rf_pred_default = best_rf.predict(X_test)
rf_proba_test = best_rf.predict_proba(X_test)[:, 1]

print(f"\n=== Overall Metrics ===")
overall_recall = recall_score(y_test, y_rf_pred_default, pos_label=1)
overall_precision = precision_score(y_test, y_rf_pred_default, pos_label=1)
print(f"Overall Recall: {overall_recall:.4f}")
print(f"Overall Precision: {overall_precision:.4f}")

print(f"\n=== Recall/Precision by Product Category (RF @ 0.5 threshold) ===")
for cat in sorted(X_test['product_category'].unique()):
    mask = (X_test['product_category'] == cat).values
    cat_recall = recall_score(y_test[mask], y_rf_pred_default[mask], pos_label=1, zero_division=0)
    cat_precision = precision_score(y_test[mask], y_rf_pred_default[mask], pos_label=1, zero_division=0)
    cat_support = mask.sum()
    print(f"  {cat}: Recall={cat_recall:.4f}, Precision={cat_precision:.4f}, N={cat_support}")

print(f"\n=== Recall/Precision by Payment Method (RF @ 0.5 threshold) ===")
for pay in sorted(X_test['payment_method'].unique()):
    mask = (X_test['payment_method'] == pay).values
    pay_recall = recall_score(y_test[mask], y_rf_pred_default[mask], pos_label=1, zero_division=0)
    pay_precision = precision_score(y_test[mask], y_rf_pred_default[mask], pos_label=1, zero_division=0)
    pay_support = mask.sum()
    print(f"  {pay}: Recall={pay_recall:.4f}, Precision={pay_precision:.4f}, N={pay_support}")

# ============================================================
# Task 9: Save Final Model
# ============================================================
joblib.dump(best_rf, 'models/return_risk_model.pkl')
print(f"\nModel saved to models/return_risk_model.pkl")
print(f"Final t*_rf value: {t_star_rf:.2f}")

# Save threshold for Part 3
with open('models/threshold_rf.txt', 'w') as f:
    f.write(str(t_star_rf))
print("Saved threshold to models/threshold_rf.txt")
