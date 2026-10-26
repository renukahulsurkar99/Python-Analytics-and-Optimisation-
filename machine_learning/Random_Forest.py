from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import DATA_PATH, QUICK_TEST, choose_threshold, capture_plots
#TRIED & TESTED MODEL: RANDOM FOREST (MODEL B)


# -------------------------------
# Step 1: Import Libraries
# -------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
capture_plots(__file__)
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import StratifiedKFold

# -------------------------------
# Step 2: Load and preprocess data
# -------------------------------

df_sample = pd.read_csv(DATA_PATH)
print(f"Total Patients: {len(df_sample)}")

# Convert target to binary
df_sample["Breachornot"] = (
    df_sample["Breachornot"].astype(str).str.strip().str.lower()
    .map({"non-breach": 0, "breach": 1}).astype(int)
)

# Define features and target
X = df_sample.drop(columns=["ID", "Breachornot", "LoS"])
y = df_sample["Breachornot"]

numeric_cols = ["Age", "Day", "Period", "noofinvestigation", "nooftreatment", "noofpatients"]
categorical_cols = ["DayofWeek", "HRG"]

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Get the column names AFTER encoding
encoded_feature_names = X_encoded.columns.tolist()

# Identify which columns are numeric (original numeric_cols)
# and which are the new dummy columns
dummy_cols = [col for col in X_encoded.columns if col not in numeric_cols]

# -------------------------------
# Step 3: Train/Validation/Test Split (70%,15%,15%)
# -------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_encoded, y, test_size=0.30, stratify=y, random_state=9999
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=9999
)

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)

# -------------------------------
# Step 4: Preprocessing
# -------------------------------
# Note: Random Forest doesn't require feature scaling, but we'll keep it for consistency
# You can skip StandardScaler for RF if you want

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols)
    ],
    remainder='passthrough'
)

# Fit the preprocessor
preprocessor.fit(X_train)

# Transform data
X_train_transformed = preprocessor.transform(X_train)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)

# Feature names after transformation
feature_names_transformed = numeric_cols + dummy_cols

# -------------------------------
# Step 5: Random Forest with Grid Search
# -------------------------------

# Random Forest classifier with class weighting for imbalanced data
rf = RandomForestClassifier(
    random_state=9999,
    class_weight='balanced',  # Handle class imbalance
    n_jobs=-1
)

# Hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

if QUICK_TEST:
    param_grid = {key: [values[0]] for key, values in param_grid.items()}
    if "n_estimators" in param_grid:
        param_grid["n_estimators"] = [30]

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3 if QUICK_TEST else 5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=2
)

# Fit on training set
print("Starting Grid Search for Random Forest...")
grid.fit(X_train_transformed, y_train)

print("\nBest GridSearchCV params:", grid.best_params_)
print("Mean CV score of best parameters:", grid.best_score_)

# Get best model
final_model = grid.best_estimator_

# -------------------------------
# Step 6: Threshold Tuning on Validation Set
# -------------------------------

val_probs = final_model.predict_proba(X_val_transformed)[:, 1]

thresholds = np.linspace(0.01, 0.99, 200)
target_recall = 0.80

best_threshold, best_precision, _ = choose_threshold(
    y_val, val_probs, target_recall
)

print(f"\nSelected threshold: {best_threshold:.3f}")
print(f"Validation Precision at Recall ≥ {target_recall}: {best_precision:.3f}")

# -------------------------------
# Step 7: Evaluate on Test Set
# -------------------------------
test_probs = final_model.predict_proba(X_test_transformed)[:, 1]
y_test_pred = (test_probs >= best_threshold).astype(int)

print("\n" + "="*50)
print("TEST SET PERFORMANCE")
print("="*50)
print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_probs):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# -------------------------------
# Step 8: Confusion Matrix
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
cm_display = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_test_pred,
    display_labels=["Non-breach", "Breach"],
    ax=ax,
    cmap='Blues'
)
plt.title("Confusion Matrix – Test Set (Random Forest)")
plt.tight_layout()
plt.show()

# -------------------------------
# Step 9: Feature Importance (RF-specific)
# -------------------------------

# Random Forest provides feature importance based on Gini impurity/entropy
feature_importance = pd.DataFrame({
    'Feature': feature_names_transformed,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*50)
print("TOP 15 MOST IMPORTANT FEATURES")
print("="*50)
print(feature_importance.head(15).to_string(index=False))

# Visualize top features
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
sns.barplot(x='Importance', y='Feature', data=top_features, hue='Feature', legend=False, palette='viridis')
plt.title("Top 15 Features by Importance (Random Forest)")
plt.xlabel("Feature Importance (Gini Importance)")
plt.tight_layout()
plt.show()

# Plot all features (if you want to see the full distribution)
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), feature_importance['Importance'])
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("All Feature Importances")
plt.tight_layout()
plt.show()

# -------------------------------
# Step 10: Precision-Recall Curve on Validation Set
# -------------------------------

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds_pr = precision_recall_curve(y_val, val_probs)

plt.figure(figsize=(10, 6))
plt.plot(recalls, precisions, marker='.', label='Random Forest')
plt.axvline(x=target_recall, color='red', linestyle='--',
            label=f'Chosen Recall = {target_recall}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Validation Set)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# Step 11: Compare with Different Recall Thresholds
# -------------------------------

print("\n" + "="*50)
print("PERFORMANCE AT DIFFERENT RECALL TARGETS")
print("="*50)

recall_targets = [0.70, 0.75, 0.80, 0.85, 0.90]
comparison_results = []

for target in recall_targets:
    best_t, best_p, _ = choose_threshold(y_val, val_probs, target)

    # Evaluate on test set with this threshold
    y_test_pred_temp = (test_probs >= best_t).astype(int)
    test_recall = recall_score(y_test, y_test_pred_temp)
    test_precision = precision_score(y_test, y_test_pred_temp)
    test_f1 = f1_score(y_test, y_test_pred_temp)

    comparison_results.append({
        'Target_Recall': target,
        'Threshold': best_t,
        'Val_Precision': best_p,
        'Test_Recall': test_recall,
        'Test_Precision': test_precision,
        'Test_F1': test_f1
    })

comparison_df = pd.DataFrame(comparison_results)
print(comparison_df.to_string(index=False))

# -------------------------------
# Step 12: Model Comparison Summary
# -------------------------------

print("\n" + "="*50)
print("RANDOM FOREST MODEL SUMMARY")
print("="*50)
print(f"Best Parameters: {grid.best_params_}")
print(f"Training ROC-AUC: {grid.best_score_:.4f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")
print(f"Number of Trees: {final_model.n_estimators}")
print(f"Max Depth: {final_model.max_depth}")
print(f"Total Features Used: {len(feature_names_transformed)}")
print(f"Top 5 Most Important Features:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")
# Save only aggregate validation information. Patient-level scores stay local.
from project_config import save_model_summary
save_model_summary(
    Path(__file__).stem, y_test, test_probs, y_test_pred,
    best_threshold, grid.best_params_, QUICK_TEST
)
