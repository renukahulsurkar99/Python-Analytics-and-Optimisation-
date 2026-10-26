from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import DATA_PATH, QUICK_TEST, choose_threshold, capture_plots
#TASK 06: APPLICATION OF SUITABLE MACHINE LEARNING MODEL TO PREDICT THE TARGET BREACH IN AED AMONG PATIENTS

#MODEL 01: LOGISTIC REGRESSION (SELECTED MODEL)


# -------------------------------
# Step 1: Import Libraries
# -------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
capture_plots(__file__)
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFECV
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
# Step 3: Train/Validation/Test Split (70%, 15%, 15%)
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

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols)
    ],
    remainder='passthrough' # keep other features as-is (one-hot encoded)
)

# Fit the preprocessor
preprocessor.fit(X_train)

# Transforming data based on the preprocessor (transforming both train and test data)
X_train_transformed = preprocessor.transform(X_train)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)


# -------------------------------
# Step 5: Feature Selection with RFECV()
# -------------------------------

logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=9999)


min_features_to_select = 1  # Minimum number of features to select
cv = StratifiedKFold(3 if QUICK_TEST else 33)

rfecv = RFECV(
    estimator=logreg,
    step=10 if QUICK_TEST else 1,
    cv=cv,
    scoring='roc_auc',
    min_features_to_select=min_features_to_select,
    n_jobs=1,
)
rfecv.fit(X_train_transformed, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")


feature_names_transformed = numeric_cols + dummy_cols



selected_features = np.array(feature_names_transformed)[rfecv.support_]
print("Selected features:")
print(selected_features)

data = {
    key: value
    for key, value in rfecv.cv_results_.items()
    if key in ["n_features", "mean_test_score", "std_test_score"]
}
cv_results = pd.DataFrame(data)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean cross-validation ROC-AUC")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()


# Keep only selected features
X_train_sel = X_train_transformed[:, rfecv.support_]
X_val_sel   = X_val_transformed[:, rfecv.support_]
X_test_sel  = X_test_transformed[:, rfecv.support_]



# -------------------------------
# Step 6: Grid Search
# -------------------------------

# Hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}


if QUICK_TEST:
    param_grid = {key: [values[0]] for key, values in param_grid.items()}
    if "n_estimators" in param_grid:
        param_grid["n_estimators"] = [30]

grid = GridSearchCV(
    estimator=logreg,
    param_grid=param_grid,
    cv=3 if QUICK_TEST else 5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

# Fit only on training set
grid.fit(X_train_sel, y_train)


print("Best GridSearchCV params:", grid.best_params_)
print("Mean CV score of best parameters:", grid.best_score_)


# Refit logistic regression on whole dataset
final_model = grid.best_estimator_

final_model.fit(X_train_sel, y_train)


# -------------------------------
# Step 7: Threshold Tuning on Validation Set
# -------------------------------

#For selecting the best recall for threshold tuning

val_probs = final_model.predict_proba(X_val_sel)[:, 1]

thresholds = np.linspace(0.01, 0.99, 200)
target_recall = 0.75

best_threshold, best_precision, _ = choose_threshold(
    y_val, val_probs, target_recall
)

print(f"Selected threshold: {best_threshold:.3f}")
print(f"Validation Precision at Recall ≥ {target_recall}: {best_precision:.3f}")


# Construct the validation trade-off table before plotting it.
results_df = pd.DataFrame([
    {"Min_Recall_Target": target,
     "Achieved_Precision": choose_threshold(y_val, val_probs, target)[1]}
    for target in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
])

#Plot to visualise the Precision–Recall Trade-off)

plt.plot(
    results_df["Min_Recall_Target"],
    results_df["Achieved_Precision"],
    marker="o"
)

plt.axvline(0.75, linestyle="--", label="Chosen Recall = 0.75")
plt.xlabel("Minimum Recall Requirement")
plt.ylabel("Best Achievable Precision")
plt.title("Precision–Recall Trade-off (Validation Set)")
plt.legend()
plt.show()


# -------------------------------
# Step 8: Evaluate on Test Set
# -------------------------------
test_probs = final_model.predict_proba(X_test_sel)[:, 1]
y_test_pred = (test_probs >= best_threshold).astype(int)

print("TEST SET PERFORMANCE")
print("---------------------")
print("Recall:", recall_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("F1-score:", f1_score(y_test, y_test_pred))
print("ROC-AUC:", roc_auc_score(y_test, test_probs))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# -------------------------------
# Step 9: Confusion Matrix
# -------------------------------
cm_display = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_test_pred,
    display_labels=["Non-breach", "Breach"]
)
plt.title("Confusion Matrix – Test Set")
plt.show()

# -------------------------------
# Step 10: Feature Importance
# -------------------------------
# Logistic Regression coefficients (scaled features)
# and numeric_cols + dummy_cols are known

coef_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': final_model.coef_[0],
    'Odds_Ratio': np.exp(final_model.coef_[0])
})

# Split by sign (NOT head/tail)
positive = coef_df[coef_df['Coefficient'] > 0].sort_values('Coefficient', ascending=False)
negative = coef_df[coef_df['Coefficient'] < 0].sort_values('Coefficient')

plt.figure(figsize=(8,6))
sns.barplot(x='Coefficient', y='Feature', data=positive.head(10))
plt.title("Top Positive Predictors (Increase Breach Risk)")
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(x='Coefficient', y='Feature', data=negative.head(10))
plt.title("Top Negative Predictors (Reduce Breach Risk)")
plt.show()
# Save only aggregate validation information. Patient-level scores stay local.
from project_config import save_model_summary
save_model_summary(
    Path(__file__).stem, y_test, test_probs, y_test_pred,
    best_threshold, grid.best_params_, QUICK_TEST
)
