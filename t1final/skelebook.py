#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===============================================
# Clean → Encode → Classify (KNN) — Simplified
# ===============================================
# Dataset: Kaggle "Mushroom Classification"
# Expected file: ./mushrooms.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer  # used ONCE as an example
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

RANDOM_STATE = 67 #use a better random state (42 is redundant)
np.random.seed(RANDOM_STATE)


# In[2]:


# -----------------------------
# 0) Load data
# -----------------------------
csv_path = "./mushrooms.csv"  # local dataset path
#decide dataset path
if not os.path.exists(csv_path):  #fallback path when running on Kaggle
    csv_path = "/kaggle/input/mushroom-classification/mushrooms.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path} — download from Kaggle and try again.")

df = pd.read_csv(csv_path)
print("Initial shape:", df.shape)
#preview raw dataframe
df.head()


# In[4]:


# -----------------------------
# 1) Basic cleaning (duplicates, empty rows)
# -----------------------------
before = df.shape[0]

#dedup dataset
df = df.drop_duplicates()

print("Removed duplicates:", before - df.shape[0])


before = df.shape[0]
#clean missing markers
df = df.replace("?", np.nan).dropna(how="all")

print("Dropped fully empty rows:", before - df.shape[0])

#show missing summary
print("\nMissing values per column (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))



# In[5]:


# -----------------------------
# 2) Target & split
# -----------------------------
TARGET_COL = "class"  # mushroom edibility label column
if TARGET_COL not in df.columns:
    raise ValueError("Update TARGET_COL to match your dataset's label column.")

#split features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("\nTrain/Test shapes:", X_train.shape, X_test.shape)

# Detect types BEFORE imputation/encoding
#track original types
orig_numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
orig_categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
print("Original numeric cols:", orig_numeric_cols if orig_numeric_cols else "None")
print("Original categorical cols (sample):", orig_categorical_cols[:10])


# In[6]:


# --------------------------------------------
# 3) IMPUTATION (MANUAL) + one sklearn example
# --------------------------------------------
# Manual imputation rules:
#  - Numeric: fill with TRAIN median
#  - Categorical: fill with TRAIN mode
# NOTE: Compute imputation values on TRAIN only, then apply to both TRAIN & TEST.

X_train_imp = X_train.copy()
X_test_imp  = X_test.copy()

# a) Numeric → manual median
#impute numeric median
for col in orig_numeric_cols:
    median_val = X_train_imp[col].median()
    X_train_imp[col] = X_train_imp[col].fillna(median_val)
    X_test_imp[col]  = X_test_imp[col].fillna(median_val)

# b) Categorical → manual mode
#impute categorical mode
for col in orig_categorical_cols:
    mode_series = X_train_imp[col].mode(dropna=True)
    fill_value = mode_series.iloc[0] if not mode_series.empty else ""
    X_train_imp[col] = X_train_imp[col].fillna(fill_value)
    X_test_imp[col]  = X_test_imp[col].fillna(fill_value)

# c) One EXAMPLE using sklearn SimpleImputer on a SINGLE column (categorical)
#sklearn imputer example
IMPUTE_EXAMPLE_COL = "odor"  #changeable
if IMPUTE_EXAMPLE_COL in X_train_imp.columns:
    cat_imp = SimpleImputer(strategy="most_frequent")

    # SimpleImputer returns a 2D array -> flatten with [:, 0]
    X_train_imp[IMPUTE_EXAMPLE_COL] = cat_imp.fit_transform(
        X_train_imp[[IMPUTE_EXAMPLE_COL]]
    )[:, 0]
    X_test_imp[IMPUTE_EXAMPLE_COL] = cat_imp.transform(
        X_test_imp[[IMPUTE_EXAMPLE_COL]]
    )[:, 0]

    print(f"\nUsed sklearn SimpleImputer on column: '{IMPUTE_EXAMPLE_COL}'")
else:
    print(f"\nExample imputer column '{IMPUTE_EXAMPLE_COL}' not found. Skipping sklearn example.")


# In[7]:


# --------------------------------------------
# 4) ENCODING
#    - EXPLICIT One-Hot Encoding on ONE chosen column
#    - Then get_dummies for the remaining categoricals
# --------------------------------------------
# Choose the column for explicit OHE:
EXPLICIT_OHE_COL = "odor"  # explicit one-hot example column

X_train_enc = X_train_imp.copy()
X_test_enc  = X_test_imp.copy()

#manual ohe example
if EXPLICIT_OHE_COL in X_train_enc.columns:
    # EXPLICIT OHE on this ONE column — fit on TRAIN only and TEST only separately (2 lines of code)
    train_ohe = pd.get_dummies(X_train_enc[[EXPLICIT_OHE_COL]], drop_first=False)
    test_ohe = pd.get_dummies(X_test_enc[[EXPLICIT_OHE_COL]], drop_first=False)
    # Align test columns to train (avoid unseen-category issues)

    test_ohe = test_ohe.reindex(columns=train_ohe.columns, fill_value=0)

    # Drop original column and concat OHE columns
    X_train_enc = pd.concat([X_train_enc.drop(columns=[EXPLICIT_OHE_COL]), train_ohe], axis=1)
    X_test_enc  = pd.concat([X_test_enc.drop(columns=[EXPLICIT_OHE_COL]),  test_ohe], axis=1)

    print(f"\nExplicit OHE applied to column: {EXPLICIT_OHE_COL}")
else:
    print(f"\n[Note] Explicit OHE column '{EXPLICIT_OHE_COL}' not found. Skipping explicit OHE step.")

#get remaining categories
remaining_cat_cols = X_train_enc.select_dtypes(exclude=[np.number]).columns.tolist()


#apply get dummies
X_train_enc, X_test_enc = [pd.get_dummies(df, columns=remaining_cat_cols, drop_first=False) for df in (X_train_enc, X_test_enc)]

# Align test columns to train columns (VERY IMPORTANT)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

print("Encoded train shape:", X_train_enc.shape)
print("Encoded test shape:", X_test_enc.shape)


# In[8]:


#inspect encoded train frame
X_train_enc.head()


# In[9]:


# --------------------------------------------
# 5) NORMALIZE numeric columns (if any)
#    (Scale ONLY the original numeric columns; don't scale one-hot columns)
# --------------------------------------------
scaler = StandardScaler()

numeric_cols_to_scale = [c for c in orig_numeric_cols if c in X_train_enc.columns]

#scale original numeric columns
if numeric_cols_to_scale:
    scaler.fit(X_train_enc[numeric_cols_to_scale])
    X_train_enc[numeric_cols_to_scale] = scaler.transform(X_train_enc[numeric_cols_to_scale])
    X_test_enc[numeric_cols_to_scale] = scaler.transform(X_test_enc[numeric_cols_to_scale])
    print("Scaled numeric columns:", numeric_cols_to_scale)
else:
    print("Scaled numeric columns: None")


# In[10]:


# --------------------------------------------
# 6) KNN + Grid Search
# --------------------------------------------
#setup knn grid search
knn = KNeighborsClassifier()
param_grid = {"n_neighbors": [3, 5, 7, 9, 11, 15, 21], "weights": ["uniform", "distance"], "p": [1, 2]}
grid = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_enc, y_train)
best_knn = grid.best_estimator_
print("Best estimator:", best_knn)
print("Best params:", grid.best_params_)


# In[11]:


# --------------------------------------------
# 7) Evaluate on held-out TEST
# --------------------------------------------
#predict on test set
y_pred = best_knn.predict(X_test_enc)


test_acc = accuracy_score(y_test, y_pred)
print("\nTest accuracy: {:.4f}".format(test_acc))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
plt.figure()
disp.plot(values_format="d")
plt.title("Confusion Matrix (Test)")
plt.tight_layout()
plt.show()


# In[12]:


# --------------------------------------------
# 8) Validation curve (CV accuracy vs k)
# --------------------------------------------
cvres = pd.DataFrame(grid.cv_results_)
plotdf = cvres[["param_n_neighbors", "param_weights", "param_p", "mean_test_score"]].rename(
    columns={"param_n_neighbors":"k", "param_weights":"weights", "param_p":"p"}
)

plt.figure()
for (w, pval), sub in plotdf.groupby(["weights", "p"]):
    sub = sub.sort_values("k")
    plt.plot(sub["k"], sub["mean_test_score"], marker="o", label=f"weights={w}, p={pval}")
plt.xlabel("k (n_neighbors)")
plt.ylabel("Mean CV Accuracy")
plt.title("CV Accuracy vs k (by weights & p)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[13]:


# --------------------------------------------
# 9) ROC curve (binary; micro-average if multiclass)
# --------------------------------------------
if hasattr(best_knn, "predict_proba"):
    classes_ = np.unique(y_train)
    y_test_enc_int = pd.Categorical(y_test, categories=classes_).codes
    proba = best_knn.predict_proba(X_test_enc)

    if len(classes_) == 2:
        fpr, tpr, _ = roc_curve(y_test_enc_int, proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Test)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    else:
        y_bin = label_binarize(y_test_enc_int, classes=range(len(classes_)))
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"Micro-average AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Micro-average)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()


# In[14]:


# --------------------------------------------
# 10) Print summary of features
# Sample Output: Final feature count: 25
# Best KNN: KNeighborsClassifier(n_neighbors=7, p=2, weights='distance')
# --------------------------------------------
print(f"Final feature count: {X_train_enc.shape[1]}")
print(f"Best KNN: {best_knn}")


# In[ ]:




