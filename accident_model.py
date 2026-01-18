import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1️ LOAD BOTH DATASETS

df1 = pd.read_csv("industrial_accident_prediction_dataset.csv")
df2 = pd.read_csv("converted_predictive_to_accident.csv")

print("Dataset 1 shape:", df1.shape)
print("Dataset 2 shape:", df2.shape)


# 2️ ALIGN COMMON COLUMNS

common_columns = list(set(df1.columns).intersection(set(df2.columns)))
print("\nCommon columns:", common_columns)

df1 = df1[common_columns]
df2 = df2[common_columns]

# 3️ MERGE DATASETS


df = pd.concat([df1, df2], ignore_index=True)
print("\nMerged dataset shape:", df.shape)

# 4️ BASIC DATA VALIDATION

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove negative values where they don't make sense
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    df = df[df[col] >= 0]

print("After validation shape:", df.shape)

# 5️ HANDLE MISSING VALUES

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=["object"]):
    df[col].fillna(df[col].mode()[0], inplace=True)

# 6️ OUTLIER HANDLING (IQR METHOD)

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

print("After outlier removal:", df.shape)

# 7️ TARGET & FEATURE SPLIT

X = df.drop("accident_risk", axis=1)
y = df["accident_risk"]

import joblib
joblib.dump(X.columns.tolist(), "feature_names.pkl")


print("\nTarget distribution:")
print(y.value_counts())

# 8️ TRAIN–TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9️ FEATURE SCALING

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  FINAL CHECK


print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])
