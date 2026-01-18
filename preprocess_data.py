import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================
# LOAD DATASETS
# ==============================

df1 = pd.read_csv("industrial_accident_prediction_dataset.csv")
df2 = pd.read_csv("converted_predictive_to_accident.csv")

# ==============================
# ALIGN COLUMNS
# ==============================

common_columns = list(set(df1.columns).intersection(df2.columns))
df = pd.concat([df1[common_columns], df2[common_columns]], ignore_index=True)

# ==============================
# HANDLE MISSING VALUES
# ==============================

for col in df.select_dtypes(include=["int64", "float64"]):
    df[col].fillna(df[col].median(), inplace=True)

# ==============================
# FEATURES & TARGET
# ==============================

X = df.drop("accident_risk", axis=1)
y = df["accident_risk"]

# SAVE FEATURE ORDER
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# SCALING
# ==============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# SAVE OUTPUTS
# ==============================

joblib.dump(X_train_scaled, "X_train_scaled.pkl")
joblib.dump(X_test_scaled, "X_test_scaled.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Preprocessing completed and data saved")
