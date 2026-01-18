import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ==============================
# LOAD FINAL DATASET
# ==============================

df = pd.read_csv("final_industrial_safety_dataset.csv")

# ==============================
# FINAL FEATURE SET
# ==============================

FEATURE_COLS = [
    "machine_id",
    "equipment_age_years",
    "temperature",
    "vibration",
    "pressure",
    "humidity",
    "downtime_hours_last_month",
    "maintenance_count_last_year",
    "safety_violations",
    "operator_experience_years"
]

TARGET = "accident_risk"

X = df[FEATURE_COLS]
y = df[TARGET]

# ==============================
# TRAIN–TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# XGBOOST MODEL
# ==============================

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# SAVE MODEL
# ==============================

joblib.dump(model, "xgboost_model.pkl")
print("\n✅ Model retrained and saved successfully")
