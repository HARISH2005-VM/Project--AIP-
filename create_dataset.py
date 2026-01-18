import pandas as pd
import numpy as np

np.random.seed(42)
rows = 3000

# ==============================
# CREATE BASE DATA
# ==============================

df = pd.DataFrame({
    "machine_id": np.random.randint(1, 200, rows),
    "equipment_age_years": np.random.randint(1, 25, rows),
    "temperature": np.random.normal(70, 12, rows),
    "vibration": np.random.normal(0.6, 0.25, rows),
    "pressure": np.random.normal(32, 6, rows),
    "humidity": np.random.uniform(30, 90, rows),
    "downtime_hours_last_month": np.random.randint(0, 200, rows),
    "maintenance_count_last_year": np.random.randint(0, 15, rows),
    "safety_violations": np.random.randint(0, 6, rows),
    "operator_experience_years": np.random.randint(0, 20, rows)
})

# ==============================
# FIX 2 – REALISTIC RISK LOGIC
# ==============================

risk_score = (
    (df["temperature"] > 75).astype(int) +
    (df["vibration"] > 0.7).astype(int) +
    (df["equipment_age_years"] > 10).astype(int) +
    (df["safety_violations"] > 2).astype(int)
)

df["accident_risk"] = (risk_score >= 2).astype(int)

# ==============================
# SAVE DATASET
# ==============================

df.to_csv("industrial_accident_prediction_realistic.csv", index=False)

print("✅ Dataset regenerated successfully")
print(df["accident_risk"].value_counts())
