import pandas as pd
import numpy as np

np.random.seed(1)
rows = 3000

data = {
    "machine_id": np.random.randint(1, 150, rows),
    "equipment_age_years": np.random.randint(1, 25, rows),
    "temperature": np.round(np.random.normal(70, 12, rows), 2),
    "vibration": np.round(np.random.normal(0.6, 0.25, rows), 3),
    "pressure": np.round(np.random.normal(32, 6, rows), 2),
    "humidity": np.round(np.random.uniform(30, 90, rows), 2),
    "downtime_hours_last_month": np.random.randint(0, 200, rows),
    "maintenance_count_last_year": np.random.randint(0, 15, rows),
    "safety_violations": np.random.randint(0, 7, rows),
    "operator_experience_years": np.random.randint(0, 20, rows),
    "accident_risk": np.random.choice([0, 1], rows, p=[0.82, 0.18])
}

df = pd.DataFrame(data)
df.to_csv("industrial_accident_prediction_extended_dataset.csv", index=False)

print("CSV file created successfully")

df = pd.read_csv("industrial_accident_prediction_extended_dataset.csv")
print(df.head())

