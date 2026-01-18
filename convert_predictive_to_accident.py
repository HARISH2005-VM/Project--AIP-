import pandas as pd
import numpy as np

# ==============================
# 1️⃣ LOAD DATASETS
# ==============================

df_pred = pd.read_csv("predictive_maintenance.csv")
df_ind = pd.read_csv("industrial_accident_prediction_dataset.csv")

print("Predictive dataset loaded:", df_pred.shape)
print("Industrial dataset loaded:", df_ind.shape)

# ==============================
# 2️⃣ CREATE MACHINE ID FOR PREDICTIVE DATA
# ==============================
# Assumption: One machine produces 25 products

products_per_machine = 25
df_pred["machine_id"] = (df_pred.index // products_per_machine) + 1

# ==============================
# 3️⃣ CONVERT PREDICTIVE → ACCIDENT FORMAT
# ==============================

accident_df = pd.DataFrame()

accident_df["machine_id"] = df_pred["machine_id"]

# Average temperature
accident_df["temperature"] = (
    df_pred["Air temperature [K]"] +
    df_pred["Process temperature [K]"]
) / 2

# Vibration proxy
accident_df["vibration"] = (
    df_pred["Rotational speed [rpm]"] /
    df_pred["Rotational speed [rpm]"].max()
)

# Pressure proxy
accident_df["pressure"] = (
    df_pred["Torque [Nm]"] /
    df_pred["Torque [Nm]"].max()
)

# Equipment age (tool wear → years)
accident_df["equipment_age_years"] = df_pred["Tool wear [min]"] / 525600

# Maintenance count
accident_df["maintenance_count_last_year"] = df_pred["Failure Type"].apply(
    lambda x: 0 if x == "No Failure" else 1
)

# Safety violations (synthetic but logical)
accident_df["safety_violations"] = (
    (accident_df["temperature"] > accident_df["temperature"].mean())
    .astype(int)
)

# Downtime (hours)
accident_df["downtime_hours_last_month"] = df_pred["Tool wear [min]"] / 60

# Target
accident_df["accident_risk"] = df_pred["Target"]

# ==============================
# 4️⃣ SAVE CONVERTED DATASET
# ==============================

accident_df.to_csv("converted_predictive_to_accident.csv", index=False)

print("Converted predictive dataset saved:", accident_df.shape)

# ==============================
# 5️⃣ MERGE BOTH DATASETS
# ==============================

final_df = pd.concat([df_ind, accident_df], ignore_index=True)

final_df.to_csv("final_industrial_accident_dataset.csv", index=False)

print("Final merged dataset saved:", final_df.shape)
print(final_df.head())
