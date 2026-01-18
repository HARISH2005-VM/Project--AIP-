import pandas as pd

# ==============================
# LOAD BOTH DATASETS
# ==============================

df1 = pd.read_csv("industrial_accident_prediction_dataset.csv")
df2 = pd.read_csv("converted_predictive_to_accident.csv")

print("Industrial dataset shape:", df1.shape)
print("Predictive dataset shape:", df2.shape)

# ==============================
# ADD MACHINE ID IF MISSING
# ==============================

if "machine_id" not in df2.columns:
    df2["machine_id"] = range(
        df1["machine_id"].max() + 1,
        df1["machine_id"].max() + 1 + len(df2)
    )

# ==============================
# REQUIRED FINAL FEATURES
# ==============================

required_features = [
    "humidity",
    "operator_experience_years"
]

# ==============================
# ADD MISSING FEATURES SAFELY
# ==============================

for col in required_features:

    if col not in df1.columns:
        df1[col] = 50 if col == "humidity" else 5

    if col not in df2.columns:
        df2[col] = 50 if col == "humidity" else 5

# ==============================
# MERGE DATASETS
# ==============================

final_df = pd.concat([df1, df2], ignore_index=True)

# ==============================
# SAVE FINAL DATASET
# ==============================

final_df.to_csv("final_industrial_safety_dataset.csv", index=False)

print("\n✅ FINAL DATASET CREATED SUCCESSFULLY")
print("Final shape:", final_df.shape)
print(final_df.head())
