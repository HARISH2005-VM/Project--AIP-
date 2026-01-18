import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# ==============================
# 1️⃣ LOAD MODEL & TEST DATA
# ==============================

model = joblib.load("xgboost_accident_model.pkl")
X_test = joblib.load("X_test_scaled.pkl")
y_test = joblib.load("y_test.pkl")

print("Model and test data loaded successfully")

# ==============================
# 2️⃣ PREDICTIONS
# ==============================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==============================
# 3️⃣ NUMERICAL METRICS
# ==============================

print("\n📊 MODEL PERFORMANCE METRICS")
print("--------------------------------")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1-Score :", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 4))

# ==============================
# 4️⃣ CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix – Accident Risk Prediction")
plt.show()

# ==============================
# 5️⃣ ROC CURVE
# ==============================

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.show()

# ==============================
# 6️⃣ PRECISION–RECALL CURVE
# ==============================

PrecisionRecallDisplay.from_predictions(y_test, y_prob)
plt.title("Precision–Recall Curve")
plt.show()
