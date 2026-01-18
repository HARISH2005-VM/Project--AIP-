import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
from streamlit_autorefresh import st_autorefresh
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Industrial Safety Intelligence", layout="wide")

# ---------------- LOAD MODEL & DATA ----------------
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("final_industrial_safety_dataset.csv")

# ---------------- EMAIL CONFIG ----------------
SMTP_EMAIL = "your_email@gmail.com"
SMTP_PASSWORD = "your_app_password"
ALERT_RECEIVER = "receiver@gmail.com"

# ---------------- SESSION STATE ----------------
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ---------------- EMAIL FUNCTION ----------------
def send_email(subject, message):
    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = SMTP_EMAIL
        msg["To"] = ALERT_RECEIVER

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
    except:
        pass  # silently fail (safe for demo)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body { background-color: #0f172a; }
.card {
    background: linear-gradient(135deg, #111827, #1f2933);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
.title { font-size: 34px; font-weight: 800; color: #22d3ee; }
.subtitle { color: #cbd5e1; font-size: 15px; }
.metric { font-size: 28px; font-weight: 700; }
.low { color: #22c55e; }
.medium { color: #facc15; }
.high { color: #ef4444; }
.label { color: #94a3b8; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="card">
  <div class="title">🏭 Industrial Safety Intelligence Dashboard</div>
  <div class="subtitle">
    AI-powered accident prediction & predictive maintenance system
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------- MACHINE SEARCH ----------------
st.sidebar.header("🔍 Asset Search")
machine_id = st.sidebar.selectbox("Select Machine ID", df["machine_id"].unique())
machine = df[df["machine_id"] == machine_id].iloc[0]

# ---------------- LIVE SENSOR TOGGLE ----------------
st.sidebar.markdown("### 🔴 Live Sensor Tracking")

if "live" not in st.session_state:
    st.session_state.live = False

toggle_label = "⏹ Stop Live Tracking" if st.session_state.live else "▶️ Start Live Tracking"

if st.sidebar.button(toggle_label):
    st.session_state.live = not st.session_state.live

# ---------------- LIVE SENSOR SIMULATION ----------------
if st.session_state.live:
    st_autorefresh(interval=2000, key="live_refresh")
    temperature = machine["temperature"] + random.uniform(-2, 2)
    vibration = machine["vibration"] + random.uniform(-0.1, 0.1)
    pressure = machine["pressure"] + random.uniform(-1, 1)
else:
    temperature = machine["temperature"]
    vibration = machine["vibration"]
    pressure = machine["pressure"]

# ---------------- INPUT DATA ----------------
input_df = pd.DataFrame([{
    "machine_id": machine["machine_id"],
    "equipment_age_years": machine["equipment_age_years"],
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "humidity": machine["humidity"],
    "downtime_hours_last_month": machine["downtime_hours_last_month"],
    "maintenance_count_last_year": machine["maintenance_count_last_year"],
    "safety_violations": machine["safety_violations"],
    "operator_experience_years": machine["operator_experience_years"]
}])

# ---------------- PREDICTION ----------------
prob = model.predict_proba(input_df)[0][1]
risk_score = int(prob * 100)

if risk_score < 30:
    risk, cls = "LOW", "low"
elif risk_score < 60:
    risk, cls = "MEDIUM", "medium"
else:
    risk, cls = "HIGH", "high"

# ---------------- ALERT SYSTEM ----------------
# ---------------- ALERT SYSTEM ----------------
if st.session_state.live:
    alert_msg = None

    if temperature > 90:
        alert_msg = "High Temperature"
    elif vibration > 1.2:
        alert_msg = "High Vibration"
    elif pressure > 40:
        alert_msg = "High Pressure"
    elif risk_score > 60:
        alert_msg = "High Accident Risk"

    if alert_msg:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.error(f"🚨 {alert_msg} detected for Asset {machine_id}")

        st.session_state.alerts.append({
            "Time": timestamp,
            "Machine ID": machine_id,
            "Alert": alert_msg,
            "Risk Score": risk_score
        })

        send_email(
            "🚨 Industrial Safety Alert",
            f"{alert_msg}\nMachine ID: {machine_id}\nRisk Score: {risk_score}%\nTime: {timestamp}"
        )
    else:
        st.success("✅ All parameters are within safe operating limits")


# ---------------- ASSET SUMMARY ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
      <div class="label">Asset ID</div>
      <div class="metric">{machine_id}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
      <div class="label">Risk Level</div>
      <div class="metric {cls}">{risk}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
      <div class="label">Risk Score</div>
      <div class="metric">{risk_score}%</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- SENSOR DETAILS ----------------
st.markdown("### 🔧 Live Machine Parameters")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🌡 Temperature", f"{temperature:.2f} °C")
c2.metric("📳 Vibration", f"{vibration:.2f} mm/s")
c3.metric("🔩 Pressure", f"{pressure:.2f} bar")
c4.metric("⏱ Downtime", f"{machine['downtime_hours_last_month']} hrs")

# ---------------- VISUALIZATION (UNCHANGED) ----------------
st.markdown("### 📊 Operational Overview")

fig, ax = plt.subplots(figsize=(10,5))
features = input_df.drop(columns=["machine_id"]).iloc[0]
features.plot(kind="bar", ax=ax)
ax.set_ylabel("Value")
ax.set_title("Machine Parameter Distribution")
ax.grid(alpha=0.3)
st.pyplot(fig)

# ---------------- RISK TREND (UNCHANGED) ----------------
st.markdown("### 📈 Risk Trend vs Temperature")

temps = np.arange(30, 120, 5)
trend = []

for t in temps:
    temp_df = input_df.copy()
    temp_df["temperature"] = t
    trend.append(model.predict_proba(temp_df)[0][1])

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(temps, trend, linewidth=3)
ax2.set_xlabel("Temperature (°C)")
ax2.set_ylabel("Accident Risk")
ax2.grid(alpha=0.4)
st.pyplot(fig2)

# ---------------- ALERT HISTORY ----------------
st.markdown("### 🧾 Alert History")
if st.session_state.alerts:
    st.dataframe(pd.DataFrame(st.session_state.alerts), use_container_width=True)
else:
    st.info("No alerts triggered yet.")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="card">
  <div class="subtitle">
    🚀 Built for Industrial AI | Predictive Maintenance | Safety Intelligence
  </div>
</div>
""", unsafe_allow_html=True)
