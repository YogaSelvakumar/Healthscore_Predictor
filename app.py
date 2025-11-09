import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ----------------------------
# PAGE CONFIG & STYLING
# ----------------------------
st.set_page_config(page_title="Health Score Predictor", page_icon="💪", layout="wide")

# ---- Custom CSS for Beautiful UI ----
st.markdown("""
    <style>
        /* Main background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to right, #d9afd9 0%, #97d9e1 100%);
        }
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        /* Title style */
        .main-title {
            text-align: center;
            color: white;
            background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
            padding: 20px;
            border-radius: 10px;
            font-size: 32px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #333;
            font-size: 18px;
            margin-top: -10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Title Section ----
st.markdown('<div class="main-title">💪 Health Score Predictor</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning–based Health Assessment Tool</p>', unsafe_allow_html=True)

# ----------------------------
# SIDEBAR INFO
# ----------------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your health dataset (.csv)", type=["csv"])
st.sidebar.markdown("## ℹ️ About")
st.sidebar.info("This app predicts your **Health Score (Good/Bad)** using health parameters like BMI, cholesterol, stress level, and lifestyle habits.")

default_path = r"C:\Users\Sys\Desktop\HealthScore_prediction_app\Health_score_balanced.csv"

# ----------------------------
# DATA LOADING
# ----------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv(default_path)
        st.sidebar.success("Loaded default dataset successfully.")
    except Exception as e:
        st.sidebar.error("Upload a valid CSV file or check file path.")
        st.stop()

# Drop Gender column if exists
if "Gender" in df.columns:
    df = df.drop(columns=["Gender"])

# ----------------------------
# DATA PREVIEW SECTION
# ----------------------------
st.markdown("### 📊 Step 1: Dataset Overview")
with st.expander("🔍 View Dataset Preview"):
    st.dataframe(df.head())
    st.write("**Basic Statistics:**")
    st.write(df.describe())

# ----------------------------
# TARGET CHECK & CLEANING
# ----------------------------
if "HealthScore" not in df.columns:
    st.error("❌ 'HealthScore' column not found. Ensure correct name in dataset.")
    st.stop()

df["HealthScore"] = df["HealthScore"].str.strip().str.capitalize()
valid_labels = ["Good", "Bad"]
if not df["HealthScore"].isin(valid_labels).all():
    st.error("❌ Target column contains invalid labels (only Good/Bad allowed).")
    st.stop()

# ----------------------------
# FEATURE ENGINEERING & SPLIT
# ----------------------------
X = df.drop(columns=["HealthScore"])
y = df["HealthScore"]

X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

st.success("✅ Data cleaned, encoded, and split into training/testing sets.")

# ----------------------------
# MODEL TRAINING
# ----------------------------
st.markdown("### ⚙️ Step 2: Model Training & Evaluation")

col1, col2, col3, col4 = st.columns(4)
col1.info("📂 Data Loaded")
col2.success("🧹 Preprocessed")
col3.warning("🤖 Training Models")
col4.error("📈 Evaluating Results")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42)
}

results = {}
best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    if acc > best_acc:
        best_acc = acc
        best_model = model

res_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)

with st.expander("📈 Model Accuracy Comparison"):
    st.dataframe(res_df)

st.success(f"🏆 Best Model: **{res_df.iloc[0,0]}** (Accuracy: **{res_df.iloc[0,1]:.2f}**)")

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
from sklearn.metrics import confusion_matrix
import seaborn as sns

st.markdown("### 🧩 Step 3: Confusion Matrix of the Best Model")

# Predict test set using best model
y_pred_best = best_model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_best, labels=["Good", "Bad"])

# Plot confusion matrix heatmap
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Predicted Good", "Predicted Bad"],
            yticklabels=["Actual Good", "Actual Bad"], ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Optional: Display classification metrics
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred_best, target_names=["Good", "Bad"], output_dict=True)
report_df = pd.DataFrame(report).transpose()

with st.expander("📋 Classification Report Details"):
    st.dataframe(report_df)


# ----------------------------
# ROC CURVE
# ----------------------------
st.markdown("### 📊 Step 3: ROC Curve Comparison")
plt.figure(figsize=(7, 5))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test.map({"Bad": 0, "Good": 1}), y_prob)
        plt.plot(fpr, tpr, label=f"{name}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
st.pyplot(plt)

joblib.dump(best_model, "best_health_model.pkl")
joblib.dump(scaler, "scaler.pkl")
st.success("✅ Model trained and saved successfully!")

# ----------------------------
# PREDICTION SECTION
# ----------------------------
st.markdown("### 🚀 Step 4: Health Score Prediction")

col1, col2, col3 = st.columns(3)
age = col1.number_input("Age", min_value=1, max_value=120, value=25)
bmi = col2.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
blood_pressure = col3.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)

col4, col5, col6 = st.columns(3)
cholesterol = col4.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
physical_activity = col5.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
sleep_duration = col6.number_input("Sleep Duration (hours)", min_value=0.0, max_value=12.0, value=7.0)

col7, col8 = st.columns(2)
stress_level = col7.slider("Stress Level (1 = Low, 10 = High)", 1, 10, 5)
diet_quality = col8.selectbox("Diet Quality", ["Poor", "Average", "Good"])

input_dict = {
    "Age": age,
    "BMI": bmi,
    "BloodPressure": blood_pressure,
    "Cholesterol": cholesterol,
    "PhysicalActivity": physical_activity,
    "SleepDuration": sleep_duration,
    "StressLevel": stress_level,
    "DietQuality": diet_quality
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)
input_scaled = scaler.transform(input_df)

if st.button("🔮 Predict Health Score"):
    prediction = best_model.predict(input_scaled)[0]
    if prediction == "Good":
        st.success("🌟 Your Predicted Health Status: **GOOD HEALTH** ✅")
    else:
        st.error("⚠️ Your Predicted Health Status: **BAD HEALTH** ❌")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("""
    <hr>
    <div style='text-align:center; color:grey; font-size:14px;'>
    © 2025 Health Score Predictor | Developed by <b>Yoga</b>
    </div>
""", unsafe_allow_html=True)

