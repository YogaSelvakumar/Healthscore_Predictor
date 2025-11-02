💪 Health Score Predictor

🩺 A Machine Learning web app that predicts an individual’s Health Score (Good/Bad) based on lifestyle and medical parameters.

🌟 Overview

This project demonstrates how Machine Learning can be applied to health analytics.
The app predicts whether a person’s overall health is Good or Bad based on measurable lifestyle factors such as BMI, cholesterol, stress, and sleep duration.

It includes:

Data Cleaning & Feature Engineering

Model Training & Evaluation

Model Deployment with Streamlit

⚙️ Tech Stack
Category	Tools / Libraries
Language	Python 🐍
ML Framework	Scikit-learn 🤖
Web Framework	Streamlit 🌐
Visualization	Matplotlib 📊
Others	Pandas, NumPy, Joblib
🧠 Machine Learning Models Used

Logistic Regression ✅ (Best performing model — Accuracy: ~82.5%)

Random Forest Classifier 🌲

Gradient Boosting Classifier 🚀

The Logistic Regression model was chosen as the best based on its accuracy, balanced precision-recall scores, and interpretability.

📊 Evaluation Metrics
Metric	Description
Accuracy	Overall correct predictions (~82.5%)
Precision & Recall	Measured for both Good/Bad classes
F1-Score	Harmonic mean of precision & recall
ROC Curve	Model discrimination power
Confusion Matrix	True vs Predicted class visualization
🧩 Dataset Information
Feature	Description
Age	Age of the individual
BMI	Body Mass Index
BloodPressure	Systolic BP (mmHg)
Cholesterol	Cholesterol level (mg/dL)
PhysicalActivity	Low / Moderate / High
SleepDuration	Average sleep hours per day
StressLevel	Scale (1 = Low → 10 = High)
DietQuality	Poor / Average / Good
HealthScore (Target)	Good / Bad

🧾 Dataset used: Health_score_balanced.csv
🧮 Number of samples: 200 balanced records

🚀 Run the App Locally

Clone the repository

git clone https://github.com/YogaSelvakumar/Healthscore_Predictor.git


Navigate to the folder

cd Healthscore_Predictor


Install required libraries

pip install -r requirements.txt


Launch the Streamlit web app

streamlit run app.py


✅ The app will open in your browser (default: http://localhost:8501)

📁 Project Structure
Healthscore_Predictor/
│
├── app.py                     # Streamlit web application
├── Health_score_balanced.csv  # Dataset
├── DS project presentation.pptx # Optional presentation
└── README.md                  # Project documentation

📈 Model Results Snapshot

Accuracy: 82.5%

True Positives (Good Health): 17

True Negatives (Bad Health): 16

Confusion Matrix & ROC Curve displayed in Streamlit dashboard

👩‍💻 About the Developer

Yoga Selvakumar
🎓 M.Sc. Biochemistry | Aspiring Data Analyst & AI Enthusiast
💡 Passionate about applying Data Science in Healthcare and Biomedical Research.
🔗 LinkedIn Profile
https://www.linkedin.com/in/yoga-selvakumar

🏁 Future Enhancements

Add more health features (heart rate, glucose levels, etc.)

Deploy app online (e.g., Streamlit Cloud / Hugging Face Spaces)

Integrate model explainability (SHAP / LIME)

⭐ If you liked this project, consider giving it a star on GitHub! 🌟
