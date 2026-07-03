# Employee Attrition Prediction

## 📌 Project Overview
This project predicts whether an employee will leave the company using machine learning techniques. It is based on the IBM HR Analytics dataset.

## 🎯 Objective
To analyze employee data and build a model that can predict employee attrition (Yes/No).

---

## 📊 Dataset
- Source: IBM HR Analytics Attrition Dataset (Kaggle)
- Rows: 1470
- Features: 35

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🔍 Workflow
1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Visualization
4. Data Preprocessing (Label Encoding)
5. Train-Test Split
6. Model Training (Random Forest)
7. Model Evaluation

---

## 🤖 Model Used
- Random Forest Classifier

---

## 📈 Results
- Accuracy: ~85–90% (varies slightly)
- Evaluated using:
  - Confusion Matrix
  - Classification Report

---

## 📌 Key Insights
- Overtime is a strong factor in attrition
- Monthly income and job role also influence employee retention
- Younger employees show higher attrition tendency

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python employee_attrition.py