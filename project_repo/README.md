# ⚽ Football Match Outcome Prediction

End-to-end **Machine Learning project** for predicting football match outcomes using historical match data.

The system integrates **data engineering, exploratory analysis, machine learning modeling, and an interactive Streamlit application** for real-time predictions.

---

# Project Architecture

The project follows a modular **data science workflow**:

```
Data Ingestion
      ↓
Data Cleaning & Preprocessing
      ↓
Exploratory Data Analysis (EDA)
      ↓
Feature Engineering
      ↓
Machine Learning Models
      ↓
Model Evaluation
      ↓
Interactive Prediction App (Streamlit)
```

Project structure:

```
project_repo
│
├── notebooks
│   ├── 01_ingestion.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_export_for_app.ipynb
│
├── models
│   ├── model.joblib
│   ├── model_xgb.joblib
│   ├── ftr_model_logreg.joblib
│   └── label_encoder.joblib
│
├── app
│   └── app.py
│
├── data
│   ├── raw
│   ├── interim
│   └── gold
│
├── reports
│   └── figures
│
├── requirements.txt
└── README.md
```

---

# Dataset

The dataset consists of **historical football match results** collected from multiple seasons.

Sources:

* https://football-data.co.uk
* FiveThirtyEight football statistics

Typical features used in the model:

* Division (league)
* Home team
* Away team
* Season
* Year
* Month

Target variable:

```
H → Home Win
D → Draw
A → Away Win
```

---

# Machine Learning Pipeline

The project evaluates multiple classification algorithms:

* Logistic Regression
* Random Forest
* XGBoost

Key steps:

1️⃣ Data cleaning and feature selection
2️⃣ Encoding categorical variables
3️⃣ Training multiple classifiers
4️⃣ Cross-validation
5️⃣ Hyperparameter tuning

The final model is exported using **joblib** for deployment in the Streamlit application.

---

# Model Evaluation

Evaluation metrics:

* **F1-macro**
* Accuracy
* Confusion matrix
* Cross-validation performance

The main challenge in football prediction is the **draw class**, which tends to be harder to predict accurately.

---

# Streamlit Web Application

The project includes an **interactive Streamlit application** that allows users to input match information and obtain predictions in real time.

Features:

* Match outcome prediction
* Probability distribution for all outcomes
* Interactive UI
* Clean and user-friendly interface

Run locally:

```
pip install -r requirements.txt
streamlit run app/app.py
```

---

# Example Prediction

Example output from the Streamlit application:

```
Predicted Outcome: Home Win

Prediction Probabilities
Home Win  → 45%
Away Win  → 27%
Draw      → 28%
```

---

# Technologies Used

Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib / Seaborn
Streamlit

---

# Author

Mahdieh Fakhar Shahreza
Data Science Bootcamp Final Project

---

# License

This project is intended for educational and research purposes.
