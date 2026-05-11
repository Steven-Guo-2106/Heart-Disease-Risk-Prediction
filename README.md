# Heart-Disease-Risk-Prediction

## Overview
This project uses machine learning to predict the 10-year risk of Coronary Heart Disease (CHD) using the Framingham Heart Study dataset. The workflow includes data preprocessing, exploratory data analysis (EDA), class imbalance handling with SMOTE, model training, evaluation, and deployment through a web application.

## Models Used
The following classification models were compared:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost

## Results
The ensemble models (Random Forest and XGBoost) achieved the strongest performance for CHD risk prediction. Important predictive features included age, blood pressure, cholesterol, and glucose levels.

## Streamlit App
The Streamlit application allows users to:
- Enter patient health information
- Generate real-time CHD risk predictions
- Interactively explore model outputs
Run the app with:
```streamlit run app.py```

