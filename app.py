# app.py

# IMPORTS
# app
import streamlit as st

# data analysis
import numpy as np
import pandas as pd

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import clone

# evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_auc_score, roc_curve


# config
st.set_page_config(
    page_title="CHD Risk Prediction",
    layout="wide"
)

# model config
MODELS = {
    "Logistic Regression": LogisticRegression(
        penalty="l2",
        C=0.1,
        class_weight="balanced",
        solver="liblinear",
        random_state=2026
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=1),
    "Decision Tree": DecisionTreeClassifier(
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=2026
    ),
    "Support Vector Machine": SVC(
        kernel="rbf",
        C=10,
        gamma=1,
        probability=True,
        random_state=2026
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=2026
    ),
    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=1,
        random_state=2026
    )
}

# data loading and preprocessing
@st.cache_data
def load_data():
    """Load and preprocess the Framingham dataset."""
    df = pd.read_csv("data/framingham_kaggle.csv")
    df.dropna(inplace=True)
    return df

@st.cache_data
def preprocess_data(df):
    """Preprocess data with SMOTE and scaling."""
    X = df.drop(columns=['TenYearCHD', 'currentSmoker'])
    y = df["TenYearCHD"]

    over = SMOTE(sampling_strategy=0.8, random_state=2026)
    under = RandomUnderSampler(sampling_strategy=0.8, random_state=2026)
    pipeline = Pipeline([("over", over), ("under", under)])

    X_resampled, y_resampled = pipeline.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=2026
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, X.columns.tolist()

# MODEL TRAINING AND EVALUATION
@st.cache_resource
def train_model(model_name, X_train, y_train):
    """Train and cache a model by name."""
    model = clone(MODELS[model_name])
    model.fit(X_train, y_train)
    return model

def evaluate_model(X_train, X_test, y_train, y_test, model_name):
    """Evaluate model performance."""
    trained_model = train_model(model_name, X_train, y_train)

    y_pred = trained_model.predict(X_test)
    y_prob = trained_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    roc_fig, fpr, tpr = plot_roc_curve(y_test, y_prob, model_name)

    return {
        "model": trained_model,
        "pred": y_pred,
        "prob": y_prob,
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
        "roc_fig": roc_fig,
        "fpr": fpr,
        "tpr": tpr
    }

def plot_roc_curve(y_true, y_prob, model_name):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.plot(fpr, tpr, marker=".", label=f"AUC = {auc:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} ROC Curve")
    ax.legend()
    plt.close(fig)
    return fig, fpr, tpr

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted: 0", "Predicted: 1"],
                yticklabels=["Actual: 0", "Actual: 1"])
    ax.set_title("Confusion Matrix")
    plt.close(fig)
    return fig

# main app
df = load_data()
X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To",
    ["Overview", "Dataset", "Model Performance", "Model Comparison", "Prediction"]
)

# overview page
if page == "Overview":
    st.title("Predicting 10-Year Risk of Heart Disease")
    st.markdown("""
    This project applies machine learning models to predict
    the 10-year risk of coronary heart disease using the
    Framingham Heart Study dataset.
    """)

    st.subheader("Models Used")
    st.write("""
    - Logistic Regression
    - K-Nearest Neighbors
    - Decision Tree
    - Support Vector Machine
    - Random Forest
    - XGBoost
    """)

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# dataset page
elif page == "Dataset":
    st.title("Dataset Overview")

    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Variable Overview")

    variable_info = pd.DataFrame({
    "Variable": [
        "male",
        "age",
        "education",
        "currentSmoker",
        "cigsPerDay",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose",
        "TenYearCHD"
    ],
    "Description": [
        "Gender of the participant",
        "Age in years",
        "Education level",
        "Participant is a current smoker",
        "Average number of cigarettes smoked per day",
        "Participant is taking blood pressure medication",
        "History of stroke",
        "History of hypertension",
        "Participant has diabetes",
        "Total cholesterol level",
        "Systolic blood pressure",
        "Diastolic blood pressure",
        "Body Mass Index",
        "Resting heart rate",
        "Blood glucose level",
        "10-year risk of coronary heart disease"
    ],
    "Type": [
        "Binary",
        "Numeric",
        "Categorical",
        "Binary",
        "Numeric",
        "Binary",
        "Binary",
        "Binary",
        "Binary",
        "Numeric",
        "Numeric",
        "Numeric",
        "Numeric",
        "Numeric",
        "Numeric",
        "Target Variable"
        ]
    })

    st.dataframe(
    variable_info,
    use_container_width=True
    )

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

# model performance page
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("Evaluate individual models on the test set.")

    selected_model_name = st.selectbox("Select a model", list(MODELS.keys()))

    with st.spinner("Evaluating model..."):
        results = evaluate_model(
            X_train, X_test, y_train, y_test, selected_model_name
        )

    st.header(selected_model_name)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{results['accuracy']:.3f}")
    col2.metric("F1 Score", f"{results['f1']:.3f}")
    col3.metric("AUC", f"{results['auc']:.3f}")

    st.subheader("Confusion Matrix")
    st.pyplot(plot_confusion_matrix(y_test, results['pred']))

    st.subheader("Classification Report")
    report = classification_report(y_test, results['pred'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.index = ['Not at Risk', 'At Risk', 'Accuracy', 'Macro Avg', 'Weighted Avg']
    st.dataframe(report_df.round(3))

    st.subheader("ROC Curve")
    st.pyplot(results['roc_fig'])

# model comparison page
elif page == "Model Comparison":    
    st.title("Model Comparison")

    with st.spinner("Evaluating all models..."):
        results = {}
        for name in MODELS.keys():
            results[name] = evaluate_model(
                X_train, X_test, y_train, y_test, name
            )

    comparison_data = []

    for name, res in results.items():
        report = classification_report(
            y_test,
            res["pred"],
            output_dict=True,
            zero_division=0
        )

        comparison_data.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "F1 Score": report["1"]["f1-score"],
            "Precision": report["1"]["precision"],
            "Recall": report["1"]["recall"],
            "AUC": res["auc"]
        })

    comparison_df = pd.DataFrame(comparison_data)

    st.subheader("Performance Summary")
    st.dataframe(comparison_df.round(3), use_container_width=True)

    st.subheader("Combined ROC-AUC Curves")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2,
        label="Random Classifier"
    )
    for name, res in results.items():
        ax.plot(
            res["fpr"],
            res["tpr"],
            linewidth=2,
            label=f"{name} (AUC = {res['auc']:.3f})"
        )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Combined ROC-AUC Curves")
    ax.legend(loc="lower right")
    
    st.pyplot(fig)

# prediction page
elif page == "Prediction":
    st.title("Individual CHD Risk Prediction")

    st.subheader("Enter Patient Information")

    trained_pred_model = train_model(
        "Random Forest",
        X_train,
        y_train
    )

    with st.form("prediction_form"):
        st.subheader("Patient Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")

            sex = st.selectbox("Sex", ["Male", "Female"])

            age = st.slider("Age", 30, 80, 50)

            education = st.selectbox(
                "Education Level",
                [1, 2, 3, 4],
                index=1,
                help=(
                    "1 = Some high school, 2 = High school/GED, "
                    "3 = Some college/vocational, 4 = College"
                )
            )

        with col2:
            st.markdown("**Medical History**")

            BPMeds = st.selectbox(
                "Blood Pressure Medication",
                ["No", "Yes"]
            )

            prevalentStroke = st.selectbox(
                "History of Stroke",
                ["No", "Yes"]
            )

            prevalentHyp = st.selectbox(
                "Hypertension",
                ["No", "Yes"]
            )

            diabetes = st.selectbox(
                "Diabetes",
                ["No", "Yes"]
            )

        with col3:
            st.markdown("**Lifestyle & Measurements**")

            cigsPerDay = st.slider(
                "Cigarettes Per Day",
                0,
                70,
                0
            )

            totChol = st.slider(
                "Total Cholesterol",
                100,
                700,
                200
            )

            heartRate = st.slider(
                "Heart Rate",
                40,
                150,
                70
            )

        st.markdown("**Blood Pressure & Body Metrics**")

        bp_col1, bp_col2, bp_col3, bp_col4 = st.columns(4)

        with bp_col1:
            sysBP = st.slider(
                "Systolic BP",
                80,
                300,
                120
            )

        with bp_col2:
            diaBP = st.slider(
                "Diastolic BP",
                40,
                200,
                80
            )

        with bp_col3:
            BMI = st.slider(
                "BMI",
                15.0,
                60.0,
                25.0
            )

        with bp_col4:
            glucose = st.slider(
                "Glucose",
                40,
                400,
                80
            )

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_data = pd.DataFrame({
            "male": [1 if sex == "Male" else 0],
            "age": [age],
            "education": [education],
            "cigsPerDay": [cigsPerDay],
            "BPMeds": [1 if BPMeds == "Yes" else 0],
            "prevalentStroke": [1 if prevalentStroke == "Yes" else 0],
            "prevalentHyp": [1 if prevalentHyp == "Yes" else 0],
            "diabetes": [1 if diabetes == "Yes" else 0],
            "totChol": [totChol],
            "sysBP": [sysBP],
            "diaBP": [diaBP],
            "BMI": [BMI],
            "heartRate": [heartRate],
            "glucose": [glucose]
        })

        input_data = input_data[feature_names]
        input_scaled = scaler.transform(input_data)

        risk_prob = trained_pred_model.predict_proba(input_scaled)[0, 1]
        risk_pred = trained_pred_model.predict(input_scaled)[0]

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.metric(
                "Predicted Probability",
                f"{risk_prob:.2%}"
            )

        with result_col2:
            if risk_pred == 1:
                st.error("High Risk of 10-Year CHD")
            else:
                st.success("Lower Risk of 10-Year CHD")