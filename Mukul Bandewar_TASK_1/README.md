# 💳 CreditScore AI – Creditworthiness Prediction

CreditScore AI is a machine learning project that predicts an individual's creditworthiness using financial information such as annual income, existing debts, payment history, credit utilization, and employment experience.

The project uses classification algorithms to analyze financial features and predict whether a person belongs to the **Creditworthy (1)** or **Not Creditworthy (0)** category. It includes an interactive Streamlit web application designed to work on both desktop and mobile devices.

## 🚀 Features

- **Creditworthiness Prediction:** Predicts creditworthiness using a trained machine learning model.
- **Multiple ML Algorithms:** Compares Logistic Regression and Random Forest.
- **Feature Engineering:** Uses relevant financial attributes while excluding customer IDs from training.
- **Model Evaluation:** Measures Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- **Confusion Matrix:** Visualizes actual and predicted classifications.
- **Interactive Dashboard:** Provides a simple interface for entering financial details.
- **Responsive Design:** Supports mobile phones, tablets, and desktop browsers.
- **Dataset Preview:** Displays the uploaded CSV dataset inside the application.
- **CSV Download:** Allows users to download the dataset.
- **Model Persistence:** Saves the trained model for future predictions.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## 📂 Project Structure

```text
CreditScore-AI/
│
├── app.py
├── train_model.py
├── credit_data.csv
├── requirements.txt
├── README.md
│
└── models/
    ├── credit_model.joblib
    └── metrics.json
```

## 📊 Dataset Description

The project uses `credit_data.csv`, which contains financial information and a target column for creditworthiness classification.

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `annual_income_usd` | Annual income in USD |
| `existing_debts_usd` | Existing debts in USD |
| `payment_history_score` | Payment history score |
| `credit_utilization_ratio` | Credit utilization ratio between 0 and 1 |
| `employment_years` | Years of employment |
| `creditworthy` | Target variable: 1 or 0 |

### Target Variable

- `1` – Creditworthy
- `0` – Not Creditworthy

The `customer_id` column is excluded from model training because it is an identifier rather than a meaningful financial feature.

## 🤖 Machine Learning Algorithms

### 1. Logistic Regression

Logistic Regression is a classification algorithm that estimates the probability of a record belonging to a particular class. It provides a useful baseline for binary creditworthiness prediction.

### 2. Random Forest Classifier

Random Forest combines multiple decision trees to produce a classification. It can capture nonlinear relationships between financial features.

### Model Selection

Both algorithms are trained and evaluated using a train-test split. The model with the higher F1-score on the test set is saved for use by the Streamlit application.

## 📈 Model Evaluation Metrics

The project evaluates the trained classifiers using the following metrics:

- **Accuracy:** Proportion of correctly classified records.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model ranks positive cases above negative cases.
- **Confusion Matrix:** Shows correct and incorrect predictions for both classes.

The application displays the evaluation results generated during model training. Actual performance depends on the dataset and the train-test split.

## 💻 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CreditScore-AI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment (Optional)

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Machine Learning Model

```bash
python train_model.py
```

This command trains both classification algorithms, calculates the evaluation metrics, selects the model with the higher F1-score, and saves the trained model and metrics in the `models/` directory.

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## 🌐 Deployment

The application can be deployed using Streamlit Community Cloud.

1. Upload the project files to a GitHub repository.
2. Ensure `app.py`, `credit_data.csv`, and `requirements.txt` are included.
3. Train the model and include the generated `models/credit_model.joblib` and `models/metrics.json` files.
4. Visit https://share.streamlit.io/.
5. Connect your GitHub account and select the repository.
6. Set `app.py` as the main application file.
7. Deploy and test the app on desktop and mobile browsers.

**Note:** GitHub Pages does not directly run Python-based Streamlit applications. Use a compatible Python hosting service instead.

## 🔒 Data Privacy

- Do not upload confidential customer financial information to a public repository.
- Use appropriately licensed datasets.
- Avoid collecting unnecessary personal information.
- Apply suitable security and access controls when hosting applications that process financial data.

## ⚠️ Limitations and Disclaimer

This project is developed for educational and demonstration purposes. Predictions depend on the quality and representativeness of the training data.

A small dataset may produce unstable evaluation results. High test-set metrics do not guarantee good performance on new customers or real-world financial data.

The model does not calculate an official credit score, guarantee loan approval, or replace a professional credit assessment. It should not be used as the sole basis for actual lending decisions without appropriate validation, fairness testing, explainability, and human oversight.

## 🔮 Future Improvements

- Add more financial features, such as debt-to-income ratio and missed payments.
- Use a larger, representative, and properly labeled dataset.
- Add feature importance and prediction explanations.
- Perform cross-validation and hyperparameter tuning.
- Evaluate fairness and model calibration.
- Add prediction history and batch CSV predictions.
- Improve accessibility and responsive dashboard design.

## 👨‍💻 Project Information

**Project Name:** CreditScore AI  
**Domain:** Machine Learning / FinTech  
**Application:** Streamlit Web App  
**Task:** Credit Scoring Model

---

*Built with Python and Machine Learning to demonstrate creditworthiness classification through an interactive web application.*
