# Telecom Customer Churn (`ds-churn`)

End-to-end customer churn modeling and threshold analysis on the Telco Customer Churn dataset (`7,043` rows, `21` columns, binary `Churn` target) using a reproducible `scikit-learn` pipeline and `XGBoost`.

## Project Structure

```text
ds-churn/
├── data/
│   └── telecom_churn_dataset.csv
├── models/
│   ├── churn_pipeline.joblib
│   ├── churn_metrics.json
│   └── legacy/
│       ├── best_churn_model.pkl
│       ├── feature_names.pkl
│       ├── label_encoders.pkl
│       ├── scaler.pkl
│       └── README.md
├── notebooks/
│   ├── 01_eda_and_threshold.ipynb
│   └── Copy of Team_Soluzion_Project.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── .gitignore
└── README.md
