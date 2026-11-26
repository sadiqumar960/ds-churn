import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def load_and_clean_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Convert TotalCharges from string (with blanks) to numeric
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    if "churn" not in df.columns:
        raise ValueError("Expected 'churn' target column in dataset.")

    y = df["churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected values found in 'churn' column.")

    drop_cols = [c for c in ["customerid", "churn"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


def build_pipeline(X: pd.DataFrame, scale_pos_weight: float) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "telecom_churn_dataset.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_and_clean_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    pipeline = build_pipeline(X_train, scale_pos_weight=scale_pos_weight)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
        "scale_pos_weight": round(float(scale_pos_weight), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    print("Evaluation Metrics (Test Set):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    pipeline_path = models_dir / "churn_pipeline.joblib"
    metrics_path = models_dir / "churn_metrics.json"

    joblib.dump(pipeline, pipeline_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved pipeline to: {pipeline_path}")
    print(f"Saved metrics to:  {metrics_path}")


if __name__ == "__main__":
    main()
