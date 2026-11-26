import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    work = df.copy()
    work.columns = [c.strip().lower() for c in work.columns]

    customer_ids = work["customerid"] if "customerid" in work.columns else None

    if "totalcharges" in work.columns:
        work["totalcharges"] = pd.to_numeric(work["totalcharges"], errors="coerce")

    drop_cols = [c for c in ["customerid", "churn"] if c in work.columns]
    X = work.drop(columns=drop_cols)
    return X, customer_ids


def print_top_features(pipeline, top_n: int = 10) -> None:
    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")

    if preprocessor is None or model is None or not hasattr(model, "feature_importances_"):
        return

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    print(f"\nTop {min(top_n, len(order))} Feature Importances (XGBoost Gain/Split):")
    for idx in order:
        clean_name = str(feature_names[idx]).replace("num__", "").replace("cat__", "")
        print(f"  {clean_name:<35} {importances[idx]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customer churn probabilities.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to input CSV with customer features. If omitted, samples from data/telecom_churn_dataset.csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifying Churn (default: 0.5).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=10,
        help="Number of rows to display when --input is omitted (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path to save scored predictions.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=10,
        help="Number of top features to display (0 to hide).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    pipeline_path = project_root / "models" / "churn_pipeline.joblib"
    default_data_path = project_root / "data" / "telecom_churn_dataset.csv"

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Trained pipeline not found at {pipeline_path}. Run `python src/train.py` first."
        )

    pipeline = joblib.load(pipeline_path)

    if args.input is not None:
        raw_df = pd.read_csv(args.input)
    else:
        raw_df = pd.read_csv(default_data_path).sample(n=args.sample_rows, random_state=42)

    X, customer_ids = prepare_features(raw_df)
    churn_proba = pipeline.predict_proba(X)[:, 1]
    churn_pred = (churn_proba >= args.threshold).astype(int)

    results = pd.DataFrame(
        {
            "churn_probability": np.round(churn_proba, 4),
            "churn_prediction": np.where(churn_pred == 1, "Yes", "No"),
        }
    )
    if customer_ids is not None:
        results.insert(0, "customerID", customer_ids.values)

    print(f"Scored {len(results)} rows (threshold={args.threshold}):")
    print(results.to_string(index=False))

    if args.top_features > 0:
        print_top_features(pipeline, top_n=args.top_features)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"\nSaved predictions to: {args.output}")


if __name__ == "__main__":
    main()
