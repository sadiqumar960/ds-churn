import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import predict as predict_mod  # noqa: E402
import train as train_mod      # noqa: E402


def _make_telco_df(n_rows: int = 24, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contracts = ["Month-to-month", "One year", "Two year"]
    internets = ["DSL", "Fiber optic", "No"]
    payments = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    tenure = rng.integers(0, 73, size=n_rows)
    monthly = np.round(rng.uniform(18.5, 118.0, size=n_rows), 2)
    total_raw = np.round(tenure * monthly, 2).astype(str)
    # Inject blank strings and whitespace to test TotalCharges coercion
    total_raw[0] = " "
    total_raw[1] = ""

    churn = np.where((tenure < 12) & (np.arange(n_rows) % 2 == 0), "Yes", "No")

    return pd.DataFrame(
        {
            "customerID": [f"CUST-{i:04d}" for i in range(n_rows)],
            "gender": rng.choice(["Male", "Female"], size=n_rows),
            "SeniorCitizen": rng.integers(0, 2, size=n_rows),
            "Partner": rng.choice(["Yes", "No"], size=n_rows),
            "Dependents": rng.choice(["Yes", "No"], size=n_rows),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], size=n_rows),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n_rows),
            "InternetService": rng.choice(internets, size=n_rows),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n_rows),
            "Contract": rng.choice(contracts, size=n_rows),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n_rows),
            "PaymentMethod": rng.choice(payments, size=n_rows),
            "MonthlyCharges": monthly,
            "TotalCharges": total_raw,
            "Churn": churn,
        }
    )


def test_totalcharges_coercion_and_target_mapping(tmp_path: Path):
    csv_path = tmp_path / "telco.csv"
    _make_telco_df().to_csv(csv_path, index=False)

    X, y = train_mod.load_and_clean_data(csv_path)

    assert "customerid" not in X.columns
    assert "churn" not in X.columns
    assert pd.api.types.is_numeric_dtype(X["totalcharges"])
    assert X["totalcharges"].iloc[0] is not None and np.isnan(X["totalcharges"].iloc[0])
    assert np.isnan(X["totalcharges"].iloc[1])
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y) == 24


def test_prepare_features_preserves_customer_id_and_coerces_totalcharges():
    raw = _make_telco_df(n_rows=5)
    X, customer_ids = predict_mod.prepare_features(raw)

    assert customer_ids is not None
    assert list(customer_ids) == ["CUST-0000", "CUST-0001", "CUST-0002", "CUST-0003", "CUST-0004"]
    assert "customerid" not in X.columns
    assert "churn" not in X.columns
    assert pd.api.types.is_numeric_dtype(X["totalcharges"])
    assert np.isnan(X["totalcharges"].iloc[0])


def test_unseen_categories_do_not_fail_inference(tmp_path: Path):
    train_df = _make_telco_df(n_rows=30, seed=10)
    X_train, y_train = train_mod.load_and_clean_data(
        _write_csv(train_df, tmp_path / "train.csv")
    )

    pipe = train_mod.build_pipeline(X_train, scale_pos_weight=1.5)
    pipe.fit(X_train, y_train)

    # Introduce unseen categorical values and blank TotalCharges at inference time
    unseen_df = _make_telco_df(n_rows=4, seed=99).drop(columns=["Churn"])
    unseen_df.loc[0, "Contract"] = "BrandNewContractTier"
    unseen_df.loc[1, "PaymentMethod"] = "CryptoAutoPay"
    unseen_df.loc[2, "TotalCharges"] = ""

    X_infer, _ = predict_mod.prepare_features(unseen_df)
    probs = pipe.predict_proba(X_infer)[:, 1]

    assert probs.shape == (4,)
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_cli_threshold_scoring_and_csv_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    train_df = _make_telco_df(n_rows=30, seed=7)
    X_train, y_train = train_mod.load_and_clean_data(
        _write_csv(train_df, tmp_path / "train.csv")
    )
    pipe = train_mod.build_pipeline(X_train, scale_pos_weight=1.5)
    pipe.fit(X_train, y_train)

    # Create a fake repo layout inside tmp_path so predict.py loads our temp pipeline
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    fake_predict_file = fake_src / "predict.py"
    fake_predict_file.write_text("# marker\n", encoding="utf-8")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(pipe, models_dir / "churn_pipeline.joblib")

    score_csv = _write_csv(_make_telco_df(n_rows=6, seed=12).drop(columns=["Churn"]), tmp_path / "score.csv")
    out_csv = tmp_path / "preds" / "scored.csv"

    # Point predict.__file__ to fake_src/predict.py so project_root resolves to tmp_path
    monkeypatch.setattr(predict_mod, "__file__", str(fake_predict_file))

    # Threshold = 0.0 -> all rows predicted "Yes"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--input",
            str(score_csv),
            "--threshold",
            "0.0",
            "--output",
            str(out_csv),
            "--top-features",
            "3",
        ],
    )
    predict_mod.main()
    captured_low = capsys.readouterr().out
    assert out_csv.exists()
    scored_low = pd.read_csv(out_csv)
    assert list(scored_low.columns) == ["customerID", "churn_probability", "churn_prediction"]
    assert len(scored_low) == 6
    assert (scored_low["churn_prediction"] == "Yes").all()
    assert "Top 3 Feature Importances" in captured_low

    # Threshold = 1.01 -> all rows predicted "No"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--input",
            str(score_csv),
            "--threshold",
            "1.01",
            "--output",
            str(out_csv),
            "--top-features",
            "0",
        ],
    )
    predict_mod.main()
    capsys.readouterr()
    scored_high = pd.read_csv(out_csv)
    assert (scored_high["churn_prediction"] == "No").all()


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def test_train_main_generates_pipeline_and_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import json

    fake_src = tmp_path / "src"
    fake_src.mkdir()
    fake_train_file = fake_src / "train.py"
    fake_train_file.write_text("# marker\n", encoding="utf-8")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # 60 rows with enough positive cases so both train (48) and test (12) include Churn=Yes
    train_df = _make_telco_df(n_rows=60, seed=21)
    train_df.loc[::3, "Churn"] = "Yes"
    train_df.loc[1::6, "Churn"] = "No"
    _write_csv(train_df, data_dir / "telecom_churn_dataset.csv")

    monkeypatch.setattr(train_mod, "__file__", str(fake_train_file))
    train_mod.main()

    pipeline_file = tmp_path / "models" / "churn_pipeline.joblib"
    metrics_file = tmp_path / "models" / "churn_metrics.json"

    assert pipeline_file.exists()
    assert metrics_file.exists()

    loaded_pipe = joblib.load(pipeline_file)
    assert hasattr(loaded_pipe, "predict_proba")

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    for key in ("accuracy", "f1", "roc_auc", "pr_auc", "scale_pos_weight", "train_rows", "test_rows"):
        assert key in metrics

    assert metrics["train_rows"] == 48
    assert metrics["test_rows"] == 12
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["scale_pos_weight"] > 0
