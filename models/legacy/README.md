# Legacy Artifacts

Migrated from the original Windows workspace (`Project Program and code`) for reference and comparison:
- `best_churn_model.pkl`: Earlier pickled XGBoost model.
- `feature_names.pkl`: Original 19 lowercase feature names.
- `label_encoders.pkl`: Empty dictionary (`{}`) in the migrated artifact.
- `scaler.pkl`: NumPy array of feature names rather than a fitted scaler.

Active production/training artifacts are located in the parent `models/` directory (`churn_pipeline.joblib` and `churn_metrics.json`).
