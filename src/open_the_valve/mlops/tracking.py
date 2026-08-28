import logging
import os

import mlflow
import pandas as pd

from open_the_valve.causal.cate_estimators import CateRun, predict_row_cate
from open_the_valve.config_models import CateConfig, MlopsConfig
from open_the_valve.mlops.drift import DRIFT_COLUMNS

logger = logging.getLogger(__name__)

_PRIMARY_ESTIMATOR = "causal_forest_dml"


def log_causal_run(
    cate_run: CateRun,
    its_results: pd.DataFrame,
    panel: pd.DataFrame,
    cate_config: CateConfig,
    mlops_config: MlopsConfig,
    findings_dir: str,
) -> str:
    """Logs one MLflow run for a causal-analysis execution: hyperparams,
    per-method ATE metrics, the report artifacts reports/findings.py already
    wrote, and a new cate_predictions.parquet -- game_id/date, the primary
    estimator's per-row CATE, plus the DRIFT_COLUMNS snapshot -- so a later
    retrain can build a drift report against this run without needing the
    full panel parquet, just this one artifact. Returns the MLflow run_id,
    which the caller records in causal_run_history.
    """
    mlflow.set_tracking_uri(mlops_config.tracking_uri)
    mlflow.set_experiment(mlops_config.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "xgb_n_estimators": cate_config.xgb_n_estimators,
                "xgb_max_depth": cate_config.xgb_max_depth,
                "xgb_learning_rate": cate_config.xgb_learning_rate,
                "cf_n_estimators": cate_config.cf_n_estimators,
                "cf_min_samples_leaf": cate_config.cf_min_samples_leaf,
                "estimator_names": ",".join(cate_config.estimator_names),
                "refutation_num_simulations": cate_config.refutation_num_simulations,
                "panel_start_date": str(panel["date"].min()),
                "panel_end_date": str(panel["date"].max()),
                "panel_row_count": len(panel),
                "n_treated_rows": int(panel["is_discounted"].sum()),
            }
        )

        mlflow.log_metrics({f"ate_{name}": r.ate for name, r in cate_run.results.items()})
        if not its_results.empty:
            mlflow.log_metric("ate_its", its_results["mean_post_effect"].mean())

        for artifact_name in ("findings.md", "comparison_table.csv", "cate_slices.csv"):
            path = os.path.join(findings_dir, artifact_name)
            if os.path.exists(path):
                mlflow.log_artifact(path)

        primary = cate_run.results[_PRIMARY_ESTIMATOR]
        predictions = cate_run.data[["game_id", "date", *DRIFT_COLUMNS]].copy()
        predictions[_PRIMARY_ESTIMATOR] = predict_row_cate(primary, cate_run)
        predictions_path = os.path.join(findings_dir, "cate_predictions.parquet")
        predictions.to_parquet(predictions_path, index=False)
        mlflow.log_artifact(predictions_path)

        logger.info("logged mlflow run %s", run.info.run_id)
        return run.info.run_id
