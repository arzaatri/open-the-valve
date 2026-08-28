"""Cron entrypoint: rebuilds the panel and reruns the causal analysis only
when enough new daily player-count data has accumulated since the last
recorded run. Not installed automatically -- add a system crontab line, e.g.:

    0 6 * * * cd <repo> && uv run python -m open_the_valve.pipelines.retrain_if_needed
"""

import logging
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import func, select

from open_the_valve.config_models import AppConfig
from open_the_valve.db import repo
from open_the_valve.db.models import PlayerCountGranularity, PlayerCountHistory
from open_the_valve.db.session import make_engine, session_scope
from open_the_valve.features.build_panel import run as build_panel
from open_the_valve.io_utils.hydra_entrypoint import hydra_entrypoint
from open_the_valve.mlops.drift import build_drift_report
from open_the_valve.pipelines.run_causal_analysis import run as run_causal_analysis

load_dotenv()
logger = logging.getLogger(__name__)

_PRIMARY_ESTIMATOR = "causal_forest_dml"
_DRIFT_OUTPUT_DIR = "outputs/mlops"


def should_retrain(
    last_panel_row_count: int | None, current_row_count: int, threshold: int
) -> bool:
    """Pure gate check: retrain if there's never been a run, or enough new
    daily player-count rows have accumulated since the last one.
    """
    if last_panel_row_count is None:
        return True
    return (current_row_count - last_panel_row_count) >= threshold


def run(config: AppConfig) -> None:
    engine = make_engine(config.db)
    with session_scope(engine) as session:
        last_run = repo.get_latest_causal_run(session)
        # Extract scalars while the session is open -- last_run is detached
        # (and its attributes unreadable) once session_scope's `with` exits.
        last_panel_row_count = last_run.panel_row_count if last_run else None
        last_mlflow_run_id = last_run.mlflow_run_id if last_run else None
        current_row_count = session.execute(
            select(func.count())
            .select_from(PlayerCountHistory)
            .where(PlayerCountHistory.granularity == PlayerCountGranularity.DAILY)
        ).scalar_one()
    if not should_retrain(
        last_panel_row_count, current_row_count, config.retrain.new_rows_threshold
    ):
        logger.info(
            "only %d new daily player-count rows since last run (threshold=%d), skipping",
            current_row_count - (last_panel_row_count or 0),
            config.retrain.new_rows_threshold,
        )
        return

    logger.info(
        "retrain threshold met (%d daily rows now vs %s at last run), rebuilding",
        current_row_count,
        last_panel_row_count,
    )
    build_panel(config)
    run_causal_analysis(config)

    if last_mlflow_run_id is None:
        logger.info("first run ever, no prior run to compare drift against")
        return

    mlflow.set_tracking_uri(config.mlops.tracking_uri)
    reference_path = mlflow.artifacts.download_artifacts(
        run_id=last_mlflow_run_id, artifact_path="cate_predictions.parquet"
    )
    reference = pd.read_parquet(reference_path)
    current = pd.read_parquet(
        os.path.join(
            os.path.dirname(config.causal.findings.output_path), "cate_predictions.parquet"
        )
    )

    os.makedirs(_DRIFT_OUTPUT_DIR, exist_ok=True)
    build_drift_report(
        reference,
        current,
        reference[_PRIMARY_ESTIMATOR].to_numpy(),
        current[_PRIMARY_ESTIMATOR].to_numpy(),
        os.path.join(_DRIFT_OUTPUT_DIR, "drift_report.html"),
    )
    logger.info("wrote drift report comparing against mlflow_run_id=%s", last_mlflow_run_id)


main = hydra_entrypoint(run)

if __name__ == "__main__":
    main()
