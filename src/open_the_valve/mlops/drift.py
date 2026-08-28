import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

DRIFT_COLUMNS = ["log_player_count", "depth_pct"]
_CATE_DRIFT_P_THRESHOLD = 0.05


def _extract_drift_summary(result_dict: dict) -> tuple[float, bool]:
    """Pulls the two numbers Grafana needs out of Evidently's metrics list --
    matched by metric_name prefix rather than a fixed list index, so this
    doesn't silently break if DataDriftPreset's internal metric ordering
    changes in a future Evidently version."""
    metrics = result_dict["metrics"]
    dataset_drift_share = next(
        m["value"]["share"] for m in metrics if m["metric_name"].startswith("DriftedColumnsCount")
    )
    cate_p_value = next(m["value"] for m in metrics if "column=cate" in m["metric_name"])
    return dataset_drift_share, cate_p_value < _CATE_DRIFT_P_THRESHOLD


def build_drift_report(
    reference_panel: pd.DataFrame,
    current_panel: pd.DataFrame,
    reference_cate: np.ndarray,
    current_cate: np.ndarray,
    output_path: str,
) -> tuple[float, bool]:
    """Data drift (has the underlying panel distribution shifted since the
    last run -- new games, seasonal shift, etc.) and CATE-output drift (has
    the *estimated effect itself* drifted) in one Evidently report. Both are
    plain numeric-column drift checks under DataDriftPreset, so the CATE
    arrays are merged in as an extra column rather than run as a separate
    report -- one HTML file, one preset, covering both questions. Returns
    (dataset_drift_share, cate_drift_detected) so the caller can persist them
    for the Grafana ops dashboard.
    """
    reference = reference_panel[DRIFT_COLUMNS].copy()
    reference["cate"] = reference_cate
    current = current_panel[DRIFT_COLUMNS].copy()
    current["cate"] = current_cate

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    result.save_html(output_path)
    return _extract_drift_summary(result.dict())
