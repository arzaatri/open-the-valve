import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

DRIFT_COLUMNS = ["log_player_count", "depth_pct"]


def build_drift_report(
    reference_panel: pd.DataFrame,
    current_panel: pd.DataFrame,
    reference_cate: np.ndarray,
    current_cate: np.ndarray,
    output_path: str,
) -> None:
    """Data drift (has the underlying panel distribution shifted since the
    last run -- new games, seasonal shift, etc.) and CATE-output drift (has
    the *estimated effect itself* drifted) in one Evidently report. Both are
    plain numeric-column drift checks under DataDriftPreset, so the CATE
    arrays are merged in as an extra column rather than run as a separate
    report -- one HTML file, one preset, covering both questions.
    """
    reference = reference_panel[DRIFT_COLUMNS].copy()
    reference["cate"] = reference_cate
    current = current_panel[DRIFT_COLUMNS].copy()
    current["cate"] = current_cate

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    result.save_html(output_path)
