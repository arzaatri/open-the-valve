from datetime import date

import numpy as np
import pandas as pd

from open_the_valve.features.build_panel import (
    build_treatment_frame,
    compute_detrended_outcome,
    flag_platform_sale_windows,
)


def _spine(game_ids: list[int], start: date, end: date) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D").date
    return pd.DataFrame([(gid, d) for gid in game_ids for d in dates], columns=["game_id", "date"])


def test_build_treatment_frame_expands_intervals_and_takes_max_depth():
    spine = _spine([1], date(2026, 1, 1), date(2026, 1, 10))
    events = pd.DataFrame(
        [
            {
                "game_id": 1,
                "start_at": pd.Timestamp("2026-01-03", tz="UTC"),
                "end_at": pd.Timestamp("2026-01-05", tz="UTC"),
                "depth_pct": 25,
            },
            {
                "game_id": 1,
                "start_at": pd.Timestamp("2026-01-04", tz="UTC"),
                "end_at": pd.Timestamp("2026-01-04", tz="UTC"),
                "depth_pct": 50,
            },
        ]
    )
    result = build_treatment_frame(events, spine).set_index("date")

    assert not result.loc[date(2026, 1, 2), "is_discounted"]
    assert result.loc[date(2026, 1, 3), "depth_pct"] == 25
    assert result.loc[date(2026, 1, 4), "depth_pct"] == 50  # overlapping events -> max depth
    assert result.loc[date(2026, 1, 5), "depth_pct"] == 25
    assert not result.loc[date(2026, 1, 6), "is_discounted"]


def test_build_treatment_frame_treats_open_event_as_active_through_spine_end():
    spine = _spine([1], date(2026, 1, 1), date(2026, 1, 5))
    events = pd.DataFrame(
        [
            {
                "game_id": 1,
                "start_at": pd.Timestamp("2026-01-04", tz="UTC"),
                "end_at": pd.NaT,
                "depth_pct": 10,
            }
        ]
    )
    result = build_treatment_frame(events, spine).set_index("date")

    assert result.loc[date(2026, 1, 4), "is_discounted"]
    assert result.loc[date(2026, 1, 5), "is_discounted"]  # spine's last date, event still open


def test_compute_detrended_outcome_excludes_discounted_days_and_current_day():
    panel = pd.DataFrame(
        {
            "game_id": [1] * 5,
            "date": pd.date_range("2026-01-01", periods=5).date,
            "is_discounted": [False, False, True, False, False],
            "player_count": [100, 100, 10_000, 300, 300],
        }
    )
    detrended = compute_detrended_outcome(panel, window_days=14)

    # day 3 (index 2) is discounted and shouldn't corrupt day 4's baseline
    assert detrended.iloc[3] == 300 - 100
    # day 1 has no prior history at all -> NaN baseline
    assert np.isnan(detrended.iloc[0])


def test_flag_platform_sale_windows_thresholds_on_daily_discount_fraction():
    panel = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 1, 2, 3],
            "date": [date(2026, 1, 1)] * 3 + [date(2026, 1, 2)] * 3,
            "is_discounted": [True, True, False, True, False, False],
        }
    )
    flags = flag_platform_sale_windows(panel, threshold_frac=0.6)

    assert flags[panel["date"] == date(2026, 1, 1)].all()  # 2/3 discounted >= 0.6
    assert not flags[panel["date"] == date(2026, 1, 2)].any()  # 1/3 discounted < 0.6
