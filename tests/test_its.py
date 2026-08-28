from datetime import date

import pandas as pd
import pytest

from open_the_valve.causal.its import estimate_its_effect, run_its_all_events

_OUTCOME = "player_count"


def _flat_game_panel(
    game_id: int, start: date, n_days: int, discounted_dates: set[date]
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D").date
    return pd.DataFrame(
        {
            "game_id": game_id,
            "date": dates,
            "is_discounted": [d in discounted_dates for d in dates],
            _OUTCOME: [200.0] * n_days,
        }
    )


def test_estimate_its_effect_detects_a_post_treatment_bump():
    event_start = date(2026, 2, 1)
    dates = pd.date_range(date(2026, 1, 1), date(2026, 2, 14), freq="D").date
    panel = pd.DataFrame({"game_id": 1, "date": dates})
    panel["is_discounted"] = panel["date"] >= event_start
    panel[_OUTCOME] = [200.0 if d < event_start else 400.0 for d in dates]

    result = estimate_its_effect(
        panel,
        game_id=1,
        event_start=event_start,
        pre_window_days=14,
        post_window_days=7,
        min_pre_period_obs=5,
        outcome_col=_OUTCOME,
    )

    assert result is not None
    assert result.mean_post_effect == pytest.approx(200.0, abs=1.0)


def test_estimate_its_effect_skips_when_pre_period_is_too_thin():
    event_start = date(2026, 2, 1)
    panel = _flat_game_panel(1, date(2026, 1, 30), 10, discounted_dates=set())
    panel.loc[panel["date"] >= event_start, "is_discounted"] = True

    result = estimate_its_effect(
        panel,
        game_id=1,
        event_start=event_start,
        pre_window_days=14,
        post_window_days=7,
        min_pre_period_obs=5,  # only 2 pre-period days available
        outcome_col=_OUTCOME,
    )

    assert result is None


def test_run_its_all_events_skips_insufficient_events():
    event_start = date(2026, 2, 1)
    dates = pd.date_range(date(2026, 1, 1), date(2026, 2, 14), freq="D").date
    panel = pd.DataFrame({"game_id": 1, "date": dates})
    panel["is_discounted"] = panel["date"] >= event_start
    panel[_OUTCOME] = [200.0 if d < event_start else 400.0 for d in dates]

    events = pd.DataFrame(
        [
            {"game_id": 1, "start_at": pd.Timestamp(event_start, tz="UTC")},
            {
                "game_id": 1,
                "start_at": pd.Timestamp(date(2026, 1, 2), tz="UTC"),
            },  # too little pre-history
        ]
    )
    from open_the_valve.config_models import ItsConfig

    config = ItsConfig(pre_window_days=14, post_window_days=7, min_pre_period_obs=5)
    result = run_its_all_events(panel, events, config, outcome_col=_OUTCOME)

    assert len(result) == 1
    assert result.iloc[0]["mean_post_effect"] == pytest.approx(200.0, abs=1.0)
