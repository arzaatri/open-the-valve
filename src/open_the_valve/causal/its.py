import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import statsmodels.api as sm

from open_the_valve.config_models import ItsConfig

logger = logging.getLogger(__name__)


@dataclass
class ItsEffectResult:
    game_id: int
    event_start: date
    n_pre_obs: int
    daily_effects: pd.DataFrame  # columns: date, actual, counterfactual, effect
    mean_post_effect: float


def fit_pretrend(
    game_panel: pd.DataFrame, event_start: date, pre_window_days: int, outcome_col: str
) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    """OLS(outcome ~ day_offset) over non-discounted days in
    [event_start - pre_window_days, event_start). Returns None if there's no
    usable pre-period data at all (caller enforces the minimum-observations gate).
    """
    window_start = event_start - timedelta(days=pre_window_days)
    pre = game_panel[
        (game_panel["date"] >= window_start)
        & (game_panel["date"] < event_start)
        & (~game_panel["is_discounted"])
    ].dropna(subset=[outcome_col])
    if pre.empty:
        return None

    day_offset = pre["date"].apply(lambda d: (d - event_start).days)
    X = sm.add_constant(day_offset)
    return sm.OLS(pre[outcome_col], X).fit()


def estimate_its_effect(
    game_panel: pd.DataFrame,
    game_id: int,
    event_start: date,
    pre_window_days: int,
    post_window_days: int,
    min_pre_period_obs: int,
    outcome_col: str,
) -> ItsEffectResult | None:
    """Fits the pre-treatment trend and compares its forward extrapolation
    against actual post-treatment outcomes. Returns None (caller skips/logs)
    when there isn't enough pre-period history to trust a trend line -- the
    real branching logic in this module.

    ponytail: pre-period days aren't checked against other nearby events, so a
    prior discount's lingering effect on non-discounted days can leak into a
    pretrend fit for events spaced closer together than pre_window_days;
    revisit with an explicit inter-event gap filter if events cluster tightly.
    """
    window_start = event_start - timedelta(days=pre_window_days)
    n_pre_obs = (
        game_panel[
            (game_panel["date"] >= window_start)
            & (game_panel["date"] < event_start)
            & (~game_panel["is_discounted"])
        ][outcome_col]
        .notna()
        .sum()
    )
    if n_pre_obs < min_pre_period_obs:
        return None

    model = fit_pretrend(game_panel, event_start, pre_window_days, outcome_col)
    if model is None:
        return None

    window_end = event_start + timedelta(days=post_window_days)
    post = game_panel[
        (game_panel["date"] >= event_start) & (game_panel["date"] < window_end)
    ].dropna(subset=[outcome_col])
    if post.empty:
        return None

    day_offset = post["date"].apply(lambda d: (d - event_start).days)
    counterfactual = model.predict(sm.add_constant(day_offset, has_constant="add"))

    daily_effects = pd.DataFrame(
        {
            "date": post["date"].to_numpy(),
            "actual": post[outcome_col].to_numpy(),
            "counterfactual": counterfactual.to_numpy(),
        }
    )
    daily_effects["effect"] = daily_effects["actual"] - daily_effects["counterfactual"]

    return ItsEffectResult(
        game_id=game_id,
        event_start=event_start,
        n_pre_obs=int(n_pre_obs),
        daily_effects=daily_effects,
        mean_post_effect=float(daily_effects["effect"].mean()),
    )


def run_its_all_events(
    panel: pd.DataFrame, discount_events: pd.DataFrame, config: ItsConfig, outcome_col: str
) -> pd.DataFrame:
    """One row per qualifying discount event (game_id, event_start, n_pre_obs,
    mean_post_effect). Events without sufficient pre-treatment history are
    skipped and logged rather than silently dropped.
    """
    rows = []
    n_skipped = 0
    for event in discount_events.itertuples(index=False):
        event_start = (
            event.start_at.tz_localize(None).date()
            if event.start_at.tzinfo
            else event.start_at.date()
        )
        game_panel = panel[panel["game_id"] == event.game_id]
        result = estimate_its_effect(
            game_panel,
            game_id=event.game_id,
            event_start=event_start,
            pre_window_days=config.pre_window_days,
            post_window_days=config.post_window_days,
            min_pre_period_obs=config.min_pre_period_obs,
            outcome_col=outcome_col,
        )
        if result is None:
            n_skipped += 1
            continue
        rows.append(
            {
                "game_id": result.game_id,
                "event_start": result.event_start,
                "n_pre_obs": result.n_pre_obs,
                "mean_post_effect": result.mean_post_effect,
            }
        )

    logger.info(
        "ITS: fit %d events, skipped %d (insufficient pre-period history)", len(rows), n_skipped
    )
    return pd.DataFrame(rows)


def slice_its_by_subgroup(
    its_results: pd.DataFrame, panel: pd.DataFrame, slice_col: str
) -> pd.DataFrame:
    """Groups per-event ITS effects by a game-level covariate (genre, price_tier,
    depth_bucket, ...). Output shape mirrors causal.cate_estimators.slice_cate_by_subgroup
    so reports/findings.py can merge both without special-casing.
    """
    covariate = panel[["game_id", slice_col]].drop_duplicates()
    merged = its_results.merge(covariate, on="game_id", how="left")
    return (
        merged.groupby(slice_col, observed=True)["mean_post_effect"]
        .agg(mean_effect="mean", n_events="count")
        .reset_index()
        .rename(columns={slice_col: "slice_value"})
        .assign(slice_dim=slice_col)
    )
