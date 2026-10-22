"""
Shared data-quality utility for the ML pipeline.

The source dataset has a reporting artifact: several states show a >85%
year-over-year collapse in Total Crime for 2020 (Delhi 11,648->9,
Maharashtra 29,529->158, similarly Chandigarh/Rajasthan/Telangana/UP/West
Bengal), while the NATIONAL total barely moves that year (333,717 ->
302,186, ~9%). That mismatch means these are state-level data/reporting
gaps, not real crime drops -- training a model to predict them would just
teach it to chase noise. We flag and exclude them from training targets
(they stay visible in the raw history/charts, just not used to update model
weights), and report backtest accuracy for "clean" vs "flagged" states
separately so the accuracy numbers aren't silently inflated or deflated by
them.
"""
import pandas as pd


def flag_anomalies(df: pd.DataFrame, crime: str = "Total Crime", drop_ratio: float = 0.15) -> set:
    """Returns a set of (state, year) tuples where `crime` fell below
    `drop_ratio` of either the prior year's OR two-years-prior value for
    that state. Checking 2 years back as well as 1 catches a SUSTAINED dip
    (e.g. year T crashes, year T+1 stays near the crashed level so its
    1-year-back ratio looks "normal" even though it's still broken) --
    without the 2-year check, only the first broken year of a multi-year
    glitch gets flagged."""
    piv = df.pivot(index="state", columns="year", values=crime).sort_index(axis=1)
    ratio_1 = piv / piv.shift(1, axis=1)
    ratio_2 = piv / piv.shift(2, axis=1)
    mask = (ratio_1 < drop_ratio) | (ratio_2 < drop_ratio)
    flagged = mask.stack()
    return set(flagged[flagged].index.tolist())


def repair_series_for_input(values_by_year: dict, state: str, anomaly_set: set) -> dict:
    """Returns a copy of {year: value} with flagged years replaced by a
    sensible estimate, for use as MODEL INPUT (windows, trend fitting)
    only -- never for computing backtest error, which must always compare
    against the real reported value.

    Flagged years are linearly interpolated (in log space, by the caller)
    between the nearest clean years before/after. A flagged year with no
    clean year after it (e.g. the series ends on a flagged year) is
    extrapolated by carrying forward the most recent clean year-over-year
    trend, so a forward-looking forecast doesn't anchor itself to a known
    data glitch.
    """
    years = sorted(values_by_year.keys())
    clean_years = [y for y in years if (state, y) not in anomaly_set]
    if not clean_years:
        return dict(values_by_year)  # nothing to anchor to; leave as-is

    repaired = dict(values_by_year)
    for y in years:
        if (state, y) not in anomaly_set:
            continue
        before = [cy for cy in clean_years if cy < y]
        after = [cy for cy in clean_years if cy > y]
        if before and after:
            b, a = max(before), min(after)
            frac = (y - b) / (a - b)
            repaired[y] = values_by_year[b] + frac * (values_by_year[a] - values_by_year[b])
        elif before and not after:
            # extrapolate using the trend between the last two clean years
            b = max(before)
            prior_clean = [cy for cy in clean_years if cy < b]
            if prior_clean:
                b2 = max(prior_clean)
                slope = (values_by_year[b] - values_by_year[b2]) / (b - b2)
            else:
                slope = 0.0
            repaired[y] = values_by_year[b] + slope * (y - b)
        elif after and not before:
            a = min(after)
            repaired[y] = values_by_year[a]
    return repaired
