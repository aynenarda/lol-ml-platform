"""Matchup (rakip cifti) verisini uretir: self-join + win rate + Wilson
Score Interval + lane pressure."""

import numpy as np
import pandas as pd

MIN_GAMES = 20


def build_matchups(clean_long_df):
    """clean_long_df'i kendisiyle (matchId, TeamPosition) uzerinden
    birlestirip, Win != Win_opp filtresiyle gercek rakip ciftlerini
    dondurur (yon farkli: A'dan B'ye ve B'den A'ya ayri satirlar)."""
    matchups = clean_long_df.merge(
        clean_long_df, on=["matchId", "TeamPosition"], suffixes=("", "_opp")
    )
    return matchups[matchups["Win"] != matchups["Win_opp"]]


def _wilson_bound(wins, games, z=1.96, sign=-1):
    n = games
    p_hat = wins / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / denominator
    return center + sign * margin


def compute_counter_stats(matchups_df):
    """Her (lane, benim sampiyonum, rakip sampiyon) uclusu icin games,
    wins, win_rate, confidence_score (Wilson alt siniri), risk_score
    (Wilson araligi genisligi) ve lane pressure (first blood/tower rate)."""
    stats = (
        matchups_df
        .groupby(["TeamPosition", "ChampionName", "ChampionName_opp"])
        .agg(
            games=("Win", "size"),
            wins=("Win", "sum"),
            first_blood_rate=("FirstBloodKill", "mean"),
            first_tower_rate=("FirstTowerKill", "mean"),
        )
        .reset_index()
    )

    stats["win_rate"] = stats["wins"] / stats["games"]
    stats["confidence_score"] = _wilson_bound(stats["wins"], stats["games"], sign=-1)
    wilson_upper = _wilson_bound(stats["wins"], stats["games"], sign=1)
    stats["risk_score"] = wilson_upper - stats["confidence_score"]
    stats["has_enough_data"] = stats["games"] >= MIN_GAMES

    return stats
