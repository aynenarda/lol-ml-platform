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
    win_rate, confidence_score (Wilson alt siniri), risk_score (Wilson
    araligi genisligi) ve lane pressure (first blood/tower rate).

    matchups_df'te bir 'weight' kolonu varsa (bkz. feature_engineering.recency),
    win_rate hesabi bu agirliklara gore yapilir - guncel patch'lerin
    maclari, eski patch'lerinkinden daha fazla soz sahibi olur. Weight
    kolonu yoksa, hepsi 1.0 agirlikli sayilir (eski davranisla ayni sonuc)."""
    df = matchups_df.copy()
    if "weight" not in df.columns:
        df["weight"] = 1.0
    df["weighted_win"] = df["weight"] * df["Win"].astype(float)

    stats = (
        df
        .groupby(["TeamPosition", "ChampionName", "ChampionName_opp"])
        .agg(
            games=("Win", "size"),                # ham (agirliksiz) gozlem sayisi
            effective_games=("weight", "sum"),     # agirlikli (etkin) ornek buyuklugu
            effective_wins=("weighted_win", "sum"),
            first_blood_rate=("FirstBloodKill", "mean"),
            first_tower_rate=("FirstTowerKill", "mean"),
        )
        .reset_index()
    )

    stats["win_rate"] = stats["effective_wins"] / stats["effective_games"]
    stats["confidence_score"] = _wilson_bound(stats["effective_wins"], stats["effective_games"], sign=-1)
    wilson_upper = _wilson_bound(stats["effective_wins"], stats["effective_games"], sign=1)
    stats["risk_score"] = wilson_upper - stats["confidence_score"]
    # has_enough_data HAM gozlem sayisina gore - "gercekten kac mac oynandi"
    # sorusunun cevabi, agirliklandirmadan bagimsiz olmali.
    stats["has_enough_data"] = stats["games"] >= MIN_GAMES

    return stats
