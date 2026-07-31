"""Model 1 (Counter Pick Recommendation Engine) icin final feature
store'unu uretir: matchup istatistikleri + champion stili + statik
zorluk verisi birlestirilmis hali."""

from feature_engineering.matchups import build_matchups, compute_counter_stats
from feature_engineering.champion_style import compute_champion_style
from feature_engineering.external_data import load_difficulty_ratings, DIFFICULTY_LABELS


def build_model1_table(clean_long_df):
    matchups = build_matchups(clean_long_df)
    counter_stats = compute_counter_stats(matchups)
    champion_style = compute_champion_style(clean_long_df)

    difficulty_ratings = load_difficulty_ratings()
    champion_style = champion_style.copy()
    champion_style["difficulty_tier"] = champion_style["ChampionName"].map(difficulty_ratings)
    champion_style["difficulty_label"] = champion_style["difficulty_tier"].map(DIFFICULTY_LABELS)

    model1_table = counter_stats.merge(
        champion_style[[
            "TeamPosition", "ChampionName", "scaling_score", "power_spike",
            "split_push_score", "team_fight_score", "snowball_score",
            "difficulty_tier", "difficulty_label",
        ]],
        on=["TeamPosition", "ChampionName"],
        how="left",
    )

    return model1_table
