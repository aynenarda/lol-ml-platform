"""Gecersiz maclari tespit edip veriyi temizler.

Temizleme kurallari (Milestone 3'te kesfedildi):
1. 5 dakikadan kisa maclar (remake)
2. Erken teslim olan maclar
3. TeamPosition bilgisi eksik olan maclar
4. Kazanan bilgisi tutarsiz olan maclar (iki takim da Win=False - nadir
   bir Riot API veri kaydi hatasi, feature engineering asamasinda
   self-join row count dogrulamasi yapilirken kesfedildi)
"""


def find_invalid_match_ids(wide_df, long_df):
    """Yukaridaki 4 kuralin BIRLESIMINE uyan matchId'leri dondurur."""
    short_game_ids = set(wide_df.loc[wide_df["gameDuration"] < 300, "matchId"])
    early_surrender_ids = set(long_df.loc[long_df["GameEndedInEarlySurrender"], "matchId"])
    missing_position_ids = set(long_df.loc[long_df["TeamPosition"].isna(), "matchId"])
    no_winner_ids = set(
        long_df.groupby("matchId")["Win"].nunique().loc[lambda s: s != 2].index
    )

    return short_game_ids | early_surrender_ids | missing_position_ids | no_winner_ids


def clean_long_data(long_df, invalid_ids):
    """Gecersiz maclari cikarir, artik gereksiz olan
    GameEndedInEarlySurrender kolonunu dusurur (hep False olacak)."""
    clean = long_df[~long_df["matchId"].isin(invalid_ids)].copy()
    return clean.drop(columns=["GameEndedInEarlySurrender"])
