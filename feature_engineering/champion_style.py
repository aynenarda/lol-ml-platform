"""Belirli bir rakipten bagimsiz, sampiyonun genel oyun tarzina dair
ozellikler: scaling, power spike, split push, team fight, snowball.

Bunlar matchup degil, clean_long_df (tum mac verisi) uzerinde (lane,
champion) bazinda hesaplanir - boylece ornek boyutu cok daha buyuktur.

BILINEN SINIRLAMA: scaling_score/power_spike, win_rate'in mac suresiyle
korelasyonuna dayanir. Bu KORELASYONEL bir proxy'dir, nedensellik degil -
kisa maclar genelde "ezici galibiyet" anlamina gelir, sampiyonun kendi
gucunden bagimsiz olabilir. Buyuk orneklemde (5000+ mac) makul bir sinyal
verir ama mutlak bir dogruluk iddiasi degildir.
"""

import pandas as pd

POWER_SPIKE_LABELS = {"short": "Erken Oyun", "medium": "Orta Oyun", "long": "Geç Oyun"}


def compute_champion_style(clean_long_df):
    df = clean_long_df.copy()
    df["duration_bucket"] = pd.qcut(df["gameDuration"], q=3, labels=["short", "medium", "long"])

    duration_winrate = (
        df.groupby(["TeamPosition", "ChampionName", "duration_bucket"], observed=True)["Win"]
        .mean()
        .unstack("duration_bucket")
    )

    scaling_score = (duration_winrate["long"] - duration_winrate["short"]).rename("scaling_score").reset_index()
    power_spike = (
        duration_winrate[["short", "medium", "long"]].idxmax(axis=1).map(POWER_SPIKE_LABELS)
        .rename("power_spike").reset_index()
    )

    style = (
        df.groupby(["TeamPosition", "ChampionName"])
        .agg(
            games_total=("Win", "size"),
            avg_building_damage=("DamageDealtToBuildings", "mean"),
            avg_assists=("Assists", "mean"),
            avg_kill_spree=("LargestKillingSpree", "mean"),
        )
        .reset_index()
    )

    # Lane-ici oran (lane'ler dogal olarak farkli oynanir - top laner
    # support'tan cok daha fazla bina hasari verir - o yuzden ham degil
    # oranli kullaniyoruz).
    for col, out_col in [
        ("avg_building_damage", "split_push_score"),
        ("avg_assists", "team_fight_score"),
        ("avg_kill_spree", "snowball_score"),
    ]:
        lane_avg = style.groupby("TeamPosition")[col].transform("mean")
        style[out_col] = style[col] / lane_avg

    style = style.merge(scaling_score, on=["TeamPosition", "ChampionName"])
    style = style.merge(power_spike, on=["TeamPosition", "ChampionName"])

    return style
