"""Model 2+3+4 (birlesik): Matchup Intelligence + Item Onerisi + Rune Onerisi.

Tasarim prensibi: item/rune ISIMLERI statik referans veriden (Data
Dragon/Community Dragon aynasi) geliyor, ama HANGI item/rune'un
ONERILECEGI tamamen bizim mac verimizden (kazanilan maclardaki
frekans) cikiyor - Riot'un hazir "onerilen build"ini kullanmiyoruz.
"""

from collections import Counter

ITEM_COLS = [f"Item{i}" for i in range(7)]
RUNE_COLS = [
    "PrimaryStylePerk1", "PrimaryStylePerk2", "PrimaryStylePerk3", "PrimaryStylePerk4",
    "SubStylePerk1", "SubStylePerk2",
]

MIN_GAMES_FOR_BUILD = 15  # item/rune onerisi icin (win rate'ten daha az veri yeterli)


def compute_gold_cs_diff(matchups_df):
    """Her (lane, benim sampiyonum, rakip) icin beklenen gold ve CS farkini
    (benim - rakibin) hesaplar. matchups_df zaten self-join'den geldigi
    icin GoldEarned_opp / TotalMinionsKilled_opp kolonlari hazir."""
    df = matchups_df.copy()
    df["gold_diff"] = df["GoldEarned"] - df["GoldEarned_opp"]
    df["cs_diff"] = df["TotalMinionsKilled"] - df["TotalMinionsKilled_opp"]

    return (
        df.groupby(["TeamPosition", "ChampionName", "ChampionName_opp"])
        .agg(expected_gold_diff=("gold_diff", "mean"), expected_cs_diff=("cs_diff", "mean"))
        .reset_index()
    )


def _top_items(win_rows, top_n=6):
    counter = Counter()
    n_games = len(win_rows)
    for _, row in win_rows.iterrows():
        for col in ITEM_COLS:
            item_id = row[col]
            if item_id and item_id != 0:
                counter[item_id] += 1

    return [
        {"item_id": item_id, "pick_rate": count / n_games}
        for item_id, count in counter.most_common(top_n)
    ]


def _top_rune_combo(win_rows):
    combos = win_rows[RUNE_COLS].apply(tuple, axis=1)
    if combos.empty:
        return None
    most_common_combo, count = Counter(combos).most_common(1)[0]
    return {
        "perk_ids": dict(zip(RUNE_COLS, most_common_combo)),
        "pick_rate": count / len(win_rows),
    }


def compute_item_rune_recommendations(matchups_df):
    """Her (lane, benim sampiyonum, rakip) icin, SADECE KAZANILAN
    maclardaki en sik item seti ve rune kombinasyonunu bulur."""
    results = []
    wins_only = matchups_df[matchups_df["Win"]]

    for (lane, champ, opp), group in wins_only.groupby(["TeamPosition", "ChampionName", "ChampionName_opp"]):
        if len(group) < MIN_GAMES_FOR_BUILD:
            continue
        results.append({
            "TeamPosition": lane,
            "ChampionName": champ,
            "ChampionName_opp": opp,
            "win_games_used": len(group),
            "top_items": _top_items(group),
            "top_rune_combo": _top_rune_combo(group),
        })

    return results
