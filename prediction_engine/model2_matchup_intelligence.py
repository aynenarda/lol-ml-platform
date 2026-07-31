"""Model 2+3+4: Matchup Intelligence + Item + Rune raporu.

Girdi: benim sampiyonum + rakip sampiyon + lane (Model 1'in tersine,
burada rakibi DEGIL, kendi secimimi de biliyoruz).
Cikti: win rate, beklenen gold/cs farki, item onerisi, rune onerisi -
hepsi bizim mac verimizden, statik veri sadece isimlendirme icin."""

import json

import pandas as pd

from feature_engineering.external_data import load_item_names, load_perk_names

RUNE_COLS = [
    "PrimaryStylePerk1", "PrimaryStylePerk2", "PrimaryStylePerk3", "PrimaryStylePerk4",
    "SubStylePerk1", "SubStylePerk2",
]


def load_model2_data():
    model1_table = pd.read_parquet("data/processed/model1_counter_features.parquet")
    gold_cs_table = pd.read_parquet("data/processed/model2_gold_cs.parquet")

    with open("data/processed/model2_item_rune.json", encoding="utf-8") as f:
        item_rune_list = json.load(f)
    item_rune_lookup = {
        (r["TeamPosition"], r["ChampionName"], r["ChampionName_opp"]): r
        for r in item_rune_list
    }

    item_names = load_item_names()
    perk_names = load_perk_names()

    return model1_table, gold_cs_table, item_rune_lookup, item_names, perk_names


def get_matchup_report(my_champion, enemy_champion, lane,
                        model1_table, gold_cs_table, item_rune_lookup, item_names, perk_names):
    key = (lane, my_champion, enemy_champion)

    win_row = model1_table[
        (model1_table["TeamPosition"] == lane)
        & (model1_table["ChampionName"] == my_champion)
        & (model1_table["ChampionName_opp"] == enemy_champion)
    ]
    gold_cs_row = gold_cs_table[
        (gold_cs_table["TeamPosition"] == lane)
        & (gold_cs_table["ChampionName"] == my_champion)
        & (gold_cs_table["ChampionName_opp"] == enemy_champion)
    ]

    if win_row.empty:
        return None

    report = {
        "my_champion": my_champion,
        "enemy_champion": enemy_champion,
        "lane": lane,
        "games": int(win_row.iloc[0]["games"]),
        "win_rate": float(win_row.iloc[0]["win_rate"]),
        "confidence_score": float(win_row.iloc[0]["confidence_score"]),
        "has_enough_data": bool(win_row.iloc[0]["has_enough_data"]),
    }

    if not gold_cs_row.empty:
        report["expected_gold_diff"] = float(gold_cs_row.iloc[0]["expected_gold_diff"])
        report["expected_cs_diff"] = float(gold_cs_row.iloc[0]["expected_cs_diff"])

    item_rune = item_rune_lookup.get(key)
    if item_rune:
        report["top_items"] = [
            {"name": item_names.get(it["item_id"], f"ID:{it['item_id']}"), "pick_rate": it["pick_rate"]}
            for it in item_rune["top_items"]
        ]
        combo = item_rune["top_rune_combo"]
        report["rune_combo"] = {
            "perks": {slot: perk_names.get(pid, f"ID:{pid}") for slot, pid in combo["perk_ids"].items()},
            "pick_rate": combo["pick_rate"],
        }
        report["build_sample_size"] = item_rune["win_games_used"]
    else:
        report["top_items"] = None
        report["rune_combo"] = None

    return report
