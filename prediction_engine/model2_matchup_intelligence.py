"""Model 2+3+4: Matchup Intelligence + Item + Rune raporu.

Girdi: benim sampiyonum + rakip sampiyon + lane (Model 1'in tersine,
burada rakibi DEGIL, kendi secimimi de biliyoruz).
Cikti: win rate, beklenen gold/cs farki, item onerisi, rune onerisi -
hepsi bizim mac verimizden, statik veri sadece isimlendirme icin.

Item/rune onerisi KADEMELI: bu matchup'a ozel veri yoksa/azsa, once
cizme icin rakibin hasar tipine gore genelleme, sonra sampiyonun genel
build'ine dusuyor - Model 1'deki "yetersiz veri -> ogrenilen modele
dus" mantiginin ayni prensiple burada uygulanmis hali."""

import json

import pandas as pd

from feature_engineering.external_data import load_item_names, load_perk_names, load_champion_damage_types


def load_model2_data():
    model1_table = pd.read_parquet("data/processed/model1_counter_features.parquet")
    gold_cs_table = pd.read_parquet("data/processed/model2_gold_cs.parquet")

    with open("data/processed/model2_item_rune.json", encoding="utf-8") as f:
        item_rune_list = json.load(f)
    item_rune_lookup = {
        (r["TeamPosition"], r["ChampionName"], r["ChampionName_opp"]): r
        for r in item_rune_list
    }

    with open("data/processed/model2_boots_by_damage_type.json", encoding="utf-8") as f:
        boots_raw = json.load(f)
    boots_by_damage_type = {tuple(k.split("|")): v for k, v in boots_raw.items()}

    with open("data/processed/model2_general_build.json", encoding="utf-8") as f:
        general_raw = json.load(f)
    general_build = {tuple(k.split("|")): v for k, v in general_raw.items()}

    item_names = load_item_names()
    perk_names = load_perk_names()
    damage_types = load_champion_damage_types()

    return model1_table, gold_cs_table, item_rune_lookup, boots_by_damage_type, general_build, item_names, perk_names, damage_types


def _translate_items(item_list, item_names):
    return [
        {"name": item_names.get(it["item_id"], f"ID:{it['item_id']}"), "pick_rate": it["pick_rate"]}
        for it in item_list
    ]


def _translate_rune_combo(combo, perk_names):
    return {
        "perks": {slot: perk_names.get(pid, f"ID:{pid}") for slot, pid in combo["perk_ids"].items()},
        "pick_rate": combo["pick_rate"],
    }


def get_matchup_report(my_champion, enemy_champion, lane,
                        model1_table, gold_cs_table, item_rune_lookup, boots_by_damage_type,
                        general_build, item_names, perk_names, damage_types):
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
    }

    if not gold_cs_row.empty:
        report["expected_gold_diff"] = float(gold_cs_row.iloc[0]["expected_gold_diff"])
        report["expected_cs_diff"] = float(gold_cs_row.iloc[0]["expected_cs_diff"])

    # --- Katman 1: bu matchup'a ozel veri ---
    item_rune = item_rune_lookup.get((lane, my_champion, enemy_champion))
    if item_rune:
        report["build_source"] = "matchup_specific"
        report["build_sample_size"] = item_rune["win_games_used"]
        report["top_boots"] = _translate_items(item_rune["top_boots"], item_names)
        report["top_core_items"] = _translate_items(item_rune["top_core_items"], item_names)
        report["rune_combo"] = _translate_rune_combo(item_rune["top_rune_combo"], perk_names)
        return report

    # --- Katman 2/3: fallback - once genel build'i dene ---
    general = general_build.get((lane, my_champion))
    if general is None:
        report["build_source"] = None
        report["top_boots"] = None
        report["top_core_items"] = None
        report["rune_combo"] = None
        return report

    report["build_source"] = "general_fallback"
    report["build_sample_size"] = general["win_games_used"]
    report["top_core_items"] = _translate_items(general["top_core_items"], item_names)
    report["rune_combo"] = _translate_rune_combo(general["top_rune_combo"], perk_names)

    # Cizme icin, genel build'ten once rakibin hasar tipine gore
    # daha isabetli bir fallback deniyoruz.
    opp_damage_type = damage_types.get(enemy_champion)
    boots_key = (lane, my_champion, opp_damage_type) if opp_damage_type else None
    boots_by_dmg = boots_by_damage_type.get(boots_key) if boots_key else None

    if boots_by_dmg:
        report["top_boots"] = _translate_items([boots_by_dmg], item_names)
        report["boots_source"] = "damage_type_fallback"
    elif general["top_boots"]:
        report["top_boots"] = _translate_items(general["top_boots"], item_names)
        report["boots_source"] = "general_fallback"
    else:
        report["top_boots"] = None
        report["boots_source"] = None

    return report
