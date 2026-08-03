"""Model 2+3+4 icin final feature store'lari uretir: matchup intelligence
(gold/cs farki) tablosu + kademeli item/rune onerileri.

Kademeler (spesifikten genele):
1. Bu matchup'a ozel (MIN_GAMES_FOR_BUILD esigi asilirsa)
2. Cizme icin: rakibin hasar tipine (fiziksel/buyu/karisik) gore genelleme
3. Son care: sampiyonun rakipten bagimsiz genel build'i
"""

import json

from feature_engineering.matchups import build_matchups
from feature_engineering.matchup_intelligence import (
    compute_gold_cs_diff,
    compute_item_rune_recommendations,
    compute_boots_by_damage_type,
    compute_champion_general_build,
)
from feature_engineering.external_data import load_item_metadata, load_champion_damage_types


def build_model2_data(clean_long_df):
    matchups = build_matchups(clean_long_df)
    item_metadata = load_item_metadata()
    damage_types = load_champion_damage_types()

    gold_cs_table = compute_gold_cs_diff(matchups)
    item_rune_recs = compute_item_rune_recommendations(matchups, item_metadata)
    boots_by_damage_type = compute_boots_by_damage_type(matchups, item_metadata, damage_types)
    general_build = compute_champion_general_build(matchups, item_metadata)

    return gold_cs_table, item_rune_recs, boots_by_damage_type, general_build, damage_types


def save_model2_data(gold_cs_table, item_rune_recs, boots_by_damage_type, general_build, damage_types,
                      gold_cs_path="data/processed/model2_gold_cs.parquet",
                      item_rune_path="data/processed/model2_item_rune.json",
                      boots_fallback_path="data/processed/model2_boots_by_damage_type.json",
                      general_build_path="data/processed/model2_general_build.json"):
    gold_cs_table.to_parquet(gold_cs_path, index=False)

    with open(item_rune_path, "w", encoding="utf-8") as f:
        json.dump(item_rune_recs, f)

    # Tuple key'leri JSON'a uygun stringe ceviriyoruz ("lane|champ|dmg_type").
    boots_serializable = {
        f"{lane}|{champ}|{dmg}": boots for (lane, champ, dmg), boots in boots_by_damage_type.items()
    }
    with open(boots_fallback_path, "w", encoding="utf-8") as f:
        json.dump(boots_serializable, f)

    general_serializable = {
        f"{lane}|{champ}": data for (lane, champ), data in general_build.items()
    }
    with open(general_build_path, "w", encoding="utf-8") as f:
        json.dump(general_serializable, f)
