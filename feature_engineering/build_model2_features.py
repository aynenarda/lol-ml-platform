"""Model 2+3+4 icin final feature store'lari uretir: matchup intelligence
(gold/cs farki) tablosu + item/rune onerileri.

Item/rune onerisi icin katman 1 (matchup'a ozel) ve katman 3 (genel
build) verileri burada onceden hesaplanip kaydediliyor. Katman 2 (rakip
ozellik-profiline gore k-NN benzerlik genellemesi) ONCEDEN HESAPLANMIYOR
- prediction_engine'de, sorgu aninda, katman 1 verisi (item_rune_recs)
uzerinden canli hesaplaniyor (bkz. model2_matchup_intelligence.py).
"""

import json

from feature_engineering.matchups import build_matchups
from feature_engineering.matchup_intelligence import (
    compute_gold_cs_diff,
    compute_item_rune_recommendations,
    compute_champion_general_build,
)
from feature_engineering.external_data import load_item_metadata


def build_model2_data(clean_long_df):
    matchups = build_matchups(clean_long_df)
    item_metadata = load_item_metadata()

    gold_cs_table = compute_gold_cs_diff(matchups)
    item_rune_recs = compute_item_rune_recommendations(matchups, item_metadata)
    general_build = compute_champion_general_build(matchups, item_metadata)

    return gold_cs_table, item_rune_recs, general_build


def save_model2_data(gold_cs_table, item_rune_recs, general_build,
                      gold_cs_path="data/processed/model2_gold_cs.parquet",
                      item_rune_path="data/processed/model2_item_rune.json",
                      general_build_path="data/processed/model2_general_build.json"):
    gold_cs_table.to_parquet(gold_cs_path, index=False)

    with open(item_rune_path, "w", encoding="utf-8") as f:
        json.dump(item_rune_recs, f)

    general_serializable = {
        f"{lane}|{champ}": data for (lane, champ), data in general_build.items()
    }
    with open(general_build_path, "w", encoding="utf-8") as f:
        json.dump(general_serializable, f)
