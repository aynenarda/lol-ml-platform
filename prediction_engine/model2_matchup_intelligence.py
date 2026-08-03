"""Model 2+3+4: Matchup Intelligence + Item + Rune raporu.

Girdi: benim sampiyonum + rakip sampiyon + lane (Model 1'in tersine,
burada rakibi DEGIL, kendi secimimi de biliyoruz).
Cikti: win rate, beklenen gold/cs farki, item onerisi, rune onerisi -
hepsi bizim mac verimizden, statik veri sadece isimlendirme icin.

Item/rune onerisi 3 KATMANLI:
1. Bu matchup'a ozel veri (yeterliyse)
2. k-NN benzerlik agirlikli genelleme: rakibin ozellik profiline
   (hasar/dayaniklilik/kontrol/hareketlilik/yardim) EN YAKIN, daha once
   karsilasilmis rakiplerin build'lerinden agirlikli ortalama - "hasar
   tipi" gibi kaba bir etiketten CoK daha zengin bir benzerlik olcusu.
3. Son care: k-NN icin de HIC benzer rakip bulunamadiginda (secilen
   sampiyon o lane'de cok az/dagilmis oynanmis oldugu icin), sampiyonun
   gozlemlenmis (rakipten bagimsiz) genel item/cizme havuzunu, rakibin
   STAT PROFILINE (hasar tipi, dayaniklilik, kontrol, hareketlilik) gore
   YENIDEN SIRALAR - yine sadece bu sampiyonun gercekten aldigi item'lar
   arasindan seciyor, hicbir item uydurulmuyor (bkz. _stat_based_build).
"""

import json
import math
from collections import Counter

import pandas as pd

from feature_engineering.external_data import (
    load_item_names,
    load_perk_names,
    load_champion_attributes,
    load_champion_damage_types,
    load_item_metadata,
)
from prediction_engine.learned_fallback import predict_single_matchup

K_NEIGHBORS = 5

# Rakip stat profiline gore item kategorisi onceliklendirme esikleri.
# Meraki attributeRatings 0-3 olcek; 3 = o ozellikte belirgin sekilde guclu.
HIGH_ATTR_THRESHOLD = 3
MODERATE_ATTR_THRESHOLD = 2


def load_model2_data():
    model1_table = pd.read_parquet("data/processed/model1_counter_features.parquet")
    gold_cs_table = pd.read_parquet("data/processed/model2_gold_cs.parquet")

    with open("data/processed/model2_item_rune.json", encoding="utf-8") as f:
        item_rune_list = json.load(f)
    item_rune_lookup = {
        (r["TeamPosition"], r["ChampionName"], r["ChampionName_opp"]): r
        for r in item_rune_list
    }

    with open("data/processed/model2_general_build.json", encoding="utf-8") as f:
        general_raw = json.load(f)
    general_build = {tuple(k.split("|")): v for k, v in general_raw.items()}

    item_names = load_item_names()
    perk_names = load_perk_names()
    attributes = load_champion_attributes()
    item_metadata = load_item_metadata()
    damage_types = load_champion_damage_types()

    return (model1_table, gold_cs_table, item_rune_lookup, general_build,
            item_names, perk_names, attributes, item_metadata, damage_types)


def _translate_items(item_list, item_names):
    return [
        {
            "item_id": it["item_id"],
            "name": item_names.get(it["item_id"], f"ID:{it['item_id']}"),
            "pick_rate": it["pick_rate"],
        }
        for it in item_list
    ]


def _translate_rune_combo(combo, perk_names):
    return {
        "perks": {slot: perk_names.get(pid, f"ID:{pid}") for slot, pid in combo["perk_ids"].items()},
        "perk_ids": combo["perk_ids"],
        "pick_rate": combo["pick_rate"],
    }


def _euclidean_distance(vec_a, vec_b):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


def _similarity_weighted_build(my_champion, enemy_champion, lane, item_rune_lookup, attributes, top_n_core=5):
    """Rakibin ozellik profiline en yakin, daha once karsilasilmis
    rakiplerin build'lerini benzerlige gore agirlikli birlestirir."""
    enemy_vec = attributes.get(enemy_champion)
    if enemy_vec is None:
        return None

    candidates = []
    for (l, champ, opp), entry in item_rune_lookup.items():
        if l != lane or champ != my_champion or opp == enemy_champion:
            continue
        opp_vec = attributes.get(opp)
        if opp_vec is None:
            continue
        distance = _euclidean_distance(enemy_vec, opp_vec)
        similarity = 1.0 / (1.0 + distance)
        candidates.append((similarity, opp, entry))

    if not candidates:
        return None

    candidates.sort(key=lambda c: -c[0])
    neighbors = candidates[:K_NEIGHBORS]
    total_similarity = sum(sim for sim, _, _ in neighbors)

    boots_scores = Counter()
    core_scores = Counter()
    rune_scores = Counter()

    for similarity, _, entry in neighbors:
        weight = similarity / total_similarity
        for it in entry["top_boots"]:
            boots_scores[it["item_id"]] += weight * it["pick_rate"]
        for it in entry["top_core_items"]:
            core_scores[it["item_id"]] += weight * it["pick_rate"]
        combo_key = tuple(sorted(entry["top_rune_combo"]["perk_ids"].items()))
        rune_scores[combo_key] += weight * entry["top_rune_combo"]["pick_rate"]

    top_boots = [{"item_id": iid, "pick_rate": score} for iid, score in boots_scores.most_common(1)]
    top_core = [{"item_id": iid, "pick_rate": score} for iid, score in core_scores.most_common(top_n_core)]

    top_rune_combo = None
    if rune_scores:
        combo_key, score = rune_scores.most_common(1)[0]
        top_rune_combo = {"perk_ids": dict(combo_key), "pick_rate": score}

    return {
        "top_boots": top_boots,
        "top_core_items": top_core,
        "top_rune_combo": top_rune_combo,
        "neighbors_used": [opp for _, opp, _ in neighbors],
    }


def _item_category_score(categories, is_boots, enemy_vec, enemy_damage_type):
    """Bir item'in rakibe karsi ne kadar 'mantikli' oldugunu, rakibin stat
    profiline gore bir carpan olarak doner (1.0 = notr, notr'un uzerinde
    daha oncelikli). Tamamen Community Dragon'un gercek item
    kategorilerine (Armor, SpellBlock, Tenacity, ArmorPenetration,
    MagicPenetration, Slow) ve Meraki'nin rakip stat profiline dayanir -
    hicbir sayisal deger uydurulmuyor, sadece bilinen oyun mantigi
    (fiziksel hasara karsi zirh, agir CC'ye karsi tenacity vb.)
    kategori eslesmesi olarak kodlanmis."""
    damage, toughness, control, mobility, utility = enemy_vec
    score = 1.0
    reasons = []

    if enemy_damage_type == "kPhysical" and "Armor" in categories:
        score *= 1.5
        reasons.append("rakip agirlikli fiziksel hasar veriyor -> zirh")
    elif enemy_damage_type == "kMagic" and "SpellBlock" in categories:
        score *= 1.5
        reasons.append("rakip agirlikli buyu hasari veriyor -> buyu direnci")
    elif enemy_damage_type == "kMixed" and ("Armor" in categories or "SpellBlock" in categories):
        score *= 1.2
        reasons.append("rakip karma hasar veriyor -> karma savunma")

    if toughness >= HIGH_ATTR_THRESHOLD and ("ArmorPenetration" in categories or "MagicPenetration" in categories):
        score *= 1.4
        reasons.append("rakip dayanikli (yuksek can/direnc) -> nufuz/delme")

    if mobility >= HIGH_ATTR_THRESHOLD and "Slow" in categories:
        score *= 1.3
        reasons.append("rakip hareketliligi yuksek -> yavaslatma")

    if is_boots and control >= MODERATE_ATTR_THRESHOLD and "Tenacity" in categories:
        score *= 2.0
        reasons.append("rakipte belirgin CC var -> tenacity cizme")

    return score, reasons


def _stat_based_build(my_champion, lane, enemy_champion, general_build,
                       item_metadata, attributes, damage_types, top_n_core=5):
    """Katman 3'un son adimi: sampiyonun GOZLEMLENMIS (rakipten bagimsiz)
    item/cizme havuzunu, rakibin somut stat profiline (hasar tipi,
    dayaniklilik, hareketlilik, kontrol) gore yeniden siralar. Sadece bu
    sampiyonun gercekten aldigi item'lar arasindan seciyor - hicbir item
    Riot'un "onerilen build"inden ya da genel bir listeden gelmiyor."""
    entry = general_build.get((lane, my_champion))
    if entry is None:
        return None

    enemy_vec = attributes.get(enemy_champion)
    enemy_damage_type = damage_types.get(enemy_champion)
    if enemy_vec is None:
        return None

    reasoning = []

    def _rerank(pool, is_boots):
        scored = []
        for it in pool:
            categories = item_metadata.get(it["item_id"], {}).get("categories", [])
            multiplier, reasons = _item_category_score(categories, is_boots, enemy_vec, enemy_damage_type)
            scored.append((it["pick_rate"] * multiplier, it, reasons))
        scored.sort(key=lambda t: -t[0])
        for _, _, reasons in scored:
            for reason in reasons:
                if reason not in reasoning:
                    reasoning.append(reason)
        return [it for _, it, _ in scored]

    ranked_boots = _rerank(entry["top_boots"], is_boots=True)
    ranked_core = _rerank(entry["top_core_items"], is_boots=False)

    return {
        "top_boots": ranked_boots[:1],
        "top_core_items": ranked_core[:top_n_core],
        "top_rune_combo": entry["top_rune_combo"],
        "win_games_used": entry["win_games_used"],
        "reasoning": reasoning,
    }


def get_matchup_report(my_champion, enemy_champion, lane,
                        model1_table, gold_cs_table, item_rune_lookup, general_build,
                        item_names, perk_names, attributes, item_metadata, damage_types):
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
        # Model 1'in sayma tablosunda bu cift hic yok (0 mac) - Model 1'in
        # kendi ogrenilen model fallback'ine (Blade & Chest) dusuyoruz,
        # boylece rapor tamamen bos donmek yerine en azindan bir tahmin
        # (ve item k-NN fallback'i) uretebiliyor.
        learned_prob = predict_single_matchup(my_champion, enemy_champion, lane)
        if learned_prob is None:
            return None
        report = {
            "my_champion": my_champion,
            "enemy_champion": enemy_champion,
            "lane": lane,
            "games": 0,
            "win_rate": learned_prob,
            "confidence_score": None,
            "win_rate_source": "learned_model_fallback",
        }
    else:
        report = {
            "my_champion": my_champion,
            "enemy_champion": enemy_champion,
            "lane": lane,
            "games": int(win_row.iloc[0]["games"]),
            "win_rate": float(win_row.iloc[0]["win_rate"]),
            "confidence_score": float(win_row.iloc[0]["confidence_score"]),
            "win_rate_source": "matchup_specific",
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

    # --- Katman 2: ozellik-benzerligi agirlikli k-NN genellemesi ---
    similarity_build = _similarity_weighted_build(my_champion, enemy_champion, lane, item_rune_lookup, attributes)
    if similarity_build:
        report["build_source"] = "similarity_fallback"
        report["similar_opponents"] = similarity_build["neighbors_used"]
        report["top_boots"] = _translate_items(similarity_build["top_boots"], item_names)
        report["top_core_items"] = _translate_items(similarity_build["top_core_items"], item_names)
        report["rune_combo"] = (
            _translate_rune_combo(similarity_build["top_rune_combo"], perk_names)
            if similarity_build["top_rune_combo"] else None
        )
        return report

    # --- Katman 3: son care - rakibin stat profiline gore yeniden siralanmis
    # genel build (bkz. _stat_based_build) ---
    stat_build = _stat_based_build(my_champion, lane, enemy_champion, general_build,
                                    item_metadata, attributes, damage_types)
    if stat_build is not None:
        report["build_source"] = "stat_based_fallback"
        report["build_sample_size"] = stat_build["win_games_used"]
        report["stat_reasoning"] = stat_build["reasoning"]
        report["top_boots"] = _translate_items(stat_build["top_boots"], item_names) if stat_build["top_boots"] else None
        report["top_core_items"] = _translate_items(stat_build["top_core_items"], item_names)
        report["rune_combo"] = _translate_rune_combo(stat_build["top_rune_combo"], perk_names)
        return report

    # --- Son care: rakip stat verisi de yoksa (cok nadir), duz genel build ---
    general = general_build.get((lane, my_champion))
    if general is None:
        report["build_source"] = None
        report["top_boots"] = None
        report["top_core_items"] = None
        report["rune_combo"] = None
        return report

    report["build_source"] = "general_fallback"
    report["build_sample_size"] = general["win_games_used"]
    report["top_boots"] = _translate_items(general["top_boots"][:1], item_names) if general["top_boots"] else None
    report["top_core_items"] = _translate_items(general["top_core_items"][:5], item_names)
    report["rune_combo"] = _translate_rune_combo(general["top_rune_combo"], perk_names)

    return report
