"""Model 1 (Counter Pick) ve Model 2 (Matchup Intelligence)'i HTTP
uzerinden sunan FastAPI backend'i. Frontend, bu endpoint'leri fetch()
ile cagirip sonuclari gosteriyor.

Tasarim: butun agir veri (parquet/json/torch modelleri) sadece app
baslarken BIR KEZ yukleniyor (bkz. lifespan) - her istekte diskten
tekrar okumuyoruz.
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from feature_engineering.external_data import load_champion_icons, load_item_icons, load_perk_icons
from prediction_engine.model1_counter_pick import recommend_counters
from prediction_engine.model2_matchup_intelligence import load_model2_data, get_matchup_report
from prediction_engine.learned_fallback import predict_counters

LANE_ALIASES = {
    "top": "TOP",
    "jungle": "JUNGLE", "jg": "JUNGLE",
    "middle": "MIDDLE", "mid": "MIDDLE",
    "bottom": "BOTTOM", "bot": "BOTTOM", "adc": "BOTTOM",
    "utility": "UTILITY", "support": "UTILITY", "sup": "UTILITY",
}

RUNE_SLOT_LABELS = {
    "PrimaryStylePerk1": "Anahtar Rün (Keystone)",
    "PrimaryStylePerk2": "Ana Yol - 1. Rün",
    "PrimaryStylePerk3": "Ana Yol - 2. Rün",
    "PrimaryStylePerk4": "Ana Yol - 3. Rün",
    "SubStylePerk1": "Yardımcı Yol - 1. Rün",
    "SubStylePerk2": "Yardımcı Yol - 2. Rün",
}

BUILD_SOURCE_LABELS = {
    "matchup_specific": "Bu eşleşmeye özel veri",
    "similarity_fallback": "Rakibin özellik profiline benzer rakiplerden genelleme",
    "stat_based_fallback": "Yetersiz eşleşme verisi - rakibin stat profiline göre önceliklendirilmiş genel build",
    "general_fallback": "Yetersiz veri - şampiyonun genel build'i kullanıldı",
}

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Veri yukleniyor...")
    state["model1_table"] = pd.read_parquet("data/processed/model1_counter_features.parquet")

    (model1_table, gold_cs_table, item_rune_lookup, general_build,
     item_names, perk_names, attributes, item_metadata, damage_types) = load_model2_data()
    state["gold_cs_table"] = gold_cs_table
    state["item_rune_lookup"] = item_rune_lookup
    state["general_build"] = general_build
    state["item_names"] = item_names
    state["perk_names"] = perk_names
    state["attributes"] = attributes
    state["item_metadata"] = item_metadata
    state["damage_types"] = damage_types

    state["icons"] = load_champion_icons()
    state["item_icons"] = load_item_icons()
    state["perk_icons"] = load_perk_icons()

    all_champions = sorted(set(model1_table["ChampionName"]) | set(model1_table["ChampionName_opp"]))
    state["all_champions"] = all_champions

    print(f"Hazir. {len(all_champions)} sampiyon yuklendi.")
    yield
    state.clear()


app = FastAPI(title="LoL ML Platform API", lifespan=lifespan)


def _normalize_lane(raw_lane: str) -> str:
    lane = LANE_ALIASES.get(raw_lane.strip().lower())
    if lane is None:
        raise HTTPException(status_code=400, detail=f"Geçersiz lane: '{raw_lane}'")
    return lane


def _champion_or_404(name: str) -> str:
    if name not in state["all_champions"]:
        raise HTTPException(status_code=404, detail=f"Şampiyon bulunamadı: '{name}'")
    return name


@app.get("/api/champions")
def list_champions():
    return [
        {"name": champ, "icon_url": state["icons"].get(champ)}
        for champ in state["all_champions"]
    ]


@app.get("/api/counters")
def get_counters(enemy_champion: str, lane: str, top_n: int = 5):
    enemy_champion = _champion_or_404(enemy_champion)
    lane = _normalize_lane(lane)

    candidates = recommend_counters(state["model1_table"], enemy_champion, lane, top_n=top_n)

    if not candidates.empty:
        results = [
            {
                "champion": row.ChampionName,
                "icon_url": state["icons"].get(row.ChampionName),
                "win_rate": row.win_rate,
                "confidence_score": row.confidence_score,
                "games": int(row.games),
                "difficulty_label": row.difficulty_label,
                "explanation": row.explanation,
            }
            for row in candidates.itertuples()
        ]
        return {"source": "matchup_specific", "results": results}

    learned = predict_counters(enemy_champion, lane, top_n=top_n)
    results = [
        {
            "champion": champ,
            "icon_url": state["icons"].get(champ),
            "win_rate": prob,
            "confidence_score": None,
            "games": 0,
            "difficulty_label": None,
            "explanation": "Doğrudan maç verisi yok - Blade & Chest modelinin tahmini.",
        }
        for champ, prob in learned
    ]
    return {"source": "learned_model_fallback", "results": results}


@app.get("/api/matchup")
def get_matchup(my_champion: str, enemy_champion: str, lane: str):
    my_champion = _champion_or_404(my_champion)
    enemy_champion = _champion_or_404(enemy_champion)
    lane = _normalize_lane(lane)

    report = get_matchup_report(
        my_champion, enemy_champion, lane,
        state["model1_table"], state["gold_cs_table"], state["item_rune_lookup"],
        state["general_build"], state["item_names"], state["perk_names"],
        state["attributes"], state["item_metadata"], state["damage_types"],
    )
    if report is None:
        raise HTTPException(status_code=404, detail="Bu eşleşme için veri bulunamadı.")

    report["my_champion_icon"] = state["icons"].get(my_champion)
    report["enemy_champion_icon"] = state["icons"].get(enemy_champion)

    if report.get("build_source"):
        report["build_source_label"] = BUILD_SOURCE_LABELS.get(report["build_source"], report["build_source"])

    for item_list_key in ("top_boots", "top_core_items"):
        if report.get(item_list_key):
            for item in report[item_list_key]:
                item["icon_url"] = state["item_icons"].get(item["item_id"])

    if report.get("rune_combo"):
        report["rune_combo"]["labeled_perks"] = [
            {
                "slot_label": RUNE_SLOT_LABELS[slot],
                "name": name,
                "icon_url": state["perk_icons"].get(report["rune_combo"]["perk_ids"][slot]),
            }
            for slot, name in report["rune_combo"]["perks"].items()
        ]

    return report


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
