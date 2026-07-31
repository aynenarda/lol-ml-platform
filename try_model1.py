"""Model 1'i hizlica denemek icin interaktif script.

run_pipeline.py'yi tekrar calistirmaya gerek yok - zaten kaydedilmis
feature store'u (data/processed/model1_counter_features.parquet) okuyup
istedigin sampiyon/lane icin sorgu yapar.
"""

import pandas as pd

from prediction_engine.model1_counter_pick import recommend_counters

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

VALID_LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

model1_table = pd.read_parquet("data/processed/model1_counter_features.parquet")

print("Model 1: Counter Pick Recommendation Engine")
print(f"Gecerli lane degerleri: {', '.join(VALID_LANES)}")
print("Sampiyon adlari Riot'un ic formatinda olmali (bosluksuz): "
      "orn. Renekton, KSante, DrMundo, AurelionSol, MonkeyKing (Wukong)")
print("Cikmak icin bos birak ve Enter'a bas.")
print()

while True:
    enemy = input("Rakip sampiyon: ").strip()
    if not enemy:
        break

    lane = input("Lane (TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY): ").strip().upper()
    if lane not in VALID_LANES:
        print(f"Gecersiz lane: '{lane}'. Su degerlerden biri olmali: {VALID_LANES}\n")
        continue

    result = recommend_counters(model1_table, enemy, lane, top_n=5)

    if result.empty:
        print(f"'{enemy}' ({lane}) icin yeterli veri bulunamadi. "
              "Sampiyon adini/lane'i kontrol et.\n")
        continue

    print()
    for _, row in result.iterrows():
        print(f"#{row.ChampionName}  win_rate=%{row.win_rate*100:.1f}  "
              f"confidence=%{row.confidence_score*100:.1f}  "
              f"sample_size={row.games}  difficulty={row.difficulty_label}  "
              f"power_spike={row.power_spike}  risk_score={row.risk_score:.3f}")
        print(f"   {row.explanation}")
        print()
