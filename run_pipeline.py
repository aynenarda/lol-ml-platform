"""Pipeline'i uctan uca calistiran orkestratör script.

data_collector -> data_cleaner -> feature_engineering -> prediction_engine

Raporlama (print) islerinin burada olmasi bilincli bir tasarim tercihi:
modul fonksiyonlari saf kalir (girdi->cikti), boylece hem test edilebilir
hem de baska bir orkestrasyondan (orn. bir Backend API endpoint'i)
sessizce cagirilabilir.
"""

import pandas as pd

from data_collector.raw_loader import load_raw_wide
from data_cleaner.reshape import wide_to_long
from data_cleaner.clean import find_invalid_match_ids, clean_long_data
from feature_engineering.build_model1_features import build_model1_table
from feature_engineering.build_model2_features import build_model2_data, save_model2_data
from prediction_engine.model1_counter_pick import recommend_counters

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def main():
    print("1) Ham veri okunuyor...")
    wide_df = load_raw_wide()
    print(f"   {len(wide_df)} mac okundu.")

    print("2) Wide -> long donusumu...")
    long_df = wide_to_long(wide_df)
    print(f"   {len(long_df)} oyuncu-mac satiri uretildi.")

    print("3) Temizleme...")
    invalid_ids = find_invalid_match_ids(wide_df, long_df)
    clean_df = clean_long_data(long_df, invalid_ids)
    print(f"   {len(invalid_ids)} gecersiz mac cikarildi, {len(clean_df)} satir kaldi.")
    clean_df.to_parquet("data/processed/matches_clean.parquet", index=False)

    print("4) Model 1 feature store uretiliyor...")
    model1_table = build_model1_table(clean_df)
    model1_table.to_parquet("data/processed/model1_counter_features.parquet", index=False)
    print(f"   {len(model1_table)} satirlik feature store kaydedildi.")

    print()
    print("5) Ornek oneri: TOP lane'de Renekton'a karsi")
    result = recommend_counters(model1_table, "Renekton", "TOP", top_n=5)
    for _, row in result.iterrows():
        print(f"   #{row.ChampionName}  win_rate=%{row.win_rate*100:.1f}  "
              f"confidence=%{row.confidence_score*100:.1f}  difficulty={row.difficulty_label}")
        print(f"      {row.explanation}")

    print()
    print("6) Model 2 feature store'lari uretiliyor (gold/cs farki + item/rune onerileri)...")
    gold_cs_table, item_rune_recs, general_build = build_model2_data(clean_df)
    save_model2_data(gold_cs_table, item_rune_recs, general_build)
    print(f"   {len(gold_cs_table)} matchup icin gold/cs farki")
    print(f"   {len(item_rune_recs)} matchup icin ozel item/rune onerisi (katman 1)")
    print(f"   {len(general_build)} sampiyon icin genel build fallback'i (katman 3)")
    print("   (katman 2 - ozellik benzerligi k-NN - sorgu aninda canli hesaplanir)")


if __name__ == "__main__":
    main()
