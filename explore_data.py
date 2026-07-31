import pandas as pd

# Once sadece ilk 5 satiri oku - 743 MB'lik dosyayi tamamen belleğe almadan
# once yapisini anlamak icin. nrows parametresi pandas'a "sadece bu kadar
# satir oku, gerisini okuma" der.
preview = pd.read_csv("data/raw/matchData.csv", nrows=5)

print("KOLON SAYISI:", len(preview.columns))
print()
print("KOLON ISIMLERI:")
for col in preview.columns:
    print(" -", col)
print()
print("ILK 5 SATIR (transpoze - okumasi kolay olsun diye):")
print(preview.T)
print()
print("VERI TIPLERI (bu ornek 5 satirdan pandas'in tahmini):")
print(preview.dtypes)

# --- Adim 2: ayni mantigi 10 participant icin tekrarla, sonra birlestir ---

FIELDS = ["ChampionName", "TeamPosition", "Win", "Kills", "Deaths", "Assists", "GoldEarned", "GameEndedInEarlySurrender"]
MATCH_LEVEL_FIELDS = ["gameMode", "mapId", "gameDuration"]

def extract_participant(df, i):
    """Wide formattaki participant{i} kolonlarini long formata cevirir."""
    source_cols = [f"participant{i}{field}" for field in FIELDS]
    part = df[["matchId"] + MATCH_LEVEL_FIELDS + source_cols].copy()

    rename_map = {f"participant{i}{field}": field for field in FIELDS}
    part = part.rename(columns=rename_map)

    part["participantIndex"] = i
    return part

# Her participant icin bir DataFrame uret, listede topla
long_pieces = [extract_participant(preview, i) for i in range(10)]

# 10 parcayi alt alta birlestir (ignore_index=True: eski index'leri atip 0'dan yeniden numarala)
long_df = pd.concat(long_pieces, ignore_index=True)

print()
print(f"Eski (wide) satir sayisi: {len(preview)}")
print(f"Yeni (long) satir sayisi: {len(long_df)}  (beklenen: {len(preview)} x 10 = {len(preview) * 10})")
print()
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(long_df.sort_values(["matchId", "participantIndex"]).head(10))

# --- Adim 3: ayni islemi TUM dosyaya (743MB, 100k+ satir) uygula ---

# usecols: sadece ihtiyacimiz olan kolonlari oku. Pandas bu listeyi
# dosyayi okumadan once kontrol eder, listede olmayan kolonlari hic
# parse etmez - bu yuzden 1770 kolonun sadece ~71'ini gercekten okuruz.
all_needed_cols = ["matchId"] + MATCH_LEVEL_FIELDS + [
    f"participant{i}{field}" for i in range(10) for field in FIELDS
]

print()
print(f"Toplam okunacak kolon sayisi: {len(all_needed_cols)} (1770 yerine)")
print("Tum dosya okunuyor, birkac saniye surebilir...")

full_wide = pd.read_csv("data/raw/matchData.csv", usecols=all_needed_cols)

full_long_pieces = [extract_participant(full_wide, i) for i in range(10)]
full_long = pd.concat(full_long_pieces, ignore_index=True)

print()
print(f"Wide format satir sayisi (maç sayisi): {len(full_wide)}")
print(f"Long format satir sayisi (oyuncu-maç sayisi): {len(full_long)}")
print(f"Kontrol: {len(full_wide)} x 10 = {len(full_wide) * 10}")
print()
print("Bellek kullanimi (MB):")
print(f"  full_wide: {full_wide.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"  full_long: {full_long.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# --- Adim 4: Data Cleaning tanisi (diagnostic) ---
# Kod yazmadan once hipotezlerimizi gercek veride sayiyoruz.

print()
print("=== TANI 1: gameMode dagilimi ===")
print(full_wide["gameMode"].value_counts())

print()
print("=== TANI 2: mapId dagilimi ===")
print(full_wide["mapId"].value_counts())

print()
print("=== TANI 3: gameDuration (saniye) istatistikleri ===")
print(full_wide["gameDuration"].describe())
print("300 saniyeden (5 dk) kisa mac sayisi:", (full_wide["gameDuration"] < 300).sum())

print()
print("=== TANI 4: erken teslim (early surrender) sayisi ===")
early_surrender_col = full_long["GameEndedInEarlySurrender"]
print(early_surrender_col.value_counts())

print()
print("=== TANI 5: TeamPosition bos/gecersiz deger sayisi ===")
print(full_long["TeamPosition"].value_counts(dropna=False))

print()
print("=== TANI 6: duplicate matchId kontrolu ===")
print("Toplam wide satir:", len(full_wide))
print("Essiz matchId sayisi:", full_wide["matchId"].nunique())

# --- Adim 5: gecersiz maclari tespit et ve temizle ---

short_game_ids = set(full_wide.loc[full_wide["gameDuration"] < 300, "matchId"])
early_surrender_ids = set(full_long.loc[full_long["GameEndedInEarlySurrender"], "matchId"])
missing_position_ids = set(full_long.loc[full_long["TeamPosition"].isna(), "matchId"])

invalid_ids = short_game_ids | early_surrender_ids | missing_position_ids

print()
print("=== TEMIZLEME OZETI ===")
print(f"Kisa mac (< 5 dk):          {len(short_game_ids)} mac")
print(f"Erken teslim:               {len(early_surrender_ids)} mac")
print(f"Eksik TeamPosition:         {len(missing_position_ids)} mac")
print(f"Toplam gecersiz (birlesim): {len(invalid_ids)} mac")
print(f"Toplam mac sayisi:          {full_wide['matchId'].nunique()}")

clean_long = full_long[~full_long["matchId"].isin(invalid_ids)].copy()

print()
print(f"Temizlik oncesi satir sayisi: {len(full_long)}")
print(f"Temizlik sonrasi satir sayisi: {len(clean_long)}")
print(f"Kalan mac sayisi: {clean_long['matchId'].nunique()}")

# GameEndedInEarlySurrender artik hep False (o maclari zaten cikardik) - gereksiz kolon
clean_long = clean_long.drop(columns=["GameEndedInEarlySurrender"])

clean_output_path = "data/processed/matches_clean.parquet"
clean_long.to_parquet(clean_output_path, index=False)
print()
print(f"Kaydedildi: {clean_output_path}")
