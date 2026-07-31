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

FIELDS = ["ChampionName", "TeamPosition", "Win", "Kills", "Deaths", "Assists", "GoldEarned"]

def extract_participant(df, i):
    """Wide formattaki participant{i} kolonlarini long formata cevirir."""
    source_cols = [f"participant{i}{field}" for field in FIELDS]
    part = df[["matchId"] + source_cols].copy()

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
all_needed_cols = ["matchId"] + [
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

# --- Adim 4: long formati parquet olarak diske kaydet ---

output_path = "data/processed/matches_long.parquet"
full_long.to_parquet(output_path, index=False)
print()
print(f"Kaydedildi: {output_path}")
