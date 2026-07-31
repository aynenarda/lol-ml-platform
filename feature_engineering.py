import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_parquet("data/processed/matches_clean.parquet")
print("Girdi satir sayisi (oyuncu-mac):", len(df))
print(df.columns.tolist())

# --- Adim 1: self-join ile matchup ciftlerini uret ---
#
# Ayni tabloyu kendisiyle, matchId + TeamPosition uzerinden birlestiriyoruz.
# Bir mactaki ayni lane'de tam olarak 2 oyuncu vardir (biri her takimdan).
# merge, bu iki satiri "yan yana" koyar - ama kendisiyle de eslesir (satirin
# kendisi), o yuzden Win != Win_opp filtresiyle sadece GERCEK rakip ciftini
# birakiyoruz (kazanan-kaybeden her zaman farkli oldugu icin bu filtre
# ayni satirin kendisiyle eslesmesini de otomatik eler).
matchups = df.merge(
    df,
    on=["matchId", "TeamPosition"],
    suffixes=("", "_opp"),
)
matchups = matchups[matchups["Win"] != matchups["Win_opp"]]

print()
print("Self-join sonrasi satir sayisi:", len(matchups))
print("Beklenen: girdi satir sayisiyla ayni olmali (her oyuncu tam 1 rakiple eslesir)")
print()
print(matchups[["matchId", "TeamPosition", "ChampionName", "Win", "ChampionName_opp", "Win_opp"]].head(10))

# --- Adim 2: lane + matchup bazinda galibiyet oranini hesapla ---
#
# groupby ile her (TeamPosition, ChampionName, ChampionName_opp) ucglusu
# icin: kac mac oynanmis (games), kacinda kazanilmis (wins).
counter_stats = (
    matchups
    .groupby(["TeamPosition", "ChampionName", "ChampionName_opp"])
    .agg(games=("Win", "size"), wins=("Win", "sum"))
    .reset_index()
)
counter_stats["win_rate"] = counter_stats["wins"] / counter_stats["games"]

print()
print("Toplam matchup satiri (lane x champion x opponent):", len(counter_stats))
print()

# Ornek: Top lane'de Renekton'a karsi en cok oynanan / en yuksek win rate'li
# rakip sampiyonlar neler?
renekton_top = counter_stats[
    (counter_stats["TeamPosition"] == "TOP") & (counter_stats["ChampionName_opp"] == "Renekton")
].sort_values("games", ascending=False)

print("Top lane'de Renekton'a KARSI en cok oynanan sampiyonlar (benim acimdan win_rate):")
print(renekton_top.head(15).to_string(index=False))

# --- Adim 3: Wilson Score Interval ile guven araligi hesapla ---

import numpy as np

def wilson_lower_bound(wins, games, z=1.96):
    """Wilson skor araliginin ALT sinirini dondurur - kucuk orneklemi
    otomatik olarak cezalandiran, guven-ayarli bir tahmin."""
    n = games
    p_hat = wins / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / denominator
    return center - margin

counter_stats["confidence_score"] = wilson_lower_bound(counter_stats["wins"], counter_stats["games"])

print()
print("=== Karsilastirma: ham win_rate'e gore siralama vs confidence_score'a gore siralama ===")
renekton_top = counter_stats[
    (counter_stats["TeamPosition"] == "TOP") & (counter_stats["ChampionName_opp"] == "Renekton")
]

print()
print("--- Ham win_rate'e gore ilk 10 (dikkat: dusuk orneklemler one cikabilir) ---")
print(renekton_top.sort_values("win_rate", ascending=False).head(10).to_string(index=False))

print()
print("--- confidence_score'a (Wilson alt siniri) gore ilk 10 ---")
print(renekton_top.sort_values("confidence_score", ascending=False).head(10).to_string(index=False))

# --- Adim 4: minimum orneklem esigi ---
#
# Wilson alt siniri kucuk n'de dusuk cikma egiliminde olsa da, cok kucuk
# orneklemlerde (n=4, %100 win rate gibi) hala sansa dayali sonuclar
# one cikabiliyor. Bu yuzden ayrica MIN_GAMES esigi uyguluyoruz - bu
# esigin altindaki matchup'lari "oneri" listesine hic almiyoruz, ayri
# bir "yetersiz veri" kategorisinde tutuyoruz.
MIN_GAMES = 20

counter_stats["has_enough_data"] = counter_stats["games"] >= MIN_GAMES

print()
print(f"--- MIN_GAMES={MIN_GAMES} esigi sonrasi, confidence_score'a gore ilk 10 ---")
reliable = renekton_top[renekton_top["games"] >= MIN_GAMES]
print(reliable.sort_values("confidence_score", ascending=False).head(10).to_string(index=False))

# --- Adim 5: Model 1'in temel feature store'unu diske kaydet ---

output_path = "data/processed/counter_stats.parquet"
counter_stats.to_parquet(output_path, index=False)
print()
print(f"Kaydedildi: {output_path}  ({len(counter_stats)} satir)")
