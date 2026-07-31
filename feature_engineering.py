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

# --- Adim 6: Lane Pressure (matchup bazli) ---
#
# "Bu matchup'ta benim sampiyonum ne siklikla ilk kani aliyor?" - lane
# baskisinin somut, olculebilir bir gostergesi. FirstBloodKill sadece o
# oyunda ilk kani alan TEK oyuncuda True olur (mac bazinda), bu yuzden
# ortalamasi dogrudan "bu matchup'ta ilk kani alma orani" anlamina gelir.
lane_pressure = (
    matchups
    .groupby(["TeamPosition", "ChampionName", "ChampionName_opp"])
    .agg(
        first_blood_rate=("FirstBloodKill", "mean"),
        first_tower_rate=("FirstTowerKill", "mean"),
    )
    .reset_index()
)

counter_stats = counter_stats.merge(
    lane_pressure, on=["TeamPosition", "ChampionName", "ChampionName_opp"]
)

print()
print("Lane pressure ornegi (Jax vs Renekton, TOP):")
print(counter_stats[
    (counter_stats["TeamPosition"] == "TOP")
    & (counter_stats["ChampionName"] == "Jax")
    & (counter_stats["ChampionName_opp"] == "Renekton")
].to_string(index=False))

# --- Adim 7: Champion-level oyun tarzi ozellikleri (belirli rakipten bagimsiz) ---
#
# Bunlar "bu sampiyon genel olarak nasil oynaniyor" sorusuna cevap veriyor,
# spesifik bir matchup'a degil - o yuzden matchups degil df (matches_clean)
# uzerinde, (TeamPosition, ChampionName) bazinda hesapliyoruz. Boylece
# ornek boyutu cok daha buyuk (bir sampiyonun TUM maclari), daha guvenilir.

# Scaling / Power Spike: mac suresini kisa/orta/uzun olarak 3'e bolup, win
# rate'in sureyle nasil degistigine bakiyoruz. Win rate uzun maclarda
# yuksekse "gec oyun sampiyonu" (positive scaling_score), kisa maclarda
# yuksekse "erken oyun / snowball sampiyonu" (negative scaling_score).
df["duration_bucket"] = pd.qcut(df["gameDuration"], q=3, labels=["short", "medium", "long"])

duration_winrate = (
    df.groupby(["TeamPosition", "ChampionName", "duration_bucket"], observed=True)["Win"]
    .mean()
    .unstack("duration_bucket")
)
scaling = pd.DataFrame({
    "scaling_score": duration_winrate["long"] - duration_winrate["short"],
}).reset_index()

# Split Push: bina hasari, ayni lane icindeki ortalamaya orantili (lane'ler
# arasi dogal fark var - top laner zaten support'tan cok daha fazla bina
# hasari verir, o yuzden ham degeri degil, lane-ici oranini kullaniyoruz).
champion_style = (
    df.groupby(["TeamPosition", "ChampionName"])
    .agg(
        games_total=("Win", "size"),
        avg_building_damage=("DamageDealtToBuildings", "mean"),
        avg_assists=("Assists", "mean"),
        avg_kill_spree=("LargestKillingSpree", "mean"),
    )
    .reset_index()
)
lane_avg_building_damage = champion_style.groupby("TeamPosition")["avg_building_damage"].transform("mean")
champion_style["split_push_score"] = champion_style["avg_building_damage"] / lane_avg_building_damage

# Team Fight: assist orani, ayni sekilde lane-ici oranli
lane_avg_assists = champion_style.groupby("TeamPosition")["avg_assists"].transform("mean")
champion_style["team_fight_score"] = champion_style["avg_assists"] / lane_avg_assists

# Snowball Potential: en uzun oldurme serisi, lane-ici oranli
lane_avg_kill_spree = champion_style.groupby("TeamPosition")["avg_kill_spree"].transform("mean")
champion_style["snowball_score"] = champion_style["avg_kill_spree"] / lane_avg_kill_spree

champion_style = champion_style.merge(scaling, on=["TeamPosition", "ChampionName"])

print()
print("Champion-level ozellik ornegi (TOP lane, birkac sampiyon):")
print(champion_style[champion_style["TeamPosition"] == "TOP"]
      .sort_values("games_total", ascending=False)
      .head(10)
      .to_string(index=False))

# --- Adim 8: Difficulty Score (Riot'un resmi statik verisi, Meraki Analytics uzerinden) ---
#
# Meraki Analytics'in cdn'inde Riot'un kendi "attributeRatings" verisi var:
# damage/toughness/control/mobility/utility/difficulty (1-3 olcek: Kolay/Orta/Zor).
# Bu, mac istatistiklerinden turememis, Riot'un tasarim ekibinin belirledigi
# statik bir deger - o yuzden ayri bir kaynaktan aliyoruz.
import json

with open("data/external/champions_meraki.json", encoding="utf-8") as f:
    meraki_data = json.load(f)

# Bizim veride "FiddleSticks" yazarken meraki'de "Fiddlesticks" - tek istisna,
# elle duzeltiyoruz.
NAME_FIXES = {"FiddleSticks": "Fiddlesticks"}

def get_difficulty(champion_name):
    lookup_name = NAME_FIXES.get(champion_name, champion_name)
    champ = meraki_data.get(lookup_name)
    if champ is None:
        return None
    return champ["attributeRatings"]["difficulty"]

DIFFICULTY_LABELS = {1: "Kolay", 2: "Orta", 3: "Zor"}

champion_style["difficulty_tier"] = champion_style["ChampionName"].map(get_difficulty)
champion_style["difficulty_label"] = champion_style["difficulty_tier"].map(DIFFICULTY_LABELS)

print()
print("Eslenemeyen sampiyon sayisi (difficulty_tier bos):",
      champion_style["difficulty_tier"].isna().sum())

# --- Adim 9: hepsini birlestir + Risk Score + final Model 1 tablosu ---
#
# Risk Score: Wilson araliginin genisligi (ust sinir - alt sinir) - ne kadar
# genisse, o kadar az eminiz, yani "risk" o kadar yuksek. Bunun icin ust
# siniri da hesaplamamiz lazim (simdiye kadar sadece alt siniri kullandik).
def wilson_upper_bound(wins, games, z=1.96):
    n = games
    p_hat = wins / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / denominator
    return center + margin

counter_stats["wilson_upper"] = wilson_upper_bound(counter_stats["wins"], counter_stats["games"])
counter_stats["risk_score"] = counter_stats["wilson_upper"] - counter_stats["confidence_score"]

model1_table = counter_stats.merge(
    champion_style[[
        "TeamPosition", "ChampionName", "scaling_score", "split_push_score",
        "team_fight_score", "snowball_score", "difficulty_tier", "difficulty_label",
    ]],
    on=["TeamPosition", "ChampionName"],
    how="left",
)

print()
print("Model 1 final tablosu - kolonlar:")
print(model1_table.columns.tolist())
print("Toplam satir:", len(model1_table))

# --- Adim 10: Model 1'in final feature store'unu diske kaydet ---

output_path = "data/processed/model1_counter_features.parquet"
model1_table.to_parquet(output_path, index=False)
print()
print(f"Kaydedildi: {output_path}  ({len(model1_table)} satir)")
