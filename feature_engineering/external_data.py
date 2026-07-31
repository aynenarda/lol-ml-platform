"""Riot'un statik champion verisini (mac istatistiklerinden turemeyen,
Riot'un tasarim ekibinin belirledigi degerler) yukler.

Kaynak: Meraki Analytics CDN - Riot'un LCU/game-data API'sinin
community-maintained bir aynasi (Data Dragon'a alternatif, bu ortamda
ddragon.leagueoflegends.com dogrudan erisilemedigi icin bu kaynak
kullanildi). Tek seferlik indirilip data/external/ icinde saklanir
(gitignore'da, ~13MB oldugu icin repo'ya girmiyor).
"""

import json

# Bizim veride "FiddleSticks" yazarken meraki'de "Fiddlesticks" - tek istisna.
NAME_FIXES = {"FiddleSticks": "Fiddlesticks"}

DIFFICULTY_LABELS = {1: "Kolay", 2: "Orta", 3: "Zor"}


def load_difficulty_ratings(path="data/external/champions_meraki.json"):
    """champion adindan difficulty_tier'a (1-3) bir sozluk dondurur."""
    with open(path, encoding="utf-8") as f:
        meraki_data = json.load(f)

    ratings = {}
    for name, champ in meraki_data.items():
        ratings[name] = champ["attributeRatings"]["difficulty"]

    # Bizim veride kullanilan isimlerle de erisilebilsin diye eklenen alias'lar.
    for our_name, meraki_name in NAME_FIXES.items():
        if meraki_name in ratings:
            ratings[our_name] = ratings[meraki_name]

    return ratings
