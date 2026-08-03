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


def load_item_names(path="data/external/items.json"):
    """Item ID (orn. 3020) -> item adi (orn. "Sorcerer's Shoes") sozlugu.
    Sadece ISIM cozumleme icin - hangi item'in ONERILECEGI bizim
    verimizden (win rate'e gore) cikacak, bu sadece etiketleme."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {item["id"]: item["name"] for item in items}


def load_item_metadata(path="data/external/items.json"):
    """Item ID -> {name, categories, price} sozlugu. Item'i "cekirdek
    build" listesine mi, "cizme" onerisine mi koyacagimizi, yoksa hic
    almayacagimizi (trinket/erken oyun item'i) belirlemek icin gerekli:
    - Trinket kategorisindekiler (fiyat=0) her zaman ayni, matchup'a
      ozgu degil -> atlaniyor.
    - Fiyat < 1000 olanlar (Doran'in Halkasi, Kara Muhur gibi) erken
      oyun/gecici item'lar -> atlaniyor.
    - "Boots" kategorisindekiler ayri bir "hangi cizme" onerisine gidiyor.
    - Kalanlar (fiyat >= 1000, cizme degil) gercek "cekirdek build"."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {
        item["id"]: {
            "name": item["name"],
            "categories": item.get("categories", []),
            "price": item.get("priceTotal", 0),
        }
        for item in items
    }


def load_perk_names(path="data/external/perks.json"):
    """Rune/perk ID -> isim (orn. 8005 -> "Press the Attack") sozlugu."""
    with open(path, encoding="utf-8") as f:
        perks = json.load(f)
    return {perk["id"]: perk["name"] for perk in perks}
