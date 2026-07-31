"""Kullanici girdisinden sampiyon ismi cozumleme + kullanici onayindan
ogrenen kalici alias sistemi.

Akis:
1. Tam eslesme var mi? (buyuk/kucuk harf duyarsiz) -> direkt don.
2. Daha once ogrenilmis bir alias var mi? -> direkt don (onay istemeden,
   cunku bir kere onaylanmisti).
3. Hicbiri yoksa: fuzzy string matching ile en yakin adaylari bul,
   kullaniciya sor. Onaylanirsa alias'i KALICI olarak kaydet - bir dahaki
   sefere ayni girdi icin artik soru sorulmaz.

Bu, oyun bilgisini (kim kimi yeniyor) DEGIL, sadece "kullanici ne demek
istedi" sorusunu ogreniyor - Model 1'in istatistiklerinden tamamen ayri,
bagimsiz bir ogrenme dongusu.
"""

import json
import difflib

ALIAS_FILE = "data/learned/champion_aliases.json"


def load_aliases(path=ALIAS_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_alias(alias_key, canonical_name, path=ALIAS_FILE):
    aliases = load_aliases(path)
    aliases[alias_key] = canonical_name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(aliases, f, ensure_ascii=False, indent=2, sort_keys=True)


def resolve_champion(raw_input, known_champions, aliases, confirm_fn):
    """raw_input'u bilinen bir sampiyon adina cozer.

    known_champions: gercek sampiyon adlarinin listesi (orn. model1_table'dan).
    aliases: load_aliases() ile yuklenmis sozluk (alias_key -> canonical).
    confirm_fn: (candidate_name) -> bool don duren bir fonksiyon; kullaniciya
        soru sorup evet/hayir almak icin disaridan enjekte ediliyor (boylece
        bu fonksiyon test edilebilir kaliyor, input() ile sikica baglanmiyor).

    Donen deger: (canonical_name_or_None, newly_learned: bool)
    """
    key = raw_input.strip().lower()

    exact_lookup = {c.lower(): c for c in known_champions}
    if key in exact_lookup:
        return exact_lookup[key], False

    if key in aliases:
        return aliases[key], False

    # Once PREFIX eslesmesine bak - kullanicilar genelde ismi kisaltir
    # (heim -> heimerdinger, voli -> volibear). difflib'in genel benzerlik
    # orani bunu her zaman dogru siralamiyor (orn. "heim" ile "Hwei" bile
    # "Heimerdinger"den daha "benzer" cikabiliyor) - prefix eslesmesi cok
    # daha guvenilir bir ilk kontrol.
    prefix_matches = sorted(
        (name for key_lower, name in exact_lookup.items() if key_lower.startswith(key)),
        key=len,  # en kisa (yani en yakin) isim once
    )
    for candidate_name in prefix_matches:
        if confirm_fn(candidate_name):
            save_alias(key, candidate_name)
            return candidate_name, True

    # Prefix eslesmesi yoksa genel fuzzy matching'e dus.
    candidates = difflib.get_close_matches(key, exact_lookup.keys(), n=3, cutoff=0.4)
    for candidate_key in candidates:
        candidate_name = exact_lookup[candidate_key]
        if confirm_fn(candidate_name):
            save_alias(key, candidate_name)
            return candidate_name, True

    return None, False
