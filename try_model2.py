"""Model 2+3+4'u denemek icin interaktif script: benim sampiyonum +
rakip sampiyon + lane girip matchup raporu al."""

import pandas as pd

from prediction_engine.model2_matchup_intelligence import load_model2_data, get_matchup_report
from prediction_engine.champion_alias import load_aliases, resolve_champion

pd.set_option("display.max_columns", None)

VALID_LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

RUNE_SLOT_LABELS = {
    "PrimaryStylePerk1": "Anahtar Rün (Keystone)",
    "PrimaryStylePerk2": "Ana Yol - 1. Rün",
    "PrimaryStylePerk3": "Ana Yol - 2. Rün",
    "PrimaryStylePerk4": "Ana Yol - 3. Rün",
    "SubStylePerk1": "Yardımcı Yol - 1. Rün",
    "SubStylePerk2": "Yardımcı Yol - 2. Rün",
}
LANE_ALIASES = {
    "top": "TOP",
    "jungle": "JUNGLE", "jg": "JUNGLE", "jng": "JUNGLE", "jgl": "JUNGLE", "jung": "JUNGLE",
    "middle": "MIDDLE", "mid": "MIDDLE",
    "bottom": "BOTTOM", "bot": "BOTTOM", "adc": "BOTTOM", "ad carry": "BOTTOM", "carry": "BOTTOM",
    "utility": "UTILITY", "support": "UTILITY", "supp": "UTILITY", "sup": "UTILITY", "util": "UTILITY",
}

print("Veri yukleniyor...")
model1_table, gold_cs_table, item_rune_lookup, item_names, perk_names = load_model2_data()

all_champions = pd.concat([model1_table["ChampionName"], model1_table["ChampionName_opp"]]).unique().tolist()
learned_aliases = load_aliases()


def normalize_lane(raw_lane):
    return LANE_ALIASES.get(raw_lane.strip().lower())


def normalize_champion(raw_name):
    def ask_confirmation(candidate_name):
        answer = input(f"'{raw_name}' ile '{candidate_name}' mi demek istedin? (e/h): ").strip().lower()
        return answer in ("e", "evet", "y", "yes")

    champion, newly_learned = resolve_champion(raw_name, all_champions, learned_aliases, ask_confirmation)
    if newly_learned:
        learned_aliases[raw_name.strip().lower()] = champion
        print(f"Not edildi: bundan sonra '{raw_name}' yazinca dogrudan '{champion}' anlasilacak.\n")
    return champion


print("Model 2: Matchup Intelligence + Item + Rune Onerisi")
print("Lane: top, jungle/jg, mid, adc, support")
print("Cikmak icin bos birak ve Enter'a bas.")
print()

while True:
    my_raw = input("Senin sampiyonun: ").strip()
    if not my_raw:
        break
    my_champion = normalize_champion(my_raw)
    if my_champion is None:
        print(f"'{my_raw}' adinda bir sampiyon bulunamadi.\n")
        continue

    enemy_raw = input("Rakip sampiyon: ").strip()
    enemy_champion = normalize_champion(enemy_raw)
    if enemy_champion is None:
        print(f"'{enemy_raw}' adinda bir sampiyon bulunamadi.\n")
        continue

    lane_raw = input("Lane: ").strip()
    lane = normalize_lane(lane_raw)
    if lane is None:
        print(f"Gecersiz lane: '{lane_raw}'\n")
        continue

    report = get_matchup_report(
        my_champion, enemy_champion, lane,
        model1_table, gold_cs_table, item_rune_lookup, item_names, perk_names,
    )

    if report is None:
        print(f"'{my_champion}' vs '{enemy_champion}' ({lane}) icin veri bulunamadi.\n")
        continue

    print()
    print(f"=== {report['my_champion']} vs {report['enemy_champion']} ({lane}) ===")
    print(f"Win rate: %{report['win_rate']*100:.1f}  "
          f"(guven-ayarli: %{report['confidence_score']*100:.1f}, {report['games']} mac)")

    if "expected_gold_diff" in report:
        print(f"Beklenen gold farki (10 dk'da degil, mac genelinde): {report['expected_gold_diff']:+.0f}")
        print(f"Beklenen CS farki: {report['expected_cs_diff']:+.1f}")

    if report["top_items"]:
        print(f"\nEn cok kazandiran item'lar ({report['build_sample_size']} kazanilan mactan):")
        for item in report["top_items"]:
            print(f"   {item['name']}  (%{item['pick_rate']*100:.0f} maçta var)")

        combo = report["rune_combo"]
        print(f"\nEn cok kazandiran rune kombinasyonu (%{combo['pick_rate']*100:.0f} maçta):")
        for slot, label in RUNE_SLOT_LABELS.items():
            print(f"   {label}: {combo['perks'][slot]}")
    else:
        print("\nBu matchup icin yeterli item/rune verisi yok (az oynanmis).")

    print()
