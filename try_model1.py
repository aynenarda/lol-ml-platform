"""Model 1'i hizlica denemek icin interaktif script.

run_pipeline.py'yi tekrar calistirmaya gerek yok - zaten kaydedilmis
feature store'u (data/processed/model1_counter_features.parquet) okuyup
istedigin sampiyon/lane icin sorgu yapar.
"""

import pandas as pd

from prediction_engine.model1_counter_pick import recommend_counters
from prediction_engine.champion_alias import load_aliases, resolve_champion
from prediction_engine.learned_fallback import predict_counters

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

VALID_LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

# Kullanicilar lane'i genelde kisaltma/rol adiyla yazar (mid, jung, adc,
# support gibi), Riot'un ic ismini (MIDDLE, JUNGLE, BOTTOM, UTILITY)
# degil. Hepsini kabul edip dogru degere ceviriyoruz.
LANE_ALIASES = {
    "top": "TOP",
    "jungle": "JUNGLE", "jg": "JUNGLE", "jng": "JUNGLE", "jgl": "JUNGLE", "jung": "JUNGLE",
    "middle": "MIDDLE", "mid": "MIDDLE",
    "bottom": "BOTTOM", "bot": "BOTTOM", "adc": "BOTTOM", "ad carry": "BOTTOM", "carry": "BOTTOM",
    "utility": "UTILITY", "support": "UTILITY", "supp": "UTILITY", "sup": "UTILITY", "util": "UTILITY",
}

# Riot'un ic verisinde BOTTOM/UTILITY yazsa da, oyuncular bu rolleri
# ADC/SUPPORT olarak bilir - kullaniciya gosterirken bunu kullaniyoruz.
DISPLAY_LANE_LABELS = {
    "TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MID",
    "BOTTOM": "ADC", "UTILITY": "SUPPORT",
}

model1_table = pd.read_parquet("data/processed/model1_counter_features.parquet")

all_champions = pd.concat([model1_table["ChampionName"], model1_table["ChampionName_opp"]]).unique().tolist()
learned_aliases = load_aliases()


def normalize_lane(raw_lane):
    key = raw_lane.strip().lower()
    return LANE_ALIASES.get(key)


def normalize_champion(raw_name):
    def ask_confirmation(candidate_name):
        answer = input(f"'{raw_name}' ile '{candidate_name}' mi demek istedin? (e/h): ").strip().lower()
        return answer in ("e", "evet", "y", "yes")

    champion, newly_learned = resolve_champion(raw_name, all_champions, learned_aliases, ask_confirmation)
    if newly_learned:
        # resolve_champion diske kaydetti ama bu process'in bellekteki
        # kopyasini guncellemedi - bu satir olmadan AYNI oturumda ayni
        # sampiyon tekrar sorulunca hala eski (guncel olmayan) sozluge
        # bakilir ve soru tekrar sorulurdu.
        learned_aliases[raw_name.strip().lower()] = champion
        print(f"Not edildi: bundan sonra '{raw_name}' yazinca dogrudan '{champion}' anlasilacak.\n")
    return champion


print("Model 1: Counter Pick Recommendation Engine")
print("Lane: top, jungle/jg, mid, adc, support (kisaltmalari da kabul eder)")
print("Sampiyon adinda buyuk/kucuk harf onemli degil (orn. 'malzahar' calisir). "
      "Bosluksuz yazman lazim: KSante, DrMundo, AurelionSol, MonkeyKing (Wukong).")
print("Cikmak icin bos birak ve Enter'a bas.")
print()

while True:
    enemy_raw = input("Rakip sampiyon: ").strip()
    if not enemy_raw:
        break

    enemy = normalize_champion(enemy_raw)
    if enemy is None:
        print(f"'{enemy_raw}' adinda bir sampiyon bulunamadi. Yazimini kontrol et.\n")
        continue

    lane_raw = input("Lane: ").strip()
    lane = normalize_lane(lane_raw)
    if lane is None:
        print(f"Gecersiz lane: '{lane_raw}'. Ornek gecerli degerler: "
              "top, jungle/jg, mid, bot/adc, support/supp\n")
        continue

    result = recommend_counters(model1_table, enemy, lane, top_n=5)

    if result.empty:
        print(f"'{enemy}' ({DISPLAY_LANE_LABELS[lane]}) icin sayma yontemiyle "
              "yeterli veri bulunamadi (az oynanmis matchup).")

        fallback = predict_counters(enemy, lane, top_n=5)
        if not fallback:
            print("Ogrenilen model de bu sampiyonu bu lane'de hic gormemis, "
                  "hicbir tahmin uretilemiyor.\n")
            continue

        print("Ogrenilen model (Blade & Chest) uzerinden DUSUK GUVENILIRLIKLI "
              "bir tahmin - bu dogrudan gozlem DEGIL, modelin genellemesi:\n")
        for champion, prob in fallback:
            print(f"   #{champion}  tahmini_win_probability=%{prob*100:.1f}  (model tahmini, dogrudan veri yok)")
        print()
        continue

    print()
    for _, row in result.iterrows():
        print(f"#{row.ChampionName}  win_rate=%{row.win_rate*100:.1f}  "
              f"confidence=%{row.confidence_score*100:.1f}  "
              f"sample_size={row.games}  difficulty={row.difficulty_label}  "
              f"power_spike={row.power_spike}  risk_score={row.risk_score:.3f}")
        print(f"   {row.explanation}")
        print()
