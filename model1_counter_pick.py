import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

model1 = pd.read_parquet("data/processed/model1_counter_features.parquet")


def explain(row):
    """Bir oneri satiri icin dogal dilde aciklama uretir - sayilara
    degil, VERIYE dayali bir gerekce."""
    parts = []

    parts.append(
        f"{row.games} maclik veride %{row.win_rate*100:.1f} kazanma orani "
        f"(guven-ayarli skor: %{row.confidence_score*100:.1f})."
    )

    if row.first_blood_rate > 0.15:
        parts.append("Bu eslesmede ilk kani alma orani ortalamanin uzerinde - lane baskisi guclu.")

    if row.scaling_score > 0.03:
        parts.append("Uzun maclarda daha basarili - gec oyun (scaling) potansiyeli var.")
    elif row.scaling_score < -0.03:
        parts.append("Kisa maclarda daha basarili - erken oyunda guclu, hizli bitirmeyi hedefler.")

    parts.append(f"En guclu oldugu donem: {row.power_spike}.")

    if row.split_push_score > 1.3:
        parts.append("Ortalamanin uzerinde bina hasari veriyor - split push potansiyeli yuksek.")

    if row.team_fight_score > 1.1:
        parts.append("Ortalamanin uzerinde assist orani - takim kavgasina katilimi yuksek.")

    if row.snowball_score > 1.15:
        parts.append("Uzun oldurme serileri ortalamanin uzerinde - erken avantaji buyutme potansiyeli var.")

    if row.risk_score > 0.25:
        parts.append("Dikkat: tahmindeki belirsizlik payi (risk_score) yuksek.")

    return " ".join(parts)


def recommend_counters(enemy_champion, lane, top_n=5, min_games=20):
    """Belirli bir rakip sampiyona ve lane'e karsi en iyi karsi
    sampiyonlari onerir."""
    candidates = model1[
        (model1["TeamPosition"] == lane)
        & (model1["ChampionName_opp"] == enemy_champion)
        & (model1["games"] >= min_games)
    ].copy()

    candidates = candidates.sort_values("confidence_score", ascending=False).head(top_n)

    print(f"\n=== {lane} lane'de '{enemy_champion}' rakibine karsi en iyi {top_n} secim ===\n")
    for _, row in candidates.iterrows():
        print(f"#{row.ChampionName}  |  win_rate=%{row.win_rate*100:.1f}  "
              f"confidence=%{row.confidence_score*100:.1f}  sample_size={row.games}  "
              f"difficulty={row.difficulty_label}  power_spike={row.power_spike}  "
              f"risk_score={row.risk_score:.3f}")
        print(f"   Neden: {explain(row)}")
        print()

    return candidates


if __name__ == "__main__":
    recommend_counters("Renekton", "TOP", top_n=5)
    recommend_counters("Zed", "MIDDLE", top_n=5)
