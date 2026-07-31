"""Model 1: Counter Pick Recommendation Engine.

Girdi: rakip sampiyon + lane. Cikti: en iyi karsi sampiyonlar, her biri
icin istatistiksel gerekce (win probability, confidence, sample size,
lane pressure, scaling, power spike, split push, team fight, snowball,
difficulty, risk) ve dogal dilde aciklama.
"""

import pandas as pd


def _explain(row):
    parts = [
        f"{row.games} maclik veride %{row.win_rate*100:.1f} kazanma orani "
        f"(guven-ayarli skor: %{row.confidence_score*100:.1f})."
    ]

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


def recommend_counters(model1_table, enemy_champion, lane, top_n=5, min_games=20):
    """Belirli bir rakip sampiyona ve lane'e karsi en iyi karsi
    sampiyonlari, aciklamalariyla birlikte dondurur."""
    candidates = model1_table[
        (model1_table["TeamPosition"] == lane)
        & (model1_table["ChampionName_opp"] == enemy_champion)
        & (model1_table["games"] >= min_games)
    ].copy()

    candidates = candidates.sort_values("confidence_score", ascending=False).head(top_n)
    candidates["explanation"] = candidates.apply(_explain, axis=1)

    return candidates
