"""Bradley-Terry tarzi ogrenilebilir feature matrisi uretir.

Her satir bir matchup (benim sampiyonum vs rakip sampiyon, belirli bir
lane'de). Her sampiyon icin bir kolon var: benim sampiyonumun kolonunda
+1, rakibin kolonunda -1, digerlerinde 0. Bu "fark" temsili, modelin
her sampiyona TEK bir sayisal guc katsayisi ogrenmesini saglar - iki
katsayinin farki, o matchup'taki beklenen sonucu belirler.

Matris cok seyrek (her satirda sadece 2 dolu hucre / ~170 kolon), bu
yuzden yogun (dense) numpy array yerine scipy sparse matris kullaniyoruz
- hem bellek hem hiz acisindan onemli fark yaratir.
"""

import numpy as np
from scipy.sparse import csr_matrix


def build_champion_diff_matrix(matchups_df, lane):
    lane_df = matchups_df[matchups_df["TeamPosition"] == lane]

    champions = sorted(set(lane_df["ChampionName"]) | set(lane_df["ChampionName_opp"]))
    champ_to_idx = {champ: i for i, champ in enumerate(champions)}

    n = len(lane_df)
    my_idx = lane_df["ChampionName"].map(champ_to_idx).to_numpy()
    opp_idx = lane_df["ChampionName_opp"].map(champ_to_idx).to_numpy()

    row_idx = np.concatenate([np.arange(n), np.arange(n)])
    col_idx = np.concatenate([my_idx, opp_idx])
    data = np.concatenate([np.ones(n), -np.ones(n)])

    X = csr_matrix((data, (row_idx, col_idx)), shape=(n, len(champions)))
    y = lane_df["Win"].astype(int).to_numpy()

    return X, y, champ_to_idx
