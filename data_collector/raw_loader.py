"""Ham veriyi okur. Su an Kaggle CSV'sinden okuyor; ileride Riot API
crawler'i da bu modulun icine, ayni arayuzu (ayni sekilde kullanilan bir
fonksiyon) saglayarak eklenecek."""

import pandas as pd

# Wide formatta tasimak istedigimiz participant-bazli alanlar.
PARTICIPANT_FIELDS = [
    "ChampionName", "TeamPosition", "Win", "Kills", "Deaths", "Assists", "GoldEarned",
    "GameEndedInEarlySurrender", "FirstBloodKill", "FirstTowerKill",
    "DamageDealtToBuildings", "LargestKillingSpree", "TotalDamageDealtToChampions",
]

# Mac bazli (participant'tan bagimsiz) alanlar. gameVersion + gameCreation,
# meta kaymasini (concept drift) takip edebilmek icin gerekli - hangi
# patch'te, ne zaman oynandigi.
MATCH_LEVEL_FIELDS = ["gameMode", "mapId", "gameDuration", "gameVersion", "gameCreation"]


def load_raw_wide(path="data/raw/matchData.csv"):
    """matchData.csv'yi, sadece ihtiyacimiz olan kolonlari okuyarak
    wide formatta dondurur (her satir = 1 mac, participant0..9 kolonlari)."""
    needed_cols = ["matchId"] + MATCH_LEVEL_FIELDS + [
        f"participant{i}{field}" for i in range(10) for field in PARTICIPANT_FIELDS
    ]
    return pd.read_csv(path, usecols=needed_cols)
