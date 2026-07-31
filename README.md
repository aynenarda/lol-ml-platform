# LoL ML Decision Support Platform

Kendi veri toplama, feature engineering, model eğitimi ve tahmin altyapısına sahip,
League of Legends için Machine Learning tabanlı karar destek sistemi.

## Durum: Milestone 4 - Feature Engineering (sırada)

## Veri Kaynağı

Kaggle: `californianbill/patch-25-14-lol-league-of-legends-ranked-games`
100.843 NA ranked maç (Platinum+), gerçek Riot API (Match-V5) ile toplanmış, CC0 lisans.

- `data/raw/matchData.csv` — ham kaynak (dokunulmaz, referans)
- `data/processed/matches_long.parquet` — wide (1770 kolon, maç bazlı) formattan
  long/tidy (oyuncu-maç bazlı, 1.018.430 satır) formata çevrilmiş ara veri (temizlenmemiş)
- `data/processed/matches_clean.parquet` — temizlenmiş veri (1.001.400 satır, 100.140 maç).
  Elenenler: <5dk maçlar (1648), erken teslim (1476 maç), eksik TeamPosition (508 maç) —
  birleşimde 1.703 maç (%1.7) elendi.

**Keşfedilen yapı:** `participantIndex` 0-4 = Team1, 5-9 = Team2. Lane'ler
hizalı (0↔5 TOP, 1↔6 JUNGLE, 2↔7 MIDDLE, 3↔8 BOTTOM, 4↔9 UTILITY) — aynı
maçtaki karşılıklı rakipleri bulmak için bu hizalama kullanılacak (Model 2).

## Yol Haritası (yüksek seviye)

- [x] Milestone 0: Proje iskeleti, venv, git
- [x] Milestone 1: Veri kaynağı kararı (hibrit: Kaggle → sonra Riot API crawler)
- [x] Milestone 2: İlk EDA + wide→long reshape + parquet'e kaydetme
- [x] Milestone 3: Data Cleaning (remake/erken teslim/eksik pozisyon filtrelendi)
- [ ] Milestone 4: Feature Engineering (temel kavramlar + ilk feature seti)
- [ ] Milestone 5: Problem framing (classification/regression/ranking ayrımı)
- [ ] Milestone 6: Model 1 - Counter Pick Recommendation Engine
- [ ] Milestone 7+: Diğer modeller, Decision Engine, Backend, Frontend
- [ ] (ileride) Kendi Riot API crawler'ımızı kurup gerçek veri toplama pipeline'ına geçiş

Detaylar her milestone'a gelindiğinde bu dosyaya eklenecek.
