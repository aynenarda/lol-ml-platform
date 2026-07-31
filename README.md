# LoL ML Decision Support Platform

Kendi veri toplama, feature engineering, model eğitimi ve tahmin altyapısına sahip,
League of Legends için Machine Learning tabanlı karar destek sistemi.

## Durum: Model 1 (Counter Pick Recommendation Engine) tamamlandı

## Veri Kaynağı

Kaggle: `californianbill/patch-25-14-lol-league-of-legends-ranked-games`
100.843 NA ranked maç (Platinum+), gerçek Riot API (Match-V5) ile toplanmış, CC0 lisans.

- `data/raw/matchData.csv` — ham kaynak (dokunulmaz, referans)
- `data/processed/matches_long.parquet` — wide (1770 kolon, maç bazlı) formattan
  long/tidy (oyuncu-maç bazlı, 1.018.430 satır) formata çevrilmiş ara veri (temizlenmemiş)
- `data/processed/matches_clean.parquet` — temizlenmiş veri (1.001.350 satır, 100.135 maç).
  Elenenler: <5dk maçlar (1648), erken teslim (1476 maç), eksik TeamPosition (508 maç),
  kazanan bilgisi tutarsız (5 maç) — birleşimde 1.708 maç (%1.7) elendi.
- `data/processed/counter_stats.parquet` — Model 1'in temel feature store'u:
  (lane, ChampionName, ChampionName_opp) bazında games/wins/win_rate/confidence_score
  (Wilson score alt sınırı) ve has_enough_data (MIN_GAMES=20) — 42.354 satır.

## Model 1: Counter Pick Recommendation Engine — Tamamlandı

**Dosyalar:**
- `feature_engineering.py` — tüm feature üretim pipeline'ı (self-join,
  matchup istatistikleri, champion-level oyun tarzı özellikleri, Riot'un
  statik champion verisiyle birleştirme)
- `model1_counter_pick.py` — asıl öneri motoru: `recommend_counters(enemy_champion, lane)`
- `data/processed/model1_counter_features.parquet` — final feature store (42.354 satır:
  lane × champion × opponent)
- `data/external/champions_meraki.json` — Riot'un statik champion verisi
  (Meraki Analytics CDN üzerinden, difficulty/roles için) — büyük statik
  dosya olduğu için git'e girmiyor, `feature_engineering.py` çalıştırılırken
  otomatik indirilmesi gerekiyor (henüz otomatik indirme kodu yok, elle
  indirilen dosya kullanıldı — ileride otomatikleştirilebilir)

**Ürettiğimiz alanlar ve kaynakları:**
- `win_rate`, `games` (Sample Size) — self-join + groupby
- `confidence_score` — Wilson Score Interval alt sınırı (küçük örneklemi cezalandırır)
- `risk_score` — Wilson aralığının genişliği (üst sınır - alt sınır), belirsizlik göstergesi
- `first_blood_rate` / `first_tower_rate` — Lane Pressure göstergesi
- `scaling_score` — win rate'in maç süresiyle korelasyonu (**bilinen sınırlama:**
  bu korelasyonel bir proxy, confounding riski taşır — kısa maçlar genelde
  "ezici galibiyet" anlamına gelir, champion'ın kendi gücünden bağımsız olabilir)
- `split_push_score` — bina hasarının lane-içi ortalamaya oranı
- `team_fight_score` — assist oranının lane-içi ortalamaya oranı
- `snowball_score` — en uzun öldürme serisinin lane-içi ortalamaya oranı
- `power_spike` — 3 süre dilimi (kısa/orta/uzun) arasında win rate'in ZİRVE
  yaptığı dilim (scaling_score'dan farkı: doğrusal olmayan, tam ortada
  zirve yapan şampiyonları da yakalar)
- `difficulty_tier` / `difficulty_label` — Riot'un resmi statik zorluk derecesi
  (Meraki Analytics üzerinden, **1-3 ölçek**: Kolay/Orta/Zor — 0-10 değil)

**Bilinen sınırlamalar (dürüstçe not düşülmüştür):**
- Rank segmentasyonu yok (veri setinde maç bazlı rank bilgisi yok, sadece genel
  "Platinum+" alt sınırı var) — counter-matchup bilgisinin büyük ölçüde
  rank-bağımsız olduğu kabul edildi.
- `scaling_score` nedensellik değil korelasyon — yukarıda açıklandı.
- Sadece NA sunucusu, tek zaman dilimi (patch 25.14+) — bölgesel/patch farkları yok.

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
- [x] Milestone 6: Model 1 - Counter Pick Recommendation Engine
- [ ] Milestone 7+: Diğer modeller, Decision Engine, Backend, Frontend
- [ ] (ileride) Kendi Riot API crawler'ımızı kurup gerçek veri toplama pipeline'ına geçiş

Detaylar her milestone'a gelindiğinde bu dosyaya eklenecek.
