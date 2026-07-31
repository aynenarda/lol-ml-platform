# LoL ML Decision Support Platform

Kendi veri toplama, feature engineering, model eğitimi ve tahmin altyapısına sahip,
League of Legends için Machine Learning tabanlı karar destek sistemi.

## Durum: Model 1 (Counter Pick Recommendation Engine) tamamlandı

## Veri Kaynağı

Kaggle: `californianbill/patch-25-14-lol-league-of-legends-ranked-games`
100.843 NA ranked maç (Platinum+), gerçek Riot API (Match-V5) ile toplanmış, CC0 lisans.

- `data/raw/matchData.csv` — ham kaynak (dokunulmaz, referans)
- `data/processed/matches_clean.parquet` — temizlenmiş, long/tidy formatta veri
  (1.001.350 satır, 100.135 maç). Elenenler: <5dk maçlar (1648), erken teslim
  (1476 maç), eksik TeamPosition (508 maç), kazanan bilgisi tutarsız (5 maç) —
  birleşimde 1.708 maç (%1.7) elendi.
- `data/processed/model1_counter_features.parquet` — Model 1'in final feature
  store'u (aşağıda detaylı) — `run_pipeline.py` çalıştırıldığında ikisi de
  otomatik üretilir.

## Model 1: Counter Pick Recommendation Engine — Tamamlandı

**Modül yapısı** (mimarideki isimlendirmeyle birebir):
```
data_collector/raw_loader.py          -> ham CSV'yi okur (ileride Riot API crawler'ı da buraya girecek)
data_cleaner/reshape.py               -> wide -> long dönüşümü
data_cleaner/clean.py                 -> geçersiz maç filtreleme kuralları
feature_engineering/matchups.py       -> self-join + win rate + Wilson score + lane pressure
feature_engineering/champion_style.py -> scaling/power spike/split push/team fight/snowball
feature_engineering/external_data.py  -> Riot'un statik champion verisini yükler (difficulty)
feature_engineering/build_model1_features.py -> yukarıdakileri birleştirip final tabloyu üretir
prediction_engine/model1_counter_pick.py     -> recommend_counters(model1_table, enemy, lane)
run_pipeline.py                       -> hepsini uçtan uca çalıştıran orkestratör (raporlama burada)
```
**Tasarım prensibi:** modül fonksiyonları saf (girdi→çıktı), print/log basmaz —
raporlama sadece `run_pipeline.py`'de. Böylece her fonksiyon başka bir yerden
(örn. ileride Backend API'den) sessizce çağrılabilir ve test edilebilir.

- `data/processed/model1_counter_features.parquet` — final feature store (42.354 satır:
  lane × champion × opponent)
- `data/external/champions_meraki.json` — Riot'un statik champion verisi
  (Meraki Analytics CDN üzerinden, difficulty/roles için) — büyük statik
  dosya olduğu için git'e girmiyor, elle indirildi (ileride otomatikleştirilebilir)

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

## Kullanıcı Girdisinden Öğrenen Alias Sistemi (prediction_engine/champion_alias.py)

**Bu, Model 1'in oyun bilgisinden (yukarıdaki eğitilen modellerden) TAMAMEN
bağımsız, ayrı bir öğrenme döngüsü.** Oyun sonucu değil, "kullanıcı ne demek
istedi" sorusunu öğreniyor.

**Akış:**
1. Tam eşleşme var mı (büyük/küçük harf duyarsız)? → direkt dön.
2. Daha önce öğrenilmiş bir alias var mı (`data/learned/champion_aliases.json`)? → direkt dön, soru sorulmaz.
3. Hiçbiri yoksa: önce **prefix eşleşmesi** dene ("heim" → "heim" ile başlayan şampiyonlar), yoksa genel fuzzy string matching'e düş.
4. Kullanıcıya sor: *"'heim' ile 'Heimerdinger' mi demek istedin?"* — onaylanırsa **kalıcı olarak kaydedilir**, bir daha sorulmaz.

**Önemli düzeltme:** İlk denemede `difflib`'in genel benzerlik oranı "heim"i
yanlışlıkla "Hwei"ye eşleştirdi (genel string benzerliği kısaltmalar için
güvenilir değil). Prefix eşleşmesini öncelik olarak eklemek bunu düzeltti.

**Bu sistemin `data/learned/` klasörü diğerlerinden farklı:** `.gitignore`'da
DEĞİL — çünkü bu, tekrar üretilebilir bir önbellek değil, gerçek kalıcı
öğrenilmiş bilgi (kullanıcı etkileşiminden birikiyor).

## Model 1'in Öğrenen Versiyonu (training_pipeline/ + model_registry/)

Sayma yöntemine ek olarak, aynı problemi **gerçek ML modelleriyle** de çözdük —
amaç gradient descent'in gerçekte nasıl çalıştığını, embedding'lerin ne
öğrendiğini elle görmek.

**1) Bradley-Terry (düz lojistik regresyon, scikit-learn `SGDClassifier`):**
- Her şampiyona TEK bir "genel güç" katsayısı öğreniyor (`training_pipeline/features.py`
  + `training_pipeline/train_win_probability.py`).
- L2 regularization (`alpha=0.01`) olmadan küçük örneklemli (off-role) şampiyonlar
  katsayıları bozuyordu — regularization ekleyince gerçek TOP meta'sıyla örtüşen
  sonuçlar çıktı (Fiora, Singed, Wukong, Urgot, Riven, Jax en güçlü).
- **Sınırlama:** toplamsal (additive) model, taşlı-kağıt-makas tarzı özel
  matchup etkileşimlerini yapısal olarak yakalayamıyor. Jax vs Renekton için
  %51.4 tahmin etti (sayma yöntemi: %57.6).

**2) Blade & Chest modeli (PyTorch, embedding tabanlı, `training_pipeline/blade_chest_model.py`):**
- Oyun dengesi literatüründen bir teknik — her şampiyona genel güç dışında
  iki embedding vektörü de öğretiliyor: `blade` (saldırı profili) ve `chest`
  (savunma profili). Skor = güç farkı + (benim blade'im · rakip chest'i) -
  (rakip blade'i · benim chest'im) — bu formül kasıtlı olarak antisimetrik,
  yani P(A kazanır) + P(B kazanır) = 1 garantisi var.
- Adam optimizer + BCEWithLogitsLoss ile eğitildi (`training_pipeline/train_blade_chest.py`).
- **Early stopping eklendi** (patience=5): test loss 5 epoch boyunca iyileşmezse
  eğitim durur, en iyi test-loss'lu ağırlıklar geri yüklenir.

**Sonuç karşılaştırması (Jax vs Renekton, TOP):**

| Yöntem | Tahmin | Not |
|---|---|---|
| Sayma (276 maç) | %57.6 | Doğrudan gözlem |
| Bradley-Terry (düz) | %51.4 | Özel etkileşimi kaçırıyor |
| Blade & Chest, 30 epoch | %57.3 | Sayma yöntemine çok yakın |
| Blade & Chest, early stopping (epoch 8) | %54.0 | Genel accuracy daha iyi (0.5256 vs 0.5192) ama bu spesifik çiftte sayma yönteminden daha uzak |

**Dürüst gözlem — bias-variance trade-off:** Early stopping genel test
accuracy'yi artırdı (tüm matchup'lar ortalamasında daha iyi genelleme) ama
bu SPESİFİK, sağlam örneklemli (276 maç) çiftte tahmini sayma yönteminden
uzaklaştırdı. 30 epoch'luk model bu çifte biraz "fazla uyum" sağlamıştı
ama o uyum bu örnekte aslında gerçeğe daha yakındı. Genelleme ortalamada
iyi olmak demek, her tekil örnekte en iyi olmak demek değildir.
- **5 lane'in tamamına genişletildi** (her lane'in kendi modeli var, çünkü
  şampiyon havuzu ve dinamikler lane'e göre çok farklı):

| Lane | Test Accuracy | Tutarlılık Kontrolü (P(A)+P(B)) |
|---|---|---|
| TOP | 0.5256 | 1.0000 |
| JUNGLE | 0.5218 | 1.0000 |
| MID | 0.5237 | 1.0000 |
| ADC | 0.5247 | 1.0000 |
| SUPPORT | 0.5205 | 1.0000 |

- Kayıt: `model_registry/blade_chest_{TOP,JUNGLE,MIDDLE,BOTTOM,UTILITY}.pt`,
  `model_registry/win_probability_TOP.joblib`

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
