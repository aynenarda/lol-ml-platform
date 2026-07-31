"""Model 1'in ogrenilen versiyonu: Bradley-Terry tarzi lojistik regresyon.

SGDClassifier (Stochastic Gradient Descent Classifier) kullaniyoruz -
ismi bile gradient descent'i acikca belirtiyor. verbose=1 ile her
epoch'ta (verinin tamamindan bir kez gectiginde) modelin hatasinin
(loss) nasil azaldigini canli gorecegiz.
"""

import joblib
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

from data_cleaner.reshape import wide_to_long
from data_cleaner.clean import find_invalid_match_ids, clean_long_data
from data_collector.raw_loader import load_raw_wide
from feature_engineering.matchups import build_matchups
from training_pipeline.features import build_champion_diff_matrix


def train_for_lane(matchups_df, lane):
    X, y, champ_to_idx = build_champion_diff_matrix(matchups_df, lane)

    # Egitim/test ayrimi: modelin GORMEDIGI veride ne kadar iyi tahmin
    # ettigini olcmek icin. Egitimde kullanilan veride "iyi" gorunmek
    # kolay (ezberleme riski) - gercek test, hic gormedigi maclarda.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n=== {lane} lane: {X.shape[0]} matchup, {X.shape[1]} sampiyon ===")
    print(f"Egitim seti: {X_train.shape[0]} satir, Test seti: {X_test.shape[0]} satir")
    print()
    print("--- Gradient Descent ilerlemesi (her epoch'ta hata azalmali) ---")

    model = SGDClassifier(
        loss="log_loss",   # lojistik regresyon kaybi (binary cross-entropy)
        max_iter=50,       # veri uzerinden en fazla 50 epoch
        verbose=1,         # her epoch'un ilerlemesini yazdirir
        random_state=42,
        alpha=0.01,        # L2 regularization gucu - kucuk orneklemli
                           # sampiyonlarin katsayisini 0'a dogru ceker
    )
    model.fit(X_train, y_train)

    train_pred_proba = model.predict_proba(X_train)[:, 1]
    test_pred_proba = model.predict_proba(X_test)[:, 1]

    print()
    print("--- Sonuc ---")
    print(f"Egitim log-loss: {log_loss(y_train, train_pred_proba):.4f}")
    print(f"Test log-loss:   {log_loss(y_test, test_pred_proba):.4f}")
    print(f"Test accuracy:   {accuracy_score(y_test, model.predict(X_test)):.4f}")

    # Ogrenilen katsayilar: her sampiyona modelin atadigi tek bir "guc"
    # sayisi. Yuksek katsayi = model bu sampiyonu "genel olarak guclu"
    # olarak ogrenmis.
    coefficients = model.coef_[0]
    idx_to_champ = {i: c for c, i in champ_to_idx.items()}
    strength = pd.Series(coefficients, index=[idx_to_champ[i] for i in range(len(coefficients))])
    strength = strength.sort_values(ascending=False)

    print()
    print("--- Modelin ogrendigi EN GUCLU 10 TOP sampiyonu (katsayi) ---")
    print(strength.head(10).to_string())
    print()
    print("--- Modelin ogrendigi EN ZAYIF 10 TOP sampiyonu (katsayi) ---")
    print(strength.tail(10).to_string())

    return model, champ_to_idx, strength


def predict_matchup(model, champ_to_idx, my_champion, opp_champion):
    """Ogrenilen modelle belirli bir matchup icin win probability tahmini."""
    from scipy.sparse import csr_matrix

    n_champs = len(champ_to_idx)
    row = csr_matrix(
        ([1, -1], ([0, 0], [champ_to_idx[my_champion], champ_to_idx[opp_champion]])),
        shape=(1, n_champs),
    )
    return model.predict_proba(row)[0, 1]


def main():
    print("1) Veri hazirlaniyor (raw -> long -> clean -> matchups)...")
    wide_df = load_raw_wide()
    long_df = wide_to_long(wide_df)
    invalid_ids = find_invalid_match_ids(wide_df, long_df)
    clean_df = clean_long_data(long_df, invalid_ids)
    matchups_df = build_matchups(clean_df)

    model, champ_to_idx, strength = train_for_lane(matchups_df, "TOP")

    # Model Registry: egitilmis modeli + sampiyon-index eslemesini
    # diske kaydediyoruz (joblib, sklearn modelleri icin standart format).
    joblib.dump(
        {"model": model, "champ_to_idx": champ_to_idx},
        "model_registry/win_probability_TOP.joblib",
    )
    print("\nKaydedildi: model_registry/win_probability_TOP.joblib")

    # Karsilastirma: sayma yontemi Jax vs Renekton icin %57.6 win rate
    # bulmustu (276 mac). Ogrenilen model ne diyor?
    learned_prob = predict_matchup(model, champ_to_idx, "Jax", "Renekton")
    print()
    print(f"Ogrenilen model - Jax vs Renekton (TOP) win probability: %{learned_prob*100:.1f}")
    print("(Sayma yontemi bu matchup icin %57.6 bulmustu, 276 maclik dogrudan veriyle)")


if __name__ == "__main__":
    main()
