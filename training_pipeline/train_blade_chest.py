"""Blade & Chest modelini egitir ve sayma yontemi + duz Bradley-Terry
modeliyle karsilastirir."""

import copy

import numpy as np
import torch
import torch.nn as nn
import joblib

from data_collector.raw_loader import load_raw_wide
from data_cleaner.reshape import wide_to_long
from data_cleaner.clean import find_invalid_match_ids, clean_long_data
from feature_engineering.matchups import build_matchups
from training_pipeline.blade_chest_model import BladeChestModel

EMBED_DIM = 8
EPOCHS = 30
BATCH_SIZE = 4096
LEARNING_RATE = 0.05
WEIGHT_DECAY = 1e-4  # L2 regularization - kucuk orneklemli sampiyonlari cezalandirir
PATIENCE = 5  # test loss bu kadar epoch boyunca iyilesmezse egitimi durdur


def prepare_tensors(matchups_df, lane):
    lane_df = matchups_df[matchups_df["TeamPosition"] == lane]

    champions = sorted(set(lane_df["ChampionName"]) | set(lane_df["ChampionName_opp"]))
    champ_to_idx = {c: i for i, c in enumerate(champions)}

    my_idx = torch.tensor(lane_df["ChampionName"].map(champ_to_idx).to_numpy(), dtype=torch.long)
    opp_idx = torch.tensor(lane_df["ChampionName_opp"].map(champ_to_idx).to_numpy(), dtype=torch.long)
    y = torch.tensor(lane_df["Win"].astype(float).to_numpy(), dtype=torch.float32)

    return my_idx, opp_idx, y, champ_to_idx


def train_test_split_tensors(my_idx, opp_idx, y, test_frac=0.2, seed=42):
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(n * test_frac)
    test_ids, train_ids = perm[:n_test], perm[n_test:]
    return (
        (my_idx[train_ids], opp_idx[train_ids], y[train_ids]),
        (my_idx[test_ids], opp_idx[test_ids], y[test_ids]),
    )


def train_blade_chest(matchups_df, lane):
    my_idx, opp_idx, y, champ_to_idx = prepare_tensors(matchups_df, lane)
    (my_tr, opp_tr, y_tr), (my_te, opp_te, y_te) = train_test_split_tensors(my_idx, opp_idx, y)

    print(f"\n=== {lane} lane: {len(y)} matchup, {len(champ_to_idx)} sampiyon, embed_dim={EMBED_DIM} ===")
    print(f"Egitim: {len(y_tr)}  Test: {len(y_te)}")

    model = BladeChestModel(num_champions=len(champ_to_idx), embed_dim=EMBED_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()  # sigmoid + binary cross-entropy, birlikte (sayisal olarak daha kararli)

    n_train = len(y_tr)
    best_test_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print()
    print(f"--- Gradient Descent (Adam optimizer) ilerlemesi (early stopping, patience={PATIENCE}) ---")
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            batch_ids = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(my_tr[batch_ids], opp_tr[batch_ids])
            loss = loss_fn(logits, y_tr[batch_ids])
            loss.backward()       # geri yayilim: gradyanlari hesaplar
            optimizer.step()      # agirliklari gradyana gore gunceller
            epoch_loss += loss.item() * len(batch_ids)

        epoch_loss /= n_train

        # Test loss'u HER epoch'ta olcuyoruz (sadece raporlama icin degil,
        # early stopping karari bu olcume dayaniyor).
        with torch.no_grad():
            test_logits = model(my_te, opp_te)
            test_loss = loss_fn(test_logits, y_te).item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}  train_loss={epoch_loss:.4f}  test_loss={test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Epoch {epoch:2d}  test_loss={test_loss:.4f}  "
                      f"-> {PATIENCE} epoch boyunca iyilesme yok, durduruluyor.")
                break

    # En iyi (en dusuk test loss'lu) agirliklari geri yukluyoruz - son
    # epoch'un agirliklari degil, cunku son epoch overfitting baslamis
    # olabilir.
    model.load_state_dict(best_state)
    print(f"\nEn iyi test_loss: {best_test_loss:.4f} agirliklari geri yuklendi.")

    with torch.no_grad():
        test_logits = model(my_te, opp_te)
        test_pred = (torch.sigmoid(test_logits) > 0.5).float()
        test_accuracy = (test_pred == y_te).float().mean().item()

    print()
    print(f"Final test accuracy: {test_accuracy:.4f}")

    return model, champ_to_idx


def predict_matchup(model, champ_to_idx, my_champion, opp_champion):
    my_idx = torch.tensor([champ_to_idx[my_champion]])
    opp_idx = torch.tensor([champ_to_idx[opp_champion]])
    with torch.no_grad():
        logit = model(my_idx, opp_idx)
        return torch.sigmoid(logit).item()


def main():
    print("1) Veri hazirlaniyor...")
    wide_df = load_raw_wide()
    long_df = wide_to_long(wide_df)
    invalid_ids = find_invalid_match_ids(wide_df, long_df)
    clean_df = clean_long_data(long_df, invalid_ids)
    matchups_df = build_matchups(clean_df)

    model, champ_to_idx = train_blade_chest(matchups_df, "TOP")

    jax_vs_renekton = predict_matchup(model, champ_to_idx, "Jax", "Renekton")
    renekton_vs_jax = predict_matchup(model, champ_to_idx, "Renekton", "Jax")

    print()
    print("=== Karsilastirma: Jax vs Renekton (TOP) ===")
    print(f"Sayma yontemi:         %57.6  (276 dogrudan mac)")
    print(f"Bradley-Terry (duz):   %51.4  (sadece genel guc farki)")
    print(f"Blade & Chest:         %{jax_vs_renekton*100:.1f}  (genel guc + ozel etkilesim)")
    print()
    print(f"Tutarlilik kontrolu: P(Jax kazanir) + P(Renekton kazanir) = "
          f"{jax_vs_renekton + renekton_vs_jax:.4f}  (1.0'a yakin olmali)")

    torch.save(
        {"model_state": model.state_dict(), "champ_to_idx": champ_to_idx, "embed_dim": EMBED_DIM},
        "model_registry/blade_chest_TOP.pt",
    )
    print("\nKaydedildi: model_registry/blade_chest_TOP.pt")


if __name__ == "__main__":
    main()
