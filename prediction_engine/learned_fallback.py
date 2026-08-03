"""Sayma yontemi yeterli veri bulamadiginda (dusuk sample size) devreye
giren fallback: egitilmis Blade & Chest modelinden tahmin uretir.

Onemli fark: bu tahminler DOGRUDAN GOZLEM degil, modelin diger tum
maclardan ogrendigi genel guc + etkilesim bilgisinden turetilmis bir
TAHMIN'dir - kullaniciya bu ayrimi acikca belirtmek gerekir.
"""

import torch

from training_pipeline.blade_chest_model import BladeChestModel

_model_cache = {}


def _load_model(lane):
    if lane not in _model_cache:
        data = torch.load(f"model_registry/blade_chest_{lane}.pt", weights_only=False)
        model = BladeChestModel(num_champions=len(data["champ_to_idx"]), embed_dim=data["embed_dim"])
        model.load_state_dict(data["model_state"])
        model.eval()
        _model_cache[lane] = (model, data["champ_to_idx"])
    return _model_cache[lane]


def predict_counters(enemy_champion, lane, top_n=5):
    """Modelin, verilen rakibe karsi en iyi buldugu top_n sampiyonu ve
    tahmini win probability'lerini dondurur. Rakip modelin hic gormedigi
    bir sampiyonsa bos liste doner."""
    model, champ_to_idx = _load_model(lane)
    if enemy_champion not in champ_to_idx:
        return []

    opp_idx_val = champ_to_idx[enemy_champion]
    candidates = [c for c in champ_to_idx if c != enemy_champion]
    my_idx = torch.tensor([champ_to_idx[c] for c in candidates])
    opp_idx = torch.tensor([opp_idx_val] * len(candidates))

    with torch.no_grad():
        probs = torch.sigmoid(model(my_idx, opp_idx))

    results = sorted(zip(candidates, probs.tolist()), key=lambda pair: -pair[1])
    return results[:top_n]


def predict_single_matchup(my_champion, enemy_champion, lane):
    """Belirli bir (benim sampiyonum, rakip) cifti icin ogrenilen modelden
    tek bir win probability tahmini. Model 1'de "en iyi counter'lari
    listele" icin kullanilan predict_counters'in aksine, burada Model 2
    spesifik bir cift sordugu icin dogrudan o ciftin tahminini donduruyoruz.
    Ikisi de champ_to_idx'te yoksa None doner."""
    model, champ_to_idx = _load_model(lane)
    if my_champion not in champ_to_idx or enemy_champion not in champ_to_idx:
        return None

    my_idx = torch.tensor([champ_to_idx[my_champion]])
    opp_idx = torch.tensor([champ_to_idx[enemy_champion]])
    with torch.no_grad():
        return torch.sigmoid(model(my_idx, opp_idx)).item()
