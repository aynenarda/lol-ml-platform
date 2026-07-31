"""Blade & Chest modeli: sampiyonlar arasi asimetrik, gecisken-olmayan
(intransitive - tas-kagit-makas tarzi) iliskileri ogrenen bir embedding
modeli.

Referans fikir: oyun dengesi arastirmalarinda kullanilan "Blade & Chest"
yaklasimi - her varlik (burada: sampiyon) icin iki ayri vektor ogrenilir:
- blade: saldiri/baski profili
- chest: savunma/dayaniklilik profili

Skor(A, B) = strength[A] - strength[B]
           + blade[A] . chest[B]
           - blade[B] . chest[A]

Bu formul kasitli olarak ANTISIMETRIK: Skor(A,B) = -Skor(B,A). Bu, temel
bir olasilik tutarliligini saglar: P(A, B'yi yener) + P(B, A'yi yener) = 1
(sigmoid fonksiyonunun ozelligi sayesinde). Onceki (sadece strength farki
kullanan) modelde bu otomatikti ama etkilesim terimi yoktu; burada hem
tutarliligi koruyoruz hem etkilesimi ekliyoruz.
"""

import torch
import torch.nn as nn


class BladeChestModel(nn.Module):
    def __init__(self, num_champions, embed_dim=8):
        super().__init__()
        self.strength = nn.Embedding(num_champions, 1)
        self.blade = nn.Embedding(num_champions, embed_dim)
        self.chest = nn.Embedding(num_champions, embed_dim)

        # Kucuk rastgele baslangic degerleri - buyuk baslangic degerleri
        # egitimin basinda sigmoid'i doyurup (saturate) gradyanlarin
        # cok kucuk cikmasina (vanishing gradient) yol acabilir.
        nn.init.normal_(self.strength.weight, std=0.01)
        nn.init.normal_(self.blade.weight, std=0.01)
        nn.init.normal_(self.chest.weight, std=0.01)

    def forward(self, my_champ_idx, opp_champ_idx):
        strength_diff = self.strength(my_champ_idx) - self.strength(opp_champ_idx)
        strength_diff = strength_diff.squeeze(-1)

        my_blade = self.blade(my_champ_idx)
        my_chest = self.chest(my_champ_idx)
        opp_blade = self.blade(opp_champ_idx)
        opp_chest = self.chest(opp_champ_idx)

        interaction = (my_blade * opp_chest).sum(dim=-1) - (opp_blade * my_chest).sum(dim=-1)

        logit = strength_diff + interaction
        return logit
