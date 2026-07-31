"""Meta kaymasini (concept drift) takip etmek icin patch bazli, ustel
azalan agirliklandirma.

Neden gerekli: tum gecmis veriyi esit agirlikla ortalarsak, yeni bir
trend (bir sampiyonun yeni bir lane'de guclenmesi gibi) binlerce eski
mac altinda kaybolur. Sabit bir "sadece son patch" kesimi de riskli -
veri miktarini asiri azaltip nadir eslesmeleri tamamen verisiz birakir.
Üstel azalan agirik, ikisinin ortasi: guncel veriye agirlik veriyor
ama eski veriyi tamamen atmiyor (smoothing).

DECAY=0.5 -> her bir onceki patch, bir sonrakinin YARISI kadar agirlikli
sayilir. Bu, "yarilanma omru" (half-life) kavramiyla ayni mantik.
"""

DECAY = 0.5


def _patch_sort_key(patch_str):
    """'15.14' -> (15, 14) - dogru sayisal siralama icin (string
    siralamasi '15.9' > '15.10' gibi hatalara yol acabilirdi)."""
    return tuple(int(part) for part in patch_str.split("."))


def assign_patch_weights(df, decay=DECAY):
    """df'e, hangi patch'in en guncel oldugunu OTOMATIK tespit edip her
    satira bir 'weight' kolonu ekler. En guncel patch'e 1.0, bir onceki
    patch'e 'decay', ondan onceki 'decay^2' ... agirligi verilir."""
    patches_sorted = sorted(df["patch"].dropna().unique(), key=_patch_sort_key)
    latest_patch = patches_sorted[-1]
    patch_to_age = {p: len(patches_sorted) - 1 - i for i, p in enumerate(patches_sorted)}

    df = df.copy()
    df["weight"] = df["patch"].map(patch_to_age).map(lambda age: decay ** age)

    return df, latest_patch
