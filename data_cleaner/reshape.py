"""Wide (mac bazli, participant0..9 kolonlu) formati long (oyuncu-mac
bazli) formata cevirir."""

import pandas as pd

from data_collector.raw_loader import PARTICIPANT_FIELDS, MATCH_LEVEL_FIELDS


def _extract_participant(wide_df, i):
    """Wide formattaki participant{i} kolonlarini long formata cevirir."""
    source_cols = [f"participant{i}{field}" for field in PARTICIPANT_FIELDS]
    part = wide_df[["matchId"] + MATCH_LEVEL_FIELDS + source_cols].copy()

    rename_map = {f"participant{i}{field}": field for field in PARTICIPANT_FIELDS}
    part = part.rename(columns=rename_map)

    part["participantIndex"] = i
    return part


def wide_to_long(wide_df):
    """10 participant'i alt alta birlestirip tek bir long DataFrame dondurur."""
    pieces = [_extract_participant(wide_df, i) for i in range(10)]
    return pd.concat(pieces, ignore_index=True)
