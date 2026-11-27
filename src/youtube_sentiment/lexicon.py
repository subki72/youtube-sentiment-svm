"""Indonesian colloquial slang lexicon and safe dictionary loader."""

from __future__ import annotations

import io
import logging
from collections.abc import Mapping
from functools import lru_cache

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Curated offline Indonesian YouTube colloquial slang normalization map (>=50 entries).
CORE_SLANG_MAP: dict[str, str] = {
    "yg": "yang",
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "gk": "tidak",
    "tdk": "tidak",
    "bgt": "banget",
    "bgtt": "banget",
    "klo": "kalau",
    "kalo": "kalau",
    "bikin": "membuat",
    "mantap": "mantap",
    "mantul": "mantap betul",
    "parah": "parah",
    "parahh": "parah",
    "jelek": "jelek",
    "keren": "keren",
    "bagus": "bagus",
    "baguss": "bagus",
    "bohong": "bohong",
    "rusak": "rusak",
    "bener": "benar",
    "bnr": "benar",
    "emang": "memang",
    "emg": "memang",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "blm": "belum",
    "belom": "belum",
    "krn": "karena",
    "karna": "karena",
    "tp": "tapi",
    "tapi": "tapi",
    "dgn": "dengan",
    "sm": "sama",
    "sama": "sama",
    "aja": "saja",
    "aj": "saja",
    "kpd": "kepada",
    "utk": "untuk",
    "org": "orang",
    "orng": "orang",
    "jd": "jadi",
    "jdi": "jadi",
    "bkn": "bukan",
    "gmn": "bagaimana",
    "gimana": "bagaimana",
    "knp": "kenapa",
    "msh": "masih",
    "dlm": "dalam",
    "trs": "terus",
    "trus": "terus",
    "skrg": "sekarang",
    "gt": "begitu",
    "gitu": "begitu",
    "bgtu": "begitu",
    "ancur": "hancur",
    "payah": "payah",
    "hebat": "hebat",
    "sukses": "sukses",
    "kecewa": "kecewa",
    "gagal": "gagal",
    "hoax": "bohong",
    "tipu": "tipu",
    "gokil": "hebat",
    "zonk": "buruk",
    "kacau": "kacau",
    "baper": "terbawa perasaan",
    "lebay": "berlebihan",
    "jos": "bagus",
}

CUSTOM_STOPWORDS: frozenset[str] = frozenset(
    {
        "dan",
        "atau",
        "yg",
        "juga",
        "dari",
        "di",
        "ke",
        "nya",
        "roy",
        "sih",
        "deh",
        "dong",
        "kok",
        "kan",
        "ini",
        "itu",
        "aja",
        "kalau",
        "kalo",
        "buat",
    }
)


@lru_cache(maxsize=1)
def _get_offline_slang_cache() -> dict[str, str]:
    """Return a cached copy of the built-in Indonesian slang dictionary."""
    return dict(CORE_SLANG_MAP)


def _fetch_remote_slang_csv(url: str, timeout: float) -> dict[str, str]:
    """Download and parse remote colloquial CSV lexicon with explicit timeout."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    if "slang" not in frame.columns or "formal" not in frame.columns:
        raise ValueError("Remote lexicon CSV must contain 'slang' and 'formal' columns.")

    cleaned = frame.dropna(subset=["slang", "formal"])
    slang_terms = cleaned["slang"].astype(str).str.strip().str.lower()
    formal_terms = cleaned["formal"].astype(str).str.strip().str.lower()
    return {
        slang: formal
        for slang, formal in zip(slang_terms, formal_terms, strict=True)
        if slang and formal
    }


def load_slang_dictionary(
    url: str | None = None,
    allow_remote: bool = False,
    timeout: float = 5.0,
) -> dict[str, str]:
    """Load Indonesian slang normalization dictionary with safe offline fallback.

    Args:
        url: Optional remote CSV URL containing ``slang`` and ``formal`` columns.
        allow_remote: Whether network lookup is permitted. Defaults to ``False``.
        timeout: Explicit HTTP request timeout in seconds.

    Returns:
        Non-empty dictionary mapping colloquial tokens to formal Indonesian terms.

    Raises:
        ValueError: If ``timeout`` is not strictly positive.
    """
    if timeout <= 0:
        raise ValueError("HTTP timeout must be greater than zero.")

    fallback = dict(_get_offline_slang_cache())
    if not allow_remote or not url:
        logger.info("Using built-in offline slang lexicon (%d entries).", len(fallback))
        return fallback

    try:
        remote_entries = _fetch_remote_slang_csv(url=url, timeout=timeout)
    except (requests.RequestException, ValueError, KeyError, UnicodeDecodeError) as exc:
        logger.warning(
            "Remote slang fetch failed (%s); falling back to offline lexicon (%d entries).",
            exc,
            len(fallback),
        )
        return fallback

    merged = {**fallback, **remote_entries}
    logger.info(
        "Loaded %d slang entries (%d remote + %d core fallback).",
        len(merged),
        len(remote_entries),
        len(fallback),
    )
    return merged


def normalize_slang(
    tokens: list[str],
    slang_dict: Mapping[str, str] | None = None,
) -> list[str]:
    """Normalize Indonesian slang tokens using a provided or built-in dictionary.

    Args:
        tokens: Pre-cleaned lowercase word tokens.
        slang_dict: Optional custom slang lookup dictionary.

    Returns:
        List of normalized tokens (multi-word expansions are split cleanly).
    """
    if not tokens:
        return []

    lookup = slang_dict if slang_dict is not None else _get_offline_slang_cache()
    normalized: list[str] = []
    for token in tokens:
        mapped = lookup.get(token, token)
        normalized.extend(part for part in mapped.split() if part)
    return normalized
