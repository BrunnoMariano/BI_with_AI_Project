"""Limpeza da base de vendas: moeda, datas, promotora e DataFrame canônico.

Funções puras sobre pandas; não dependem de Streamlit, Firebase nem OpenAI.
"""

from __future__ import annotations

import math
import re
import unicodedata

import pandas as pd

from ml.schema import to_canonical

DATETIME_BR_FORMAT = "%d/%m/%Y %H:%M:%S"
DATE_BR_FORMAT = "%d/%m/%Y"

# Respostas que significam "sem promotora" (já sem acento e em minúsculas).
_PROMOTORA_VAZIA = re.compile(r"^(n+a+o+|nenhum[a]?|n/?a|-+|\.+|x+)$")
_PROMOTORA_NAO_SE_APLICA = re.compile(r"\bn+a+o+\b.*\baplica")


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return value is pd.NA or value is pd.NaT


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def _clean_text(series: pd.Series) -> pd.Series:
    """Remove espaços extras; texto vazio vira None."""
    cleaned = series.map(lambda v: None if _is_missing(v) else re.sub(r"\s+", " ", str(v)).strip())
    return cleaned.map(lambda v: v or None)


def parse_brl(value) -> float:
    """Converte moeda brasileira em float: "R$1.234,56" -> 1234.56. Vazio/inválido -> NaN."""
    if _is_missing(value):
        return float("nan")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).replace("R$", "").replace("\xa0", "").replace(" ", "").strip()
    if not text:
        return float("nan")
    text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_datetime_br(value):
    """Converte "dd/mm/aaaa hh:mm:ss" (ou só "dd/mm/aaaa") em datetime. Aceita valor único ou Series.

    Valores vazios ou inválidos viram NaT.
    """
    if not isinstance(value, pd.Series):
        return parse_datetime_br(pd.Series([value])).iloc[0]
    if pd.api.types.is_datetime64_any_dtype(value):
        return value
    text = value.map(lambda v: None if _is_missing(v) else str(v).strip())
    parsed = pd.to_datetime(text, format=DATETIME_BR_FORMAT, errors="coerce")
    missing = parsed.isna() & text.notna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], format=DATE_BR_FORMAT, errors="coerce")
    return parsed


def normalize_promotora(value) -> str | None:
    """Padroniza a promotora: minúsculo, sem acento, espaços simples.

    "Não se aplica" (e variações) e respostas vazias viram None;
    qualquer grafia com "bevi" (BEVI, Bevi Cred, bevicred...) vira "bevi".
    """
    if _is_missing(value):
        return None
    text = _strip_accents(str(value)).lower()
    text = re.sub(r"\s+", " ", text).strip(" .,;:!-_")
    if not text or _PROMOTORA_VAZIA.match(text) or _PROMOTORA_NAO_SE_APLICA.search(text):
        return None
    if "bevi" in text:
        return "bevi"
    return text


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve o DataFrame canônico e limpo, com as colunas derivadas pago, is_b2b e valor_zero.

    - valor em float; data_pedido e data_pagamento em datetime;
    - status e forma_pagamento em minúsculas; cliente em minúsculas (chave do cliente);
    - promotora normalizada; conta vazia vira None;
    - pago: status == "pago"; is_b2b: conta preenchida; valor_zero: valor == 0.
    """
    orders = to_canonical(df)

    for column in ("produto", "cupom", "pagador", "conta", "cidade", "uf", "status", "forma_pagamento", "cliente"):
        if column in orders:
            orders[column] = _clean_text(orders[column])
    for column in ("status", "forma_pagamento", "cliente"):
        if column in orders:
            orders[column] = orders[column].str.lower()
    if "uf" in orders:
        orders["uf"] = orders["uf"].str.upper()

    if "valor" in orders:
        orders["valor"] = orders["valor"].map(parse_brl).astype(float)
    for column in ("data_pedido", "data_pagamento"):
        if column in orders:
            orders[column] = parse_datetime_br(orders[column])
    if "promotora" in orders:
        orders["promotora"] = orders["promotora"].map(normalize_promotora)

    orders["pago"] = orders["status"].eq("pago") if "status" in orders else False
    orders["is_b2b"] = orders["conta"].notna() if "conta" in orders else False
    orders["valor_zero"] = orders["valor"].eq(0) if "valor" in orders else False
    return orders
