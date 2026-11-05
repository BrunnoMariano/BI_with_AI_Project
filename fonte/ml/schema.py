"""Esquema canônico da base de vendas.

Traduz os nomes de coluna da extração do cliente (ex.: "Criado em", "Valor Pago")
para nomes canônicos usados pelo módulo de ML (ex.: "data_pedido", "valor").
Funções puras: não dependem de Streamlit, Firebase nem OpenAI.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Nome canônico -> nomes aceitos na extração, em ordem de preferência.
# O primeiro de cada lista é o nome oficial de vendas_cliente.xlsx; os demais
# são nomes alternativos (Primeiro item, Total brl, Status, Pago em, Pago por).
COLUMN_ALIASES: dict[str, list[str]] = {
    "pedido_id": ["Número"],
    "data_pedido": ["Criado em"],
    "data_pagamento": ["Data Venda", "Pago em"],
    "valor": ["Valor Pago", "Total brl"],
    "status": ["Status Venda", "Status"],
    "produto": ["Produto", "Primeiro item"],
    "cupom": ["Cupom"],
    "promotora": ["Promotora"],
    "forma_pagamento": ["Forma de pagamento"],
    "cliente": ["Comprador"],
    "pagador": ["Nome Pagamento", "Pago por"],
    "conta": ["Conta"],
    "uf": ["Estado"],
    "cidade": ["Cidade"],
}

# Campos canônicos que, juntos, identificam a base de vendas do cliente.
# Equivalem às colunas Produto, Valor Pago, Status Venda, Forma de pagamento,
# Data Venda e Criado em da extração oficial.
SPECIALIZED_FILE_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"produto", "valor", "status", "forma_pagamento", "data_pagamento", "data_pedido"}
)

SALES_PROFILE_ID = "client_sales_profile_v1"
GENERIC_PROFILE_ID = "generic_file_v1"


def normalize_column_label(value) -> str:
    """Normaliza um nome de coluna: sem acento, minúsculo, espaços simples."""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip().lower()


def detect_schema(df: pd.DataFrame) -> dict[str, str]:
    """Retorna {nome_canônico: nome_da_coluna_no_df} para as colunas reconhecidas."""
    by_label: dict[str, str] = {}
    for column in df.columns:
        by_label.setdefault(normalize_column_label(column), column)

    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            found = by_label.get(normalize_column_label(alias))
            if found is not None:
                mapping[canonical] = found
                break
    return mapping


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma cópia só com as colunas reconhecidas, renomeadas para os nomes canônicos.

    Colunas não mapeadas (telefone, CEP, logradouro...) ficam de fora.
    """
    mapping = detect_schema(df)
    ordered = [canonical for canonical in COLUMN_ALIASES if canonical in mapping]
    result = df[[mapping[canonical] for canonical in ordered]].copy()
    result.columns = ordered
    return result


def detect_dataset_profile(df: pd.DataFrame, file_name: str = "") -> str:
    """Retorna "client_sales_profile_v1" se o df tem os campos de vendas exigidos; senão "generic_file_v1"."""
    if SPECIALIZED_FILE_REQUIRED_COLUMNS.issubset(detect_schema(df)):
        return SALES_PROFILE_ID
    return GENERIC_PROFILE_ID
