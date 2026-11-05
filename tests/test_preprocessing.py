"""Testes da Fase 1: esquema canônico (ml.schema) e limpeza (ml.preprocessing).

Os testes com dados sintéticos rodam sempre; os que usam a planilha oficial
são pulados se data/vendas_cliente.xlsx não existir.
"""

import math

import pandas as pd
import pytest

from ml.preprocessing import clean_orders, normalize_promotora, parse_brl, parse_datetime_br
from ml.schema import (
    COLUMN_ALIASES,
    SPECIALIZED_FILE_REQUIRED_COLUMNS,
    detect_dataset_profile,
    detect_schema,
    to_canonical,
)


def _pedidos_sinteticos() -> pd.DataFrame:
    """Três pedidos fictícios com os nomes de coluna da extração oficial."""
    return pd.DataFrame(
        {
            "Número": [1, 2, 3],
            "Conta": ["Empresa X", None, "  "],
            "Comprador": ["  Fulano  DE Tal ", "ciclano", "Beltrano"],
            "Produto": ["Certificação A", "Certificação B", None],
            "Valor Pago": ["R$1.234,56", "R$99,90", "R$0,00"],
            "Nome Pagamento": ["Fulano", None, None],
            "Forma de pagamento": ["PIX", None, None],
            "Status Venda": ["Pago", "pendente", "cancelado"],
            "Promotora": ["BEVI CRED", "Não se aplica.", "Alcif"],
            "Cupom": ["CUPOM10", None, None],
            "Telefone/Celular": ["(00) 0000-0000", None, None],
            "Cidade": ["São Paulo", "Recife", None],
            "Estado": ["sp", "PE", None],
            "Data Venda": ["02/01/2026 10:00:00", None, None],
            "Criado em": ["01/01/2026 09:30:15", "05/01/2026 12:00:00", "06/01/2026 08:00:00"],
        }
    )


# ---------- schema.py ----------

def test_detect_schema_mapeia_nomes_oficiais():
    mapping = detect_schema(_pedidos_sinteticos())
    assert mapping["data_pedido"] == "Criado em"
    assert mapping["data_pagamento"] == "Data Venda"
    assert mapping["valor"] == "Valor Pago"
    assert mapping["status"] == "Status Venda"
    assert mapping["cliente"] == "Comprador"
    assert mapping["pagador"] == "Nome Pagamento"
    assert mapping["uf"] == "Estado"
    assert "Telefone/Celular" not in mapping.values()


def test_detect_schema_aceita_nomes_alternativos_e_variacoes_de_grafia():
    df = pd.DataFrame(
        columns=["Primeiro item", "Total BRL", " status ", "Pago em", "Pago por", "criado EM", "Forma de Pagamento"]
    )
    mapping = detect_schema(df)
    assert mapping["produto"] == "Primeiro item"
    assert mapping["valor"] == "Total BRL"
    assert mapping["status"] == " status "
    assert mapping["data_pagamento"] == "Pago em"
    assert mapping["pagador"] == "Pago por"
    assert mapping["data_pedido"] == "criado EM"


def test_detect_schema_prefere_nome_oficial_quando_ha_dois():
    df = pd.DataFrame(columns=["Status", "Status Venda"])
    assert detect_schema(df)["status"] == "Status Venda"


def test_numero_1_nao_e_confundido_com_numero():
    df = pd.DataFrame(columns=["Número.1"])
    assert "pedido_id" not in detect_schema(df)


def test_to_canonical_renomeia_e_descarta_colunas_nao_mapeadas():
    canon = to_canonical(_pedidos_sinteticos())
    assert list(canon.columns) == [c for c in COLUMN_ALIASES if c in canon.columns]
    assert "Telefone/Celular" not in canon.columns
    assert canon["valor"].tolist() == ["R$1.234,56", "R$99,90", "R$0,00"]


def test_detect_dataset_profile_sintetico():
    assert detect_dataset_profile(_pedidos_sinteticos()) == "client_sales_profile_v1"
    sem_status = _pedidos_sinteticos().drop(columns=["Status Venda"])
    assert detect_dataset_profile(sem_status) == "generic_file_v1"
    assert detect_dataset_profile(pd.DataFrame({"a": [1]})) == "generic_file_v1"


def test_colunas_exigidas_sao_canonicas():
    assert SPECIALIZED_FILE_REQUIRED_COLUMNS.issubset(COLUMN_ALIASES)


# ---------- preprocessing.py ----------

@pytest.mark.parametrize(
    "texto, esperado",
    [("R$1.234,56", 1234.56), ("R$99,99", 99.99), ("R$ 0,00", 0.0), ("R$12.345.678,90", 12345678.90), (150, 150.0)],
)
def test_parse_brl(texto, esperado):
    assert parse_brl(texto) == pytest.approx(esperado)


@pytest.mark.parametrize("vazio", [None, float("nan"), "", "abc"])
def test_parse_brl_vazio_ou_invalido_vira_nan(vazio):
    assert math.isnan(parse_brl(vazio))


def test_parse_datetime_br_valor_unico_e_series():
    assert parse_datetime_br("07/04/2026 13:05:09") == pd.Timestamp(2026, 4, 7, 13, 5, 9)
    serie = parse_datetime_br(pd.Series(["01/02/2025 00:00:00", None, "lixo", "03/02/2025"]))
    assert serie.iloc[0] == pd.Timestamp(2025, 2, 1)  # dia/mês, não mês/dia
    assert pd.isna(serie.iloc[1]) and pd.isna(serie.iloc[2])
    assert serie.iloc[3] == pd.Timestamp(2025, 2, 3)


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("BEVI", "bevi"),
        ("Bevi Cred", "bevi"),
        ("bevicred", "bevi"),
        ("  Cia do Crédito ", "cia do credito"),
        ("Sólida  Promotora", "solida promotora"),
        ("Não se aplica", None),
        ("não se aplica.", None),
        ("NÃO", None),
        ("nenhuma", None),
        ("", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_normalize_promotora(texto, esperado):
    assert normalize_promotora(texto) == esperado


def test_clean_orders_sintetico():
    orders = clean_orders(_pedidos_sinteticos())
    assert orders["valor"].tolist() == pytest.approx([1234.56, 99.90, 0.0])
    assert orders["pago"].tolist() == [True, False, False]
    assert orders["is_b2b"].tolist() == [True, False, False]  # conta só com espaços não é B2B
    assert orders["valor_zero"].tolist() == [False, False, True]
    assert orders["status"].tolist() == ["pago", "pendente", "cancelado"]
    assert orders.loc[0, "cliente"] == "fulano de tal"
    assert orders.loc[0, "forma_pagamento"] == "pix"
    assert orders["uf"].tolist()[:2] == ["SP", "PE"]
    assert orders["promotora"].tolist() == ["bevi", None, "alcif"]
    assert orders.loc[0, "data_pedido"] == pd.Timestamp(2026, 1, 1, 9, 30, 15)
    assert pd.isna(orders.loc[1, "data_pagamento"])
    assert pd.api.types.is_datetime64_any_dtype(orders["data_pedido"])


# ---------- planilha oficial (números de conferência do CLAUDE.md) ----------

def test_planilha_ativa_perfil_de_vendas(vendas_df):
    assert detect_dataset_profile(vendas_df, "vendas_cliente.xlsx") == "client_sales_profile_v1"


def test_planilha_numeros_de_conferencia(vendas_df):
    orders = clean_orders(vendas_df)
    pagos = orders[orders["pago"]]
    assert len(orders) == 4558
    assert len(pagos) == 2919
    assert pagos["valor"].sum() == pytest.approx(308783.48, abs=0.005)
    b2b_pagos = pagos[pagos["is_b2b"]]
    assert len(b2b_pagos) == 82
    assert b2b_pagos["valor"].sum() == pytest.approx(88410.08, abs=0.005)
    assert int(orders["is_b2b"].sum()) == 138
    assert int(orders["valor_zero"].sum()) == 14


def test_planilha_datas_e_problemas_conhecidos(vendas_df):
    orders = clean_orders(vendas_df)
    assert orders["data_pedido"].notna().all()
    assert orders["valor"].notna().all()
    assert orders["data_pedido"].min().date() == pd.Timestamp(2024, 9, 5).date()
    assert orders["data_pedido"].max().date() == pd.Timestamp(2026, 4, 7).date()
    assert int((orders["pago"] & orders["data_pagamento"].isna()).sum()) == 4
    assert int((orders["data_pagamento"] < orders["data_pedido"]).sum()) == 22


def test_planilha_promotora_agrupa_bevi(vendas_df):
    orders = clean_orders(vendas_df)
    bruto = vendas_df["Promotora"].dropna().nunique()
    assert orders["promotora"].nunique() < bruto
    com_bevi = orders["promotora"].dropna()
    assert set(com_bevi[com_bevi.str.contains("bevi")]) == {"bevi"}
    assert not orders["promotora"].dropna().str.contains("aplica").any()
