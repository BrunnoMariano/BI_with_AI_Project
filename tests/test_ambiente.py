"""Confere se o ambiente está pronto para o módulo de ML (Fase 0)."""

import importlib

import pytest

BIBLIOTECAS_ML = ["sklearn", "statsmodels", "lightgbm", "joblib", "holidays", "pytest"]

COLUNAS_ESPERADAS = [
    "Número",
    "Conta",
    "Comprador",
    "Produto",
    "Valor Pago",
    "Nome Pagamento",
    "Forma de pagamento",
    "Link de pagamento",
    "Status Venda",
    "Promotora",
    "Cupom",
    "Telefone/Celular",
    "CEP",
    "Logradouro",
    "Número.1",
    "Bairro",
    "Cidade",
    "Estado",
    "Data Venda",
    "Criado em",
]


@pytest.mark.parametrize("nome", BIBLIOTECAS_ML)
def test_bibliotecas_ml_importam(nome):
    importlib.import_module(nome)


def test_planilha_tem_linhas_e_colunas_esperadas(vendas_df):
    assert len(vendas_df) == 4558
    assert len(COLUNAS_ESPERADAS) == 20
    assert sorted(vendas_df.columns) == sorted(COLUNAS_ESPERADAS)
