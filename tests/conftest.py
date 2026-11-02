"""Fixtures compartilhadas pelos testes do módulo de ML."""

import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
CAMINHO_PLANILHA = RAIZ_PROJETO / "data" / "vendas_cliente.xlsx"

# Permite "from ml import ..." nos testes, igual ao app que roda de dentro de fonte/.
sys.path.insert(0, str(RAIZ_PROJETO / "fonte"))


@pytest.fixture(scope="session")
def vendas_df() -> pd.DataFrame:
    """Planilha oficial do cliente (aba Worksheet). Pula o teste se o arquivo não existir."""
    if not CAMINHO_PLANILHA.exists():
        pytest.skip(f"Planilha não encontrada em {CAMINHO_PLANILHA}; coloque vendas_cliente.xlsx em data/.")
    return pd.read_excel(CAMINHO_PLANILHA, sheet_name="Worksheet")
