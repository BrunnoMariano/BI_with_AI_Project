"""Construção das séries temporais e variáveis de calendário para a previsão.

Funções puras sobre pandas; não dependem de Streamlit, Firebase nem OpenAI.
Recebem o DataFrame já limpo por ml.preprocessing.clean_orders.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

ALVOS = ("pedidos", "receita")
MIN_PAGOS_POR_MES = 100


def _pagos(df_clean: pd.DataFrame, excluir_b2b: bool) -> pd.DataFrame:
    """Pedidos pagos com data_pedido válida (opcionalmente só B2C)."""
    mask = df_clean["pago"].astype(bool) & df_clean["data_pedido"].notna()
    if excluir_b2b and "is_b2b" in df_clean:
        mask &= ~df_clean["is_b2b"].astype(bool)
    return df_clean.loc[mask]


def detect_training_start(
    df_clean: pd.DataFrame, min_pagos: int = MIN_PAGOS_POR_MES, excluir_b2b: bool = True
) -> pd.Timestamp:
    """Primeiro dia do primeiro mês com pelo menos `min_pagos` pedidos pagos.

    Se nenhum mês atingir o mínimo, devolve o dia do primeiro pedido pago.
    """
    pagos = _pagos(df_clean, excluir_b2b)
    if pagos.empty:
        raise ValueError("Não há pedidos pagos com data para montar a série.")
    por_mes = pagos["data_pedido"].dt.to_period("M").value_counts().sort_index()
    meses_ok = por_mes[por_mes >= min_pagos]
    if meses_ok.empty:
        return pagos["data_pedido"].min().normalize()
    return meses_ok.index[0].to_timestamp()


def build_daily_series(
    df_clean: pd.DataFrame,
    alvo: str = "pedidos",
    inicio=None,
    excluir_b2b: bool = True,
    descartar_ultimo_dia: bool = True,
) -> pd.Series:
    """Série diária de pedidos pagos (contagem ou receita) por data_pedido.

    - alvo="pedidos" conta pedidos pagos; alvo="receita" soma o valor pago;
    - dias sem venda entram com 0 (índice diário contínuo, freq="D");
    - exclui B2B por padrão (decisão 3 do CLAUDE.md);
    - inicio=None detecta o primeiro mês com >= 100 pagos;
    - descarta o último dia da extração, que está incompleto.
    """
    if alvo not in ALVOS:
        raise ValueError(f"alvo deve ser um de {ALVOS}, recebido: {alvo!r}")

    inicio = detect_training_start(df_clean, excluir_b2b=excluir_b2b) if inicio is None else pd.Timestamp(inicio)
    inicio = inicio.normalize()

    # O último dia é o da extração inteira (qualquer status), não só dos pagos.
    ultimo_dia = df_clean["data_pedido"].max().normalize()
    fim = ultimo_dia - pd.Timedelta(days=1) if descartar_ultimo_dia else ultimo_dia
    if fim < inicio:
        raise ValueError(f"Período vazio: início {inicio.date()} depois do fim {fim.date()}.")

    pagos = _pagos(df_clean, excluir_b2b)
    dias = pagos["data_pedido"].dt.normalize()
    pagos = pagos.loc[(dias >= inicio) & (dias <= fim)]
    dias = dias.loc[pagos.index]

    if alvo == "pedidos":
        diario = pagos.groupby(dias).size().astype(float)
    else:
        diario = pagos.groupby(dias)["valor"].sum().astype(float)

    indice = pd.date_range(inicio, fim, freq="D", name="data")
    serie = diario.reindex(indice, fill_value=0.0)
    serie.name = alvo
    return serie


def calendar_features(datas) -> pd.DataFrame:
    """Variáveis de calendário por dia: dia da semana, dia do mês, fim de semana e feriado nacional."""
    indice = pd.DatetimeIndex(datas)
    anos = range(indice.year.min(), indice.year.max() + 1) if len(indice) else []
    feriados = holidays.Brazil(years=anos)
    return pd.DataFrame(
        {
            "dia_semana": indice.dayofweek,
            "dia_mes": indice.day,
            "fim_de_semana": (indice.dayofweek >= 5).astype(int),
            "feriado": np.array([d in feriados for d in indice.date], dtype=int),
        },
        index=indice,
    )
