"""Alertas de anomalia: dias em que o real saiu da faixa de 95% prevista no backtest.

Função pura sobre pandas; não depende de Streamlit, Firebase nem OpenAI.
"""

from __future__ import annotations

import pandas as pd

COLUNAS_BACKTEST = ("data", "previsto", "lim_inf_95", "lim_sup_95")


def detect_anomalies(serie_real: pd.Series, backtest: pd.DataFrame) -> pd.DataFrame:
    """Marca cada dia do backtest como anômalo ou não, comparando o real com a faixa de 95%.

    `serie_real` tem índice diário; `backtest` vem de ml.forecasting.faixas_backtest (um modelo só).
    Devolve data, real, previsto, lim_inf_95, lim_sup_95, anomalia (bool), direcao ("acima",
    "abaixo" ou None) e desvio_pct: (real - previsto) / previsto em %, com sinal (NaN se previsto = 0).
    Só entram os dias presentes nos dois lados; exatamente no limite ainda é normal.
    """
    faltando = [coluna for coluna in COLUNAS_BACKTEST if coluna not in backtest]
    if faltando:
        raise ValueError("O backtest não tem as colunas necessárias: " + ", ".join(faltando))

    faixa = backtest[list(COLUNAS_BACKTEST)].copy()
    faixa["data"] = pd.to_datetime(faixa["data"]).dt.normalize()
    if faixa["data"].duplicated().any():
        raise ValueError("O backtest tem datas repetidas; informe as faixas de um único modelo.")

    real = pd.DataFrame(
        {"data": pd.DatetimeIndex(serie_real.index).normalize(), "real": pd.to_numeric(serie_real).to_numpy(dtype=float)}
    )
    dias = faixa.merge(real, on="data", how="inner").sort_values("data", ignore_index=True)

    acima = dias["real"] > dias["lim_sup_95"]
    abaixo = dias["real"] < dias["lim_inf_95"]
    dias["anomalia"] = acima | abaixo
    dias["direcao"] = None
    dias.loc[acima, "direcao"] = "acima"
    dias.loc[abaixo, "direcao"] = "abaixo"
    previsto_valido = dias["previsto"].where(dias["previsto"] > 0)
    dias["desvio_pct"] = (dias["real"] - previsto_valido) / previsto_valido * 100
    return dias[["data", "real", "previsto", "lim_inf_95", "lim_sup_95", "anomalia", "direcao", "desvio_pct"]]
