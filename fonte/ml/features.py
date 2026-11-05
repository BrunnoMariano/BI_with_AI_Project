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


GRANULARIDADES = ("dia", "semana", "mes")
DIAS_POR_MES = 30.4375


def periodos(datas, granularidade: str = "dia", origem=None) -> pd.DatetimeIndex:
    """Data de início do período de cada dia.

    - "dia": o próprio dia;
    - "semana": blocos de 7 dias alinhados à `origem` (início da previsão), para que
      nenhuma semana misture dias reais e previstos;
    - "mes": primeiro dia do mês do calendário.
    """
    indice = pd.DatetimeIndex(datas).normalize()
    if granularidade == "dia":
        return indice
    if granularidade == "semana":
        origem = indice.min() if origem is None else pd.Timestamp(origem).normalize()
        deslocamento = np.floor((indice - origem).days / 7).astype(int) * 7
        return pd.DatetimeIndex(origem + pd.to_timedelta(deslocamento, unit="D"))
    if granularidade == "mes":
        return indice.to_period("M").to_timestamp()
    raise ValueError(f"granularidade deve ser uma de {GRANULARIDADES}, recebido: {granularidade!r}")


def dias_no_periodo(inicio_periodo, granularidade: str) -> np.ndarray:
    """Quantos dias tem um período completo (1, 7 ou os dias do mês)."""
    inicio_periodo = pd.DatetimeIndex(inicio_periodo)
    if granularidade == "dia":
        return np.ones(len(inicio_periodo), dtype=int)
    if granularidade == "semana":
        return np.full(len(inicio_periodo), 7, dtype=int)
    return np.asarray(inicio_periodo.days_in_month, dtype=int)


def agregar_serie(serie: pd.Series, granularidade: str = "dia", origem=None) -> pd.DataFrame:
    """Soma a série diária por período. Devolve periodo, real, n_dias e parcial (período incompleto)."""
    chaves = periodos(serie.index, granularidade, origem)
    agrupado = pd.DataFrame({"periodo": chaves, "real": np.asarray(serie, dtype=float)}).groupby("periodo")
    resultado = agrupado.agg(real=("real", "sum"), n_dias=("real", "size")).reset_index()
    resultado["parcial"] = resultado["n_dias"] < dias_no_periodo(resultado["periodo"], granularidade)
    return resultado


def summarize_b2b(df_clean: pd.DataFrame, inicio, fim) -> dict:
    """Resumo descritivo das vendas corporativas (B2B) pagas: histórico completo e janela de treino.

    B2B não entra na previsão (decisão 3 do CLAUDE.md); este resumo só descreve o passado.
    """
    inicio, fim = pd.Timestamp(inicio).normalize(), pd.Timestamp(fim).normalize()
    pagos = df_clean.loc[df_clean["pago"].astype(bool) & df_clean["data_pedido"].notna()]
    b2b = pagos.loc[pagos["is_b2b"].astype(bool)]
    dias = b2b["data_pedido"].dt.normalize()
    janela = b2b.loc[(dias >= inicio) & (dias <= fim)]

    dias_pagos = pagos["data_pedido"].dt.normalize()
    receita_total_janela = float(pagos.loc[(dias_pagos >= inicio) & (dias_pagos <= fim), "valor"].sum())
    meses = ((fim - inicio).days + 1) / DIAS_POR_MES
    return {
        "historico": {
            "pedidos": int(len(b2b)),
            "receita": float(b2b["valor"].sum()),
            "inicio": dias.min().date().isoformat() if len(b2b) else None,
            "fim": dias.max().date().isoformat() if len(b2b) else None,
        },
        "janela": {
            "pedidos": int(len(janela)),
            "receita": float(janela["valor"].sum()),
            "inicio": inicio.date().isoformat(),
            "fim": fim.date().isoformat(),
            "meses": float(meses),
            "media_mensal_pedidos": float(len(janela) / meses),
            "media_mensal_receita": float(janela["valor"].sum() / meses),
            "ticket_medio": float(janela["valor"].mean()) if len(janela) else 0.0,
            "participacao_receita": float(janela["valor"].sum() / receita_total_janela) if receita_total_janela else 0.0,
        },
    }


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
