"""Orquestra a previsão de vendas: limpeza -> séries diárias -> backtest -> melhor modelo -> previsão.

Função pura sobre o DataFrame bruto (como lido do Excel); não depende de Streamlit
nem de OpenAI. O Firebase só é usado se salvar=True e uid/id_token forem informados.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ml import registry
from ml.features import ALVOS, build_daily_series, detect_training_start, summarize_b2b
from ml.forecasting import HORIZONTE_PADRAO, N_CORTES_PADRAO, fit_best, residuos_longos
from ml.preprocessing import clean_orders
from ml.schema import COLUMN_ALIASES, detect_schema

VERSAO_PIPELINE = "fase3_v1"
COLUNAS_NECESSARIAS = ("data_pedido", "status", "valor")
DESCRICAO_COLUNAS = {
    "data_pedido": "data de criação do pedido",
    "status": "status da venda (para saber quais foram pagas)",
    "valor": "valor pago",
}


def verificar_requisitos(df_raw: pd.DataFrame) -> list[str]:
    """Lista, em português, o que falta na base para rodar a previsão (vazia = pode rodar)."""
    mapeadas = detect_schema(df_raw)
    faltando = []
    for coluna in COLUNAS_NECESSARIAS:
        if coluna not in mapeadas:
            nomes = " ou ".join(f'"{nome}"' for nome in COLUMN_ALIASES[coluna])
            faltando.append(f"{DESCRICAO_COLUNAS[coluna]} (coluna {nomes})")
    return faltando


def run_predictive_pipeline(
    df_raw: pd.DataFrame,
    source_key: str,
    horizonte: int = HORIZONTE_PADRAO,
    n_cortes: int = N_CORTES_PADRAO,
    alvos=ALVOS,
    inicio=None,
    horizonte_previsao: int | None = None,
    salvar: bool = False,
    uid: str | None = None,
    id_token: str | None = None,
    pasta=None,
) -> dict:
    """Roda a previsão de pedidos e receita (B2C) e devolve um dict com:

    - "previsoes": {alvo: DataFrame data, previsto, lim_inf_80, lim_sup_80, lim_inf_95, lim_sup_95};
    - "modelos": {alvo: tabela do backtest, média dos cortes, ordenada por WAPE};
    - "historico": {alvo: DataFrame data, real} usado no treino;
    - "residuos": {alvo: DataFrame corte, passo, residuo} do vencedor no backtest (base das faixas);
    - "b2b": resumo descritivo das vendas corporativas (não entram na previsão);
    - "metadados": datas de treino, n_obs, modelo vencedor, cortes do backtest etc.

    O backtest sempre usa `horizonte` (28 dias); `horizonte_previsao` permite prever mais
    dias (ex.: 90) com o mesmo modelo. Com salvar=True grava via registry.
    """
    faltando = verificar_requisitos(df_raw)
    if faltando:
        raise ValueError("A base não tem as colunas necessárias para a previsão. Faltando: " + "; ".join(faltando))

    horizonte_previsao = horizonte if horizonte_previsao is None else int(horizonte_previsao)
    pedidos = clean_orders(df_raw)
    inicio = detect_training_start(pedidos) if inicio is None else pd.Timestamp(inicio)

    previsoes, modelos, historico, residuos, meta_alvos = {}, {}, {}, {}, {}
    for alvo in alvos:
        serie = build_daily_series(pedidos, alvo=alvo, inicio=inicio)
        ajuste = fit_best(serie, horizonte=horizonte, n_cortes=n_cortes)
        previsao = ajuste.forecast(horizonte_previsao)

        previsoes[alvo] = previsao
        modelos[alvo] = ajuste.tabela
        historico[alvo] = pd.DataFrame({"data": serie.index, "real": serie.to_numpy()})
        residuos[alvo] = residuos_longos(ajuste.residuos)
        meta_alvos[alvo] = {
            "modelo_vencedor": ajuste.modelo,
            "wape_vencedor": float(ajuste.tabela.iloc[0]["wape"]),
            "inicio_treino": serie.index[0].date().isoformat(),
            "fim_treino": serie.index[-1].date().isoformat(),
            "n_obs": int(len(serie)),
            "dias_sem_venda": int((serie == 0).sum()),
            "total_treino": float(serie.sum()),
            "inicio_previsao": previsao["data"].iloc[0].date().isoformat(),
            "fim_previsao": previsao["data"].iloc[-1].date().isoformat(),
            "total_previsto": float(previsao["previsto"].sum()),
            "cortes_backtest": [
                {
                    "corte": c["corte"],
                    "inicio_teste": c["inicio_teste"].date().isoformat(),
                    "fim_teste": c["fim_teste"].date().isoformat(),
                }
                for c in ajuste.backtest["cortes"]
            ],
            "modelos_com_falha": ajuste.backtest["falhas"],
        }

    resultado = {
        "source_key": str(source_key),
        "previsoes": previsoes,
        "modelos": modelos,
        "historico": historico,
        "residuos": residuos,
        "b2b": summarize_b2b(pedidos, serie.index[0], serie.index[-1]),
        "metadados": {
            "versao": VERSAO_PIPELINE,
            "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "horizonte": int(horizonte),
            "horizonte_previsao": int(horizonte_previsao),
            "n_cortes": int(n_cortes),
            "apenas_b2c": True,
            "metrica_principal": "wape",
            "alvos": meta_alvos,
        },
    }

    if salvar:
        resultado["metadados"]["salvo_em"] = registry.salvar_resultado(resultado, source_key, uid, id_token, pasta)
    return resultado
