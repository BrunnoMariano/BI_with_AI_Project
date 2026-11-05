"""Orquestra a previsão de vendas: limpeza -> séries diárias -> backtest -> melhor modelo -> previsão.

Função pura sobre o DataFrame bruto (como lido do Excel); não depende de Streamlit
nem de OpenAI. O Firebase só é usado se salvar=True e uid/id_token forem informados.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ml import registry
from ml.features import ALVOS, build_daily_series, detect_training_start
from ml.forecasting import HORIZONTE_PADRAO, N_CORTES_PADRAO, fit_best
from ml.preprocessing import clean_orders
from ml.schema import detect_schema

VERSAO_PIPELINE = "fase2_v1"
COLUNAS_NECESSARIAS = ("data_pedido", "status", "valor")


def run_predictive_pipeline(
    df_raw: pd.DataFrame,
    source_key: str,
    horizonte: int = HORIZONTE_PADRAO,
    n_cortes: int = N_CORTES_PADRAO,
    alvos=ALVOS,
    inicio=None,
    salvar: bool = False,
    uid: str | None = None,
    id_token: str | None = None,
    pasta=None,
) -> dict:
    """Roda a previsão de pedidos e receita (B2C) e devolve um dict com:

    - "previsoes": {alvo: DataFrame data, previsto, lim_inf_80, lim_sup_80, lim_inf_95, lim_sup_95};
    - "modelos": {alvo: tabela do backtest, média dos cortes, ordenada por WAPE};
    - "historico": {alvo: DataFrame data, real} usado no treino;
    - "metadados": datas de treino, n_obs, modelo vencedor, cortes do backtest etc.

    Com salvar=True grava o resultado via registry (joblib local e, se houver uid/id_token, Firestore).
    """
    faltando = [c for c in COLUNAS_NECESSARIAS if c not in detect_schema(df_raw)]
    if faltando:
        raise ValueError(
            "A base não tem as colunas necessárias para a previsão "
            "(data de criação, status e valor do pedido). Faltando: " + ", ".join(faltando)
        )

    pedidos = clean_orders(df_raw)
    inicio = detect_training_start(pedidos) if inicio is None else pd.Timestamp(inicio)

    previsoes, modelos, historico, meta_alvos = {}, {}, {}, {}
    for alvo in alvos:
        serie = build_daily_series(pedidos, alvo=alvo, inicio=inicio)
        ajuste = fit_best(serie, horizonte=horizonte, n_cortes=n_cortes)
        previsao = ajuste.forecast(horizonte)

        previsoes[alvo] = previsao
        modelos[alvo] = ajuste.tabela
        historico[alvo] = pd.DataFrame({"data": serie.index, "real": serie.to_numpy()})
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
        "metadados": {
            "versao": VERSAO_PIPELINE,
            "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "horizonte": int(horizonte),
            "n_cortes": int(n_cortes),
            "apenas_b2c": True,
            "metrica_principal": "wape",
            "alvos": meta_alvos,
        },
    }

    if salvar:
        resultado["metadados"]["salvo_em"] = registry.salvar_resultado(resultado, source_key, uid, id_token, pasta)
    return resultado
