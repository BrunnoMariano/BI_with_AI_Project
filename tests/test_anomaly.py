"""Testes da Fase 6: alertas de anomalia (ml.anomaly), faixas do backtest e telas da aba Previsões."""

import numpy as np
import pandas as pd
import pytest

import forecast_tab as ft
from ml.anomaly import detect_anomalies
from ml.conversion import train_conversion
from ml.forecasting import backtest, faixas_backtest, intervalos
from ml.pipeline import run_predictive_pipeline
from ml.registry import de_json, para_json
from tests.test_conversion import _pedidos
from tests.test_forecasting import _planilha_sintetica, _serie_semanal


def _cenario():
    datas = pd.date_range("2026-03-01", periods=5, freq="D")
    backtest_df = pd.DataFrame(
        {
            "data": datas,
            "previsto": [100.0, 100.0, 100.0, 0.0, 100.0],
            "lim_inf_95": [80.0, 80.0, 80.0, 0.0, 80.0],
            "lim_sup_95": [120.0, 120.0, 120.0, 10.0, 120.0],
        }
    )
    real = pd.Series([100.0, 150.0, 60.0, 30.0, 120.0], index=datas)
    return real, backtest_df


def test_detect_anomalies_direcao_e_desvio():
    resultado = detect_anomalies(*_cenario())
    assert resultado["anomalia"].tolist() == [False, True, True, True, False]  # 120 = limite: normal
    assert resultado["direcao"].tolist() == [None, "acima", "abaixo", "acima", None]
    assert resultado["desvio_pct"].iloc[1] == pytest.approx(50.0)
    assert resultado["desvio_pct"].iloc[2] == pytest.approx(-40.0)
    assert np.isnan(resultado["desvio_pct"].iloc[3])  # previsto 0: desvio indefinido


def test_detect_anomalies_so_dias_nos_dois_lados_e_ordenado():
    real, backtest_df = _cenario()
    resultado = detect_anomalies(real.iloc[1:], backtest_df.iloc[::-1])
    assert len(resultado) == 4
    assert resultado["data"].is_monotonic_increasing


def test_detect_anomalies_entrada_invalida():
    real, backtest_df = _cenario()
    with pytest.raises(ValueError, match="lim_sup_95"):
        detect_anomalies(real, backtest_df.drop(columns="lim_sup_95"))
    with pytest.raises(ValueError, match="repetidas"):
        detect_anomalies(real, pd.concat([backtest_df, backtest_df]))


def test_faixas_backtest_usa_residuos_dos_outros_cortes():
    resultado = backtest(_serie_semanal(200, ruido=2.0), 28, 4, modelos=["baseline_sazonal_7d"])
    faixas = faixas_backtest(resultado, "baseline_sazonal_7d")
    assert len(faixas) == 4 * 28 and faixas["data"].is_unique
    primeiro = faixas[faixas["corte"] == 1]
    esperado = intervalos(primeiro["previsto"].to_numpy(), resultado["residuos"]["baseline_sazonal_7d"][1:])
    np.testing.assert_allclose(primeiro["lim_sup_95"], esperado["lim_sup_95"])
    np.testing.assert_allclose(primeiro["lim_inf_80"], esperado["lim_inf_80"])


@pytest.fixture(scope="module")
def resultado():
    return run_predictive_pipeline(_planilha_sintetica(), "teste", horizonte_previsao=28)


def test_pipeline_devolve_backtest_do_vencedor_e_ida_e_volta_json(resultado):
    faixas = resultado["backtest"]["receita"]
    assert len(faixas) == 4 * 28
    assert {"data", "real", "previsto", "lim_inf_95", "lim_sup_95"} <= set(faixas.columns)
    recarregado = de_json(para_json(resultado))
    assert pd.api.types.is_datetime64_any_dtype(recarregado["backtest"]["receita"]["data"])


def test_real_do_backtest_e_o_mesmo_da_serie(resultado):
    anomalias = detect_anomalies(ft.serie_historica(resultado, "receita"), resultado["backtest"]["receita"])
    np.testing.assert_allclose(anomalias["real"], resultado["backtest"]["receita"]["real"])


# ---------- montagem da tela ----------

def test_grafico_marca_os_dias_anomalos():
    anomalias = detect_anomalies(*_cenario())
    fig = ft.construir_grafico_anomalias(anomalias, "receita")
    nomes = [trace.name for trace in fig.data if trace.showlegend is not False]
    assert nomes == ["Faixa de 95%", "Previsto no backtest", "Real", "Acima da faixa", "Abaixo da faixa"]
    acima = next(t for t in fig.data if t.name == "Acima da faixa")
    abaixo = next(t for t in fig.data if t.name == "Abaixo da faixa")
    assert list(acima.y) == [150.0, 30.0] and acima.marker.symbol == "triangle-up"
    assert list(abaixo.y) == [60.0] and abaixo.marker.symbol == "triangle-down"


def test_grafico_sem_anomalias_nao_tem_marcadores():
    real, backtest_df = _cenario()
    anomalias = detect_anomalies(real.clip(80, 10), backtest_df.assign(lim_inf_95=0.0, lim_sup_95=1000.0))
    nomes = {trace.name for trace in ft.construir_grafico_anomalias(anomalias, "pedidos").data}
    assert "Acima da faixa" not in nomes and "Abaixo da faixa" not in nomes


def test_tabela_alertas():
    tabela = ft.tabela_alertas(detect_anomalies(*_cenario()), "receita")
    assert len(tabela) == 3
    assert tabela["Dia"].iloc[0].startswith("04/03/2026")  # mais recente primeiro
    linha = tabela[tabela["Dia"].str.startswith("02/03/2026")].iloc[0]
    assert linha["Direção"] == "▲ Acima" and linha["Desvio"] == "+50%" and linha["Faixa de 95%"] == "R$ 80 a R$ 120"
    assert tabela["Desvio"].iloc[0] == "–"


def test_tabelas_da_conversao():
    treino = train_conversion(_pedidos())
    avaliacao = ft.tabela_avaliacao(treino)
    assert avaliacao["Modelo"].iloc[0].endswith("★ vencedor")
    assert avaliacao["Modelo"].str.contains("vencedor").sum() == 1
    importancia = ft.tabela_importancia(treino)
    assert "Cupom" in set(importancia["Variável"]) and len(importancia) == 8

    fila = pd.DataFrame({"pedido_id": [7], "valor": [10.0], "probabilidade": [0.5], "receita_esperada": [5.0], "outra": [1]})
    assert list(ft.tabela_pendentes(fila).columns) == ["Pedido", "Valor", "Probabilidade de pagar", "Receita esperada"]


def test_tabela_pendentes_link_em_texto_e_sem_comprador():
    fila = pd.DataFrame(
        {"pedido_id": [1, 2], "gerou_link": [True, False], "tentativa_abandonada": [True, False], "cliente": ["a", "b"]}
    )
    tabela = ft.tabela_pendentes(fila)
    assert tabela["Link de pagamento"].tolist() == ["Sim", "Não"]
    assert list(tabela.columns) == ["Pedido", "Link de pagamento"]  # comprador nunca aparece
    assert fila["gerou_link"].tolist() == [True, False]  # não altera a fila recebida
