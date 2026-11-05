"""Testes da Fase 3: funções de montagem da aba Previsões (forecast_tab).

Testam só as partes puras (formatação, KPIs, tabela, CSV, gráficos); a tela em si
roda no Streamlit.
"""

import io

import pandas as pd
import pytest

import forecast_tab as ft
from ml.pipeline import run_predictive_pipeline
from tests.test_forecasting import _planilha_sintetica


@pytest.fixture(scope="module")
def resultado():
    return run_predictive_pipeline(_planilha_sintetica(), "teste", horizonte_previsao=90)


@pytest.mark.parametrize(
    "valor, casas, esperado",
    [(1234.5, 0, "1.234"), (1234.5, 2, "1.234,50"), (88410.08, 2, "88.410,08"), (float("nan"), 0, "–")],
)
def test_formatar_numero(valor, casas, esperado):
    # 1234.5 arredonda para 1234 (arredondamento bancário do Python), igual em qualquer formatação.
    assert ft.formatar_numero(valor, casas) == esperado


def test_formatar_brl_pct_e_valor():
    assert ft.formatar_brl(88410.08, 2) == "R$ 88.410,08"
    assert ft.formatar_pct(0.511) == "51,1%"
    assert ft.formatar_pct(-0.1567, sinal=True) == "−15,7%"
    assert ft.formatar_pct(0.05, sinal=True, casas=0) == "+5%"
    assert ft.formatar_valor(10, "receita") == "R$ 10"
    assert ft.formatar_valor(10, "pedidos") == "10"


def test_calcular_kpis(resultado):
    kpis = ft.calcular_kpis(resultado)
    for alvo in ("receita", "pedidos"):
        kpi = kpis[alvo]
        previsto_28 = resultado["previsoes"][alvo]["previsto"].iloc[:28].sum()
        anterior = resultado["historico"][alvo]["real"].iloc[-28:].sum()
        assert kpi["previsto"] == pytest.approx(previsto_28)
        assert kpi["anterior"] == pytest.approx(anterior)
        assert kpi["variacao"] == pytest.approx((previsto_28 - anterior) / anterior)
        assert 0 <= kpi["lim_inf_95"] <= kpi["lim_inf_80"] <= kpi["lim_sup_80"] <= kpi["lim_sup_95"]


def test_tabela_modelos_destaca_vencedor(resultado):
    tabela = ft.tabela_modelos(resultado, "pedidos")
    assert list(tabela.columns) == ["Modelo", "WAPE", "MAE", "RMSE", "Erro no total", "Cobertura 80%"]
    assert tabela["Modelo"].iloc[0].endswith("★ vencedor")
    assert tabela["Modelo"].str.contains("vencedor").sum() == 1
    assert tabela["WAPE"].is_monotonic_increasing


@pytest.mark.parametrize("granularidade", ["dia", "semana", "mes"])
def test_grafico_previsao_e_csv(resultado, granularidade):
    historico = ft.serie_historica(resultado, "receita")
    previsao = resultado["previsoes"]["receita"].iloc[:90]
    fig, agregada = ft.construir_grafico_previsao(historico, previsao, resultado["residuos"]["receita"], "receita", granularidade)

    nomes = [trace.name for trace in fig.data if trace.showlegend is not False]
    assert nomes == ["Faixa de 95%", "Faixa de 80%", "Real", "Previsto"]
    previsto = next(t for t in fig.data if t.name == "Previsto")
    assert previsto.line.dash == "dash"
    assert len(previsto.x) == len(previsto.customdata)
    assert any("hoje" in anotacao.text for anotacao in fig.layout.annotations)

    csv = pd.read_csv(io.BytesIO(ft.tabela_csv(agregada, "receita", granularidade)), sep=";", decimal=",", encoding="utf-8-sig")
    assert len(csv) == len(agregada)
    assert {"periodo", "previsto", "lim_inf_80", "lim_sup_95"} <= set(csv.columns)
    assert csv["previsto"].sum() == pytest.approx(agregada["previsto"].sum(), rel=1e-4)


def test_grafico_simulacao(resultado):
    from ml.forecasting import simulate

    historico = ft.serie_historica(resultado, "pedidos")
    corte = historico.index[-29]
    simulacao = simulate(historico, corte, "holt_winters")
    fig = ft.construir_grafico_simulacao(historico, simulacao, "pedidos")
    nomes = [trace.name for trace in fig.data if trace.showlegend is not False]
    assert "Real" in nomes and "Previsto no corte" in nomes
    real = next(t for t in fig.data if t.name == "Real")
    assert pd.Timestamp(real.x[-1]) == historico.index[-1]  # o real continua depois do corte
