"""Testes da Fase 2: séries (ml.features), métricas, modelos/backtest (ml.forecasting),
registro (ml.registry) e pipeline (ml.pipeline).

Os testes sintéticos rodam sempre e offline; os que usam a planilha oficial
são pulados se data/vendas_cliente.xlsx não existir.
"""

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from ml import forecasting, metrics, registry
from ml.features import build_daily_series, calendar_features, detect_training_start
from ml.forecasting import (
    COLUNAS_PREVISAO,
    MODELOS,
    backtest,
    baseline_sazonal_7d,
    cortes_backtest,
    fit_best,
    forecast,
    intervalos,
    media_movel_28d,
    prever,
)
from ml.pipeline import run_predictive_pipeline


# ---------- dados sintéticos ----------

def _serie_semanal(dias: int = 200, ruido: float = 0.0, seed: int = 0) -> pd.Series:
    """Série diária com padrão semanal fixo (+ ruído opcional), sem valores negativos."""
    rng = np.random.default_rng(seed)
    padrao = np.array([10, 12, 14, 13, 11, 4, 2], dtype=float)
    valores = np.resize(padrao, dias) + rng.normal(0, ruido, dias)
    indice = pd.date_range("2025-08-01", periods=dias, freq="D", name="data")
    return pd.Series(np.maximum(valores, 0), index=indice, name="pedidos")


def _pedidos_limpos() -> pd.DataFrame:
    """DataFrame no formato de clean_orders: poucos pedidos para conferir a agregação."""
    return pd.DataFrame(
        {
            "data_pedido": pd.to_datetime(
                [
                    "2025-08-01 10:00", "2025-08-01 15:00", "2025-08-03 09:00",  # 02/08 sem venda
                    "2025-08-03 11:00", "2025-08-04 08:00", "2025-08-05 12:00",  # 05/08 = último dia
                ]
            ),
            "pago": [True, True, True, False, True, True],
            "is_b2b": [False, False, True, False, False, False],
            "valor": [100.0, 50.0, 999.0, 70.0, 30.0, 10.0],
        }
    )


def _planilha_sintetica(dias: int = 210, seed: int = 1) -> pd.DataFrame:
    """Extração fictícia com os nomes de coluna oficiais e ~8 pedidos/dia."""
    rng = np.random.default_rng(seed)
    linhas = []
    padrao = [9, 10, 11, 10, 9, 4, 3]
    for i, dia in enumerate(pd.date_range("2025-08-01", periods=dias, freq="D")):
        for j in range(rng.poisson(padrao[i % 7])):
            linhas.append(
                {
                    "Número": len(linhas) + 1,
                    "Conta": "Empresa X" if j == 0 and i % 10 == 0 else None,
                    "Produto": "Certificação A",
                    "Valor Pago": "R$99,90",
                    "Status Venda": "pago" if rng.random() < 0.7 else "pendente",
                    "Forma de pagamento": "pix",
                    "Data Venda": dia.strftime("%d/%m/%Y 12:00:00"),
                    "Criado em": dia.strftime("%d/%m/%Y 10:00:00"),
                }
            )
    return pd.DataFrame(linhas)


# ---------- metrics.py ----------

def test_metricas_valores_conhecidos():
    real, prev = [10, 0, 20], [8, 2, 25]
    assert metrics.mae(real, prev) == pytest.approx(3.0)
    assert metrics.rmse(real, prev) == pytest.approx(np.sqrt((4 + 4 + 25) / 3))
    assert metrics.wape(real, prev) == pytest.approx(9 / 30)
    assert metrics.mape_seguro(real, prev) == pytest.approx((0.2 + 0.25) / 2)  # ignora o dia com real 0
    assert metrics.erro_total(real, prev) == pytest.approx((35 - 30) / 30)
    assert metrics.cobertura(real, [9, 0, 21], [11, 1, 30]) == pytest.approx(2 / 3)


def test_metricas_casos_limite():
    assert np.isnan(metrics.wape([0, 0], [1, 1]))
    assert np.isnan(metrics.mape_seguro([0, 0], [1, 1]))
    assert metrics.wape([5, 5], [5, 5]) == 0.0
    with pytest.raises(ValueError):
        metrics.mae([1, 2], [1])


# ---------- features.py ----------

def test_build_daily_series_pedidos_preenche_zeros_e_exclui_b2b_e_ultimo_dia():
    serie = build_daily_series(_pedidos_limpos(), "pedidos", inicio="2025-08-01")
    assert list(serie.index.date.astype(str)) == ["2025-08-01", "2025-08-02", "2025-08-03", "2025-08-04"]
    assert serie.tolist() == [2.0, 0.0, 0.0, 1.0]  # 03/08: só B2B pago e um não pago
    assert serie.index.freqstr == "D"
    assert serie.name == "pedidos"


def test_build_daily_series_receita_e_opcoes():
    pedidos = _pedidos_limpos()
    receita = build_daily_series(pedidos, "receita", inicio="2025-08-01")
    assert receita.tolist() == [150.0, 0.0, 0.0, 30.0]
    com_b2b = build_daily_series(pedidos, "receita", inicio="2025-08-01", excluir_b2b=False, descartar_ultimo_dia=False)
    assert com_b2b.tolist() == [150.0, 0.0, 999.0, 30.0, 10.0]
    with pytest.raises(ValueError):
        build_daily_series(pedidos, "lucro")


def test_detect_training_start_primeiro_mes_com_100_pagos():
    datas = list(pd.date_range("2025-06-01", periods=50, freq="h")) + list(pd.date_range("2025-08-02", periods=120, freq="5h"))
    df = pd.DataFrame({"data_pedido": datas, "pago": True, "is_b2b": False, "valor": 1.0})
    assert detect_training_start(df) == pd.Timestamp("2025-08-01")
    # Nenhum mês com 100: usa o primeiro pago.
    assert detect_training_start(df.iloc[:50]) == pd.Timestamp("2025-06-01")


def test_build_daily_series_inicio_automatico():
    datas = list(pd.date_range("2025-06-01", periods=10, freq="D")) + list(pd.date_range("2025-08-01", periods=150, freq="4h"))
    df = pd.DataFrame({"data_pedido": datas, "pago": True, "is_b2b": False, "valor": 1.0})
    assert build_daily_series(df).index[0] == pd.Timestamp("2025-08-01")


def test_calendar_features():
    cal = calendar_features(pd.to_datetime(["2025-12-25", "2025-12-27", "2025-12-29"]))
    assert cal["feriado"].tolist() == [1, 0, 0]  # Natal
    assert cal["fim_de_semana"].tolist() == [0, 1, 0]
    assert cal["dia_semana"].tolist() == [3, 5, 0]
    assert cal["dia_mes"].tolist() == [25, 27, 29]


# ---------- forecasting.py: modelos ----------

def test_baseline_repete_a_ultima_semana():
    serie = _serie_semanal(30)
    previsto = baseline_sazonal_7d(serie, 10)
    # O dia seguinte ao fim tem o mesmo dia da semana de 7 dias antes.
    assert previsto[0] == serie.iloc[-7]
    assert previsto[7] == previsto[0]


def test_media_movel_28d_constante():
    serie = _serie_semanal(60)
    assert np.allclose(media_movel_28d(serie, 5), serie.iloc[-28:].mean())


@pytest.mark.parametrize("modelo", list(MODELOS))
def test_modelos_devolvem_h_valores_nao_negativos(modelo):
    serie = _serie_semanal(160, ruido=1.0)
    previsto = prever(modelo, serie, 28)
    assert previsto.shape == (28,)
    assert np.isfinite(previsto).all() and (previsto >= 0).all()


def test_lightgbm_horizonte_maior_que_28():
    previsto = prever("lightgbm", _serie_semanal(160, ruido=1.0), 40)
    assert previsto.shape == (40,) and (previsto >= 0).all()


def test_prever_corta_negativos(monkeypatch):
    monkeypatch.setitem(MODELOS, "negativo", lambda treino, h: np.full(h, -5.0))
    assert (prever("negativo", _serie_semanal(30), 3) == 0).all()
    with pytest.raises(ValueError):
        prever("nao_existe", _serie_semanal(30), 3)


# ---------- forecasting.py: backtest e escolha ----------

def test_cortes_backtest_origem_deslizante():
    serie = _serie_semanal(200)
    cortes = cortes_backtest(serie, 28, 4)
    assert [c["corte"] for c in cortes] == [1, 2, 3, 4]
    assert cortes[-1]["fim_teste"] == serie.index[-1]
    assert cortes[0]["inicio_teste"] == serie.index[200 - 4 * 28]
    for anterior, seguinte in zip(cortes, cortes[1:]):
        assert seguinte["inicio_teste"] == anterior["fim_teste"] + pd.Timedelta(days=1)


def test_backtest_serie_curta_da_erro():
    with pytest.raises(ValueError, match="curta"):
        backtest(_serie_semanal(100), 28, 4)


def test_backtest_nao_vaza_dados_do_teste(monkeypatch):
    vistos = []

    def espiao(treino, h):
        vistos.append(treino.index[-1])
        return np.zeros(h)

    monkeypatch.setitem(MODELOS, "espiao", espiao)
    serie = _serie_semanal(200)
    resultado = backtest(serie, 28, 4, modelos=["espiao"])
    for fim_treino, corte in zip(vistos, resultado["cortes"]):
        assert fim_treino == corte["inicio_teste"] - pd.Timedelta(days=1)


def test_backtest_tabela_e_residuos():
    serie = _serie_semanal(200, ruido=1.0)
    resultado = backtest(serie, 28, 4)
    tabela = resultado["tabela"]
    assert set(tabela["modelo"]) == set(MODELOS)
    assert tabela["wape"].is_monotonic_increasing
    assert (tabela["n_cortes"] == 4).all()
    assert {"wape", "mae", "rmse", "mape_seguro", "erro_total", "cobertura_80", "cobertura_95"} <= set(tabela.columns)
    assert len(resultado["por_corte"]) == 4 * len(MODELOS)
    assert resultado["residuos"]["holt_winters"].shape == (4, 28)
    # A tabela é a média dos cortes.
    hw = resultado["por_corte"].query("modelo == 'holt_winters'")
    assert tabela.set_index("modelo").loc["holt_winters", "wape"] == pytest.approx(hw["wape"].mean())


def test_backtest_modelo_com_erro_fica_de_fora(monkeypatch):
    def quebrado(treino, h):
        raise RuntimeError("falhou")

    monkeypatch.setitem(MODELOS, "quebrado", quebrado)
    with pytest.warns(UserWarning):
        resultado = backtest(_serie_semanal(200), 28, 4, modelos=["quebrado", "baseline_sazonal_7d"])
    assert list(resultado["tabela"]["modelo"]) == ["baseline_sazonal_7d"]
    assert "quebrado" in resultado["falhas"]


def test_fit_best_escolhe_menor_wape():
    serie = _serie_semanal(200)  # padrão semanal perfeito: o baseline acerta tudo
    ajuste = fit_best(serie, modelos=["media_movel_28d", "baseline_sazonal_7d"])
    assert ajuste.modelo == "baseline_sazonal_7d"
    assert ajuste.tabela.iloc[0]["wape"] == pytest.approx(0.0)


def test_intervalos_ordenados_e_sem_negativos():
    lim = intervalos(np.array([1.0, 10.0]), np.linspace(-6, 6, 101))
    assert (lim["lim_inf_95"] <= lim["lim_inf_80"]).all()
    assert (lim["lim_sup_80"] <= lim["lim_sup_95"]).all()
    assert lim["lim_inf_95"][0] == 0.0  # 1 - 5,7 cortado em zero
    assert lim["lim_sup_80"][1] == pytest.approx(10 + np.quantile(np.linspace(-6, 6, 101), 0.9))


def test_forecast_formato_datas_e_sem_negativos():
    serie = _serie_semanal(200, ruido=2.0)
    ajuste = fit_best(serie)
    previsao = ajuste.forecast()
    assert list(previsao.columns) == COLUNAS_PREVISAO
    assert len(previsao) == 28
    assert previsao["data"].iloc[0] == serie.index[-1] + pd.Timedelta(days=1)
    assert previsao["data"].diff().dropna().eq(pd.Timedelta(days=1)).all()
    numericas = previsao.drop(columns="data")
    assert (numericas >= 0).all().all()
    assert (previsao["lim_inf_95"] <= previsao["lim_inf_80"]).all()
    assert (previsao["lim_sup_80"] <= previsao["lim_sup_95"]).all()
    assert len(ajuste.forecast(14)) == 14


def test_forecast_sem_residuos_roda_backtest_proprio():
    previsao = forecast(_serie_semanal(200, ruido=1.0), "baseline_sazonal_7d", h=7)
    assert len(previsao) == 7
    assert previsao[["lim_inf_80", "lim_sup_80"]].notna().all().all()


# ---------- registry.py ----------

def test_para_json_tipos_simples():
    payload = registry.para_json(
        {
            "df": pd.DataFrame({"data": pd.to_datetime(["2026-04-07"]), "v": [np.float64(np.nan)]}),
            "n": np.int64(3),
            "x": np.float32(1.5),
            "ok": np.bool_(True),
        }
    )
    assert payload == {"df": [{"data": "2026-04-07T00:00:00", "v": None}], "n": 3, "x": 1.5, "ok": True}
    json.dumps(payload, allow_nan=False)  # aceito pelo Firestore


def test_chave_segura():
    assert registry.chave_segura("users/abc/vendas cliente.xlsx") == "users_abc_vendas_cliente.xlsx"
    with pytest.raises(ValueError):
        registry.chave_segura("///")


def _resultado_exemplo() -> dict:
    previsao = forecast(_serie_semanal(200, ruido=1.0), "baseline_sazonal_7d", h=7)
    return {
        "source_key": "arquivo 1",
        "previsoes": {"pedidos": previsao},
        "modelos": {"pedidos": pd.DataFrame({"modelo": ["baseline_sazonal_7d"], "wape": [0.1]})},
        "historico": {"pedidos": pd.DataFrame({"data": pd.date_range("2026-01-01", periods=3), "real": [1.0, 2.0, 3.0]})},
        "metadados": {"horizonte": 7},
    }


def test_salvar_e_carregar_local(tmp_path):
    resultado = _resultado_exemplo()
    salvo = registry.salvar_resultado(resultado, "arquivo 1", pasta=tmp_path)
    assert salvo["firestore"] is None
    assert salvo["local"].endswith("arquivo_1.joblib")
    carregado = registry.carregar_resultado("arquivo 1", pasta=tmp_path)
    pd.testing.assert_frame_equal(carregado["previsoes"]["pedidos"], resultado["previsoes"]["pedidos"])
    assert registry.carregar_resultado("outro", pasta=tmp_path) is None


def test_salvar_e_carregar_firestore(tmp_path, monkeypatch):
    banco = {}
    falso = types.ModuleType("firebase_service")
    falso.set_document = lambda caminho, dados, token: banco.__setitem__(caminho, json.loads(json.dumps(dados, allow_nan=False)))
    falso.get_document = lambda caminho, token: banco.get(caminho)
    monkeypatch.setitem(sys.modules, "firebase_service", falso)

    resultado = _resultado_exemplo()
    salvo = registry.salvar_resultado(resultado, "arquivo 1", uid="u1", id_token="tok", pasta=tmp_path)
    assert salvo["firestore"] == "users/u1/models/arquivo_1"

    # Sem o joblib local, carrega do Firestore e reconstrói os DataFrames.
    (tmp_path / "arquivo_1.joblib").unlink()
    carregado = registry.carregar_resultado("arquivo 1", uid="u1", id_token="tok", pasta=tmp_path)
    previsao = carregado["previsoes"]["pedidos"]
    assert pd.api.types.is_datetime64_any_dtype(previsao["data"])
    np.testing.assert_allclose(previsao["previsto"], resultado["previsoes"]["pedidos"]["previsto"])
    assert carregado["metadados"] == {"horizonte": 7}


# ---------- pipeline.py ----------

def test_pipeline_sintetico(tmp_path):
    resultado = run_predictive_pipeline(_planilha_sintetica(), "teste", salvar=True, pasta=tmp_path)
    assert set(resultado["previsoes"]) == {"pedidos", "receita"}
    meta = resultado["metadados"]["alvos"]["pedidos"]
    assert meta["inicio_treino"] == "2025-08-01"
    assert meta["n_obs"] == 209  # 210 dias menos o último (incompleto)
    assert meta["inicio_previsao"] == "2026-02-26"
    assert meta["modelo_vencedor"] == resultado["modelos"]["pedidos"].iloc[0]["modelo"]
    assert len(meta["cortes_backtest"]) == 4
    assert list(resultado["previsoes"]["receita"].columns) == COLUNAS_PREVISAO
    assert resultado["metadados"]["salvo_em"]["local"].endswith("teste.joblib")
    json.dumps(registry.para_json(resultado), allow_nan=False)


def test_pipeline_base_sem_colunas_da_erro():
    with pytest.raises(ValueError, match="colunas necessárias"):
        run_predictive_pipeline(pd.DataFrame({"a": [1]}), "x")


# ---------- Fase 3: agregação, totais, simulação e B2B ----------

from ml.features import agregar_serie, periodos, summarize_b2b  # noqa: E402
from ml.forecasting import (  # noqa: E402
    agregar_historico,
    agregar_previsao,
    matriz_residuos,
    residuos_longos,
    simular_trajetorias,
    simulate,
    total_com_intervalo,
)
from ml.pipeline import verificar_requisitos  # noqa: E402


def test_periodos_semana_alinhada_a_origem_e_mes():
    datas = pd.date_range("2026-03-30", "2026-04-14")
    semanas = periodos(datas, "semana", origem="2026-04-07")
    assert semanas[datas.get_loc("2026-04-06")] == pd.Timestamp("2026-03-31")  # semana antes da origem
    assert semanas[datas.get_loc("2026-04-07")] == pd.Timestamp("2026-04-07")
    assert semanas[datas.get_loc("2026-04-13")] == pd.Timestamp("2026-04-07")
    assert periodos(["2026-04-15"], "mes")[0] == pd.Timestamp("2026-04-01")
    with pytest.raises(ValueError):
        periodos(datas, "ano")


def test_agregar_serie_marca_periodo_incompleto():
    serie = pd.Series(1.0, index=pd.date_range("2026-03-01", "2026-04-06"))
    mensal = agregar_serie(serie, "mes")
    assert mensal["real"].tolist() == [31.0, 6.0]
    assert mensal["parcial"].tolist() == [False, True]


def test_residuos_longos_ida_e_volta():
    matriz = np.arange(8.0).reshape(2, 4)
    longo = residuos_longos(matriz)
    assert list(longo.columns) == ["corte", "passo", "residuo"]
    np.testing.assert_array_equal(matriz_residuos(longo), matriz)
    com_falha = np.vstack([matriz, np.full(4, np.nan)])
    assert matriz_residuos(com_falha).shape == (2, 4)


def test_simular_trajetorias_formato_e_sem_negativos():
    residuos = np.tile(np.array([-30.0, 5.0, 10.0, -2.0, 3.0, 1.0, 0.0]), (2, 4))
    trajetorias = simular_trajetorias(np.full(40, 10.0), residuos, n_sim=500)
    assert trajetorias.shape == (500, 40)
    assert (trajetorias >= 0).all()
    with pytest.raises(ValueError):
        simular_trajetorias(np.ones(5), np.full((2, 28), np.nan))


def test_total_com_intervalo_mais_estreito_que_soma_das_faixas_diarias():
    serie = _serie_semanal(200, ruido=2.0)
    ajuste = fit_best(serie)
    previsao = ajuste.forecast(28)
    total = total_com_intervalo(previsao, ajuste.residuos)
    assert total["previsto"] == pytest.approx(previsao["previsto"].sum())
    assert total["lim_inf_95"] <= total["lim_inf_80"] <= total["previsto"] <= total["lim_sup_80"] <= total["lim_sup_95"]
    soma_faixas = previsao["lim_sup_80"].sum() - previsao["lim_inf_80"].sum()
    assert total["lim_sup_80"] - total["lim_inf_80"] < soma_faixas
    vazio = total_com_intervalo(previsao, np.full((1, 28), np.nan))
    assert np.isnan(vazio["lim_inf_80"])


def test_agregar_previsao_semana_e_mes_com_realizado():
    serie = _serie_semanal(200, ruido=1.0)
    ajuste = fit_best(serie, modelos=["baseline_sazonal_7d", "holt_winters"])
    previsao = ajuste.forecast(40)

    diaria = agregar_previsao(previsao, ajuste.residuos, "dia")
    assert len(diaria) == 40 and diaria["n_dias"].eq(1).all()

    semanal = agregar_previsao(previsao, ajuste.residuos, "semana")
    assert semanal["periodo"].iloc[0] == previsao["data"].iloc[0]
    assert semanal["n_dias"].tolist() == [7, 7, 7, 7, 7, 5]
    assert semanal["parcial"].tolist() == [False] * 5 + [True]
    assert semanal["previsto"].sum() == pytest.approx(previsao["previsto"].sum())
    assert (semanal["lim_inf_80"] <= semanal["lim_sup_80"]).all()

    mensal = agregar_previsao(previsao, ajuste.residuos, "mes", historico=serie)
    primeiro_mes = periodos([previsao["data"].iloc[0]], "mes")[0]
    ja_realizado = serie[periodos(serie.index, "mes") == primeiro_mes].sum()
    assert mensal.loc[0, "realizado"] == pytest.approx(ja_realizado)
    assert mensal["previsto"].sum() == pytest.approx(previsao["previsto"].sum() + ja_realizado)

    historico_mensal = agregar_historico(serie, "mes", previsao["data"].iloc[0])
    assert primeiro_mes not in set(historico_mensal["periodo"])  # mês da origem só aparece na previsão

    # Semana inicial incompleta (série começa no meio de um bloco de 7 dias) sai do histórico.
    historico_semanal = agregar_historico(serie, "semana", previsao["data"].iloc[0])
    assert not historico_semanal["parcial"].iloc[0]
    assert historico_semanal["periodo"].iloc[-1] + pd.Timedelta(days=7) == previsao["data"].iloc[0]


def test_simulate_treina_so_ate_o_corte(monkeypatch):
    vistos = []
    original = MODELOS["baseline_sazonal_7d"]

    def espiao(treino, h):
        vistos.append(treino.index[-1])
        return original(treino, h)

    monkeypatch.setitem(MODELOS, "baseline_sazonal_7d", espiao)
    serie = _serie_semanal(220, ruido=1.0)
    corte = serie.index[-29]
    resultado = simulate(serie, corte, "baseline_sazonal_7d")
    assert max(vistos) <= corte  # nem a previsão nem o backtest das faixas viram o futuro
    assert resultado["previsao"]["data"].iloc[0] == corte + pd.Timedelta(days=1)
    assert len(resultado["previsao"]) == 28
    np.testing.assert_allclose(resultado["previsao"]["real"], serie.iloc[-28:])
    assert resultado["wape"] == pytest.approx(metrics.wape(serie.iloc[-28:], resultado["previsao"]["previsto"]))
    assert resultado["n_cortes_intervalo"] == 4
    assert resultado["previsao"]["lim_inf_80"].notna().all()


def test_simulate_treino_curto_usa_reserva_e_avisa():
    serie = _serie_semanal(100, ruido=1.0)
    resultado = simulate(serie, serie.index[61], "lightgbm")  # 62 dias de treino: LightGBM não roda
    assert resultado["modelo"] == "holt_winters"
    assert resultado["n_cortes_intervalo"] == 0
    assert resultado["previsao"]["lim_inf_80"].isna().all()
    assert any("LightGBM" in aviso or "lightgbm" in aviso for aviso in resultado["avisos"])
    with pytest.raises(ValueError, match="28 dias"):
        simulate(serie, serie.index[-10], "holt_winters")


def test_summarize_b2b():
    pedidos = _pedidos_limpos()
    resumo = summarize_b2b(pedidos, "2025-08-01", "2025-08-04")
    assert resumo["historico"] == {"pedidos": 1, "receita": 999.0, "inicio": "2025-08-03", "fim": "2025-08-03"}
    assert resumo["janela"]["pedidos"] == 1
    assert resumo["janela"]["participacao_receita"] == pytest.approx(999 / (100 + 50 + 999 + 30))
    fora = summarize_b2b(pedidos, "2025-08-04", "2025-08-05")
    assert fora["janela"]["pedidos"] == 0 and fora["janela"]["ticket_medio"] == 0.0


def test_verificar_requisitos():
    assert verificar_requisitos(_planilha_sintetica(30)) == []
    faltando = verificar_requisitos(pd.DataFrame({"Produto": ["A"], "Status Venda": ["pago"]}))
    assert len(faltando) == 2
    assert any("Criado em" in item for item in faltando)
    assert any("Valor Pago" in item for item in faltando)


def test_pipeline_horizonte_previsao_maior_que_backtest():
    resultado = run_predictive_pipeline(_planilha_sintetica(), "teste", alvos=("pedidos",), horizonte_previsao=90)
    assert len(resultado["previsoes"]["pedidos"]) == 90
    assert matriz_residuos(resultado["residuos"]["pedidos"]).shape == (4, 28)
    assert resultado["metadados"]["horizonte"] == 28
    assert "b2b" in resultado


# ---------- planilha oficial (referências do CLAUDE.md) ----------

@pytest.fixture(scope="module")
def resultado_planilha(vendas_df):
    return run_predictive_pipeline(vendas_df, "vendas_cliente")


def test_planilha_janela_de_treino_e_previsao(resultado_planilha):
    for alvo in ("pedidos", "receita"):
        meta = resultado_planilha["metadados"]["alvos"][alvo]
        assert meta["inicio_treino"] == "2025-08-01"
        assert meta["fim_treino"] == "2026-04-06"
        assert meta["inicio_previsao"] == "2026-04-07"
        assert meta["n_obs"] == 249
        assert meta["cortes_backtest"][0]["inicio_teste"] == "2025-12-16"


def test_planilha_wape_proximo_do_prototipo(resultado_planilha):
    tabela = resultado_planilha["modelos"]["receita"].set_index("modelo")["wape"]
    assert tabela["holt_winters"] == pytest.approx(0.51, abs=0.03)
    assert tabela["baseline_sazonal_7d"] == pytest.approx(0.62, abs=0.03)
    assert tabela["media_movel_28d"] == pytest.approx(0.73, abs=0.03)
    assert resultado_planilha["metadados"]["alvos"]["receita"]["wape_vencedor"] <= tabela["holt_winters"]


def test_planilha_resumo_b2b_numeros_de_conferencia(resultado_planilha):
    b2b = resultado_planilha["b2b"]
    assert b2b["historico"]["pedidos"] == 82
    assert b2b["historico"]["receita"] == pytest.approx(88410.08, abs=0.005)
    assert b2b["janela"]["pedidos"] == 23
    assert b2b["janela"]["receita"] == pytest.approx(10891.08, abs=0.005)
    assert b2b["janela"]["inicio"] == "2025-08-01" and b2b["janela"]["fim"] == "2026-04-06"


def test_planilha_simulacao_no_ultimo_corte_permitido(resultado_planilha):
    historico = resultado_planilha["historico"]["receita"]
    serie = pd.Series(historico["real"].to_numpy(), index=pd.DatetimeIndex(historico["data"]))
    resultado = simulate(serie, "2026-03-09", "holt_winters")
    assert resultado["previsao"]["data"].iloc[-1] == pd.Timestamp("2026-04-06")
    assert 0 < resultado["wape"] < 1


def test_planilha_previsao_sem_negativos(resultado_planilha):
    for previsao in resultado_planilha["previsoes"].values():
        assert len(previsao) == 28
        assert (previsao.drop(columns="data") >= 0).all().all()
