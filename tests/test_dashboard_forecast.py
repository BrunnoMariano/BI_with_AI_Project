"""Testes da Fase 5: visual "Previsão" do Dashboard (dashboard_service).

O visual não usa SQL do LLM: lê o resultado de ml.pipeline.run_predictive_pipeline.
"""

import json

import pandas as pd
import pytest

import dashboard_service as ds
from ml.features import build_daily_series
from ml.pipeline import run_predictive_pipeline
from ml.preprocessing import clean_orders
from tests.test_forecasting import _planilha_sintetica


@pytest.fixture(scope="module")
def planilha():
    df = _planilha_sintetica()
    # Dois produtos e um cupom para testar os recortes dos filtros globais.
    df["Produto"] = ["Certificação A" if i % 3 else "Certificação B" for i in range(len(df))]
    df["Cupom"] = ["PROMO10" if i % 2 else None for i in range(len(df))]
    df["Estado"] = "SP"
    return df


@pytest.fixture(scope="module")
def resultado(planilha):
    return run_predictive_pipeline(planilha, "teste", horizonte_previsao=90)


def _serie(spec, nome):
    return next(s for s in spec["option"]["series"] if s["name"] == nome)


def test_tipo_forecast_registrado_e_parametros_normalizados():
    assert "forecast" in ds.DASHBOARD_VISUAL_TYPES
    assert ds.normalize_forecast_params(None) == {"alvo": "receita", "granularidade": "dia", "horizonte": 28}
    assert ds.normalize_forecast_params({"alvo": "PEDIDOS", "granularidade": "mes", "horizonte": "90"}) == {
        "alvo": "pedidos", "granularidade": "mes", "horizonte": 90,
    }
    # Valores fora das opções voltam ao padrão.
    assert ds.normalize_forecast_params({"alvo": "lucro", "granularidade": "ano", "horizonte": 30})["horizonte"] == 28


@pytest.mark.parametrize("granularidade", ["dia", "semana", "mes"])
@pytest.mark.parametrize("alvo", ["pedidos", "receita"])
def test_build_forecast_spec(resultado, alvo, granularidade):
    spec = ds.build_forecast_spec(resultado, alvo, granularidade, 90)
    assert spec["render_mode"] == "echarts"
    option = spec["option"]
    categorias = option["xAxis"]["data"]

    real, previsto = _serie(spec, "Real"), _serie(spec, "Previsto")
    base, faixa = _serie(spec, "Faixa 80% (base)"), _serie(spec, "Faixa de 80%")
    assert real["lineStyle"].get("type", "solid") == "solid"
    assert previsto["lineStyle"]["type"] == "dashed"
    for serie in option["series"]:
        assert len(serie["data"]) == len(categorias)

    # Faixa: limite inferior transparente + diferença empilhada com área.
    assert base["stack"] == faixa["stack"] == "faixa_80"
    assert base["lineStyle"]["opacity"] == 0 and "areaStyle" not in base
    assert "areaStyle" in faixa
    assert base["tooltip"] == {"show": False} and faixa["tooltip"] == {"show": False}
    superior = _serie(spec, "Limite superior 80%")["data"]
    for b, d, s in zip(base["data"], faixa["data"], superior):
        if s is not None:
            assert b + d == pytest.approx(s, abs=0.02)
            assert d >= 0

    # "hoje" = primeiro período previsto; a linha prevista nasce no último ponto real.
    n_hist = sum(v is not None for v in real["data"])
    marca = previsto["markLine"]["data"][0]["xAxis"]
    assert marca == categorias[n_hist]
    inicio = pd.Timestamp(resultado["metadados"]["alvos"][alvo]["inicio_previsao"])
    assert previsto["markLine"]["label"]["formatter"] == f"hoje · {inicio:%d/%m/%Y}"
    assert previsto["data"][n_hist - 1] == real["data"][n_hist - 1]
    assert all(v is None for v in previsto["data"][: n_hist - 1])

    # Precisa virar JSON puro (o frontend repassa o option direto ao ECharts).
    json.dumps(ds._json_safe(spec), allow_nan=False)


def test_spec_dia_respeita_horizonte_e_avisa_acima_de_28(resultado):
    spec_28 = ds.build_forecast_spec(resultado, "receita", "dia", 28)
    previsto = _serie(spec_28, "Previsto")["data"]
    real = _serie(spec_28, "Real")["data"]
    assert sum(v is None for v in real) == 28
    assert sum(v is not None for v in real) == ds.FORECAST_HISTORY_PERIODS["dia"]
    assert previsto[-28:] == [round(v, 2) for v in resultado["previsoes"]["receita"]["previsto"].iloc[:28]]
    assert spec_28["avisos"] == []

    spec_90 = ds.build_forecast_spec(resultado, "receita", "dia", 90)
    assert any("não foi validada" in aviso for aviso in spec_90["avisos"])
    assert "não foi validada" in spec_90["option"]["title"]["subtext"]


def test_split_forecast_filters():
    filtros = [
        {"filter_id": "1", "column": "Produto", "selected_values": ["Certificação B"]},
        {"filter_id": "2", "column": "Cupom", "selected_values": ["PROMO10"]},
        {"filter_id": "3", "column": "Estado", "title": "UF", "selected_values": ["SP"]},
        {"filter_id": "4", "column": "Cidade", "selected_values": []},  # sem seleção: não conta
    ]
    aplicaveis, ignorados = ds.split_forecast_filters(filtros)
    assert [f["column"] for f in aplicaveis] == ["Produto", "Cupom"]
    assert [f["title"] for f in ignorados] == ["UF"]
    assert ds.forecast_filter_key([]) == ""
    assert ds.forecast_filter_key(aplicaveis) == ds.forecast_filter_key(list(reversed(aplicaveis)))


def test_build_render_items_refaz_previsao_no_recorte(planilha, resultado):
    chamadas = []

    def runner(df_recorte, chave):
        chamadas.append((df_recorte, chave))
        return resultado

    item = ds.build_forecast_item({"alvo": "pedidos", "granularidade": "semana", "horizonte": 28})
    assert item["visual_type"] == "forecast" and item["query_text"] == ""
    assert item["title"] == "Previsão de pedidos · próximos 28 dias"
    registro = {"items": [item], "filters": []}
    filtros = [
        {"filter_id": "p", "column": "Produto", "title": "Produto", "selected_values": ["Certificação B"]},
        {"filter_id": "e", "column": "Estado", "title": "Estado", "selected_values": ["SP"]},
    ]

    payload = ds.build_render_items(registro, db_mode=False, df_data=planilha, active_filters=filtros, forecast_runner=runner)[0]
    assert payload["status"] == "ready" and payload["render_spec"]["render_mode"] == "echarts"
    df_recorte, chave = chamadas[-1]
    assert set(df_recorte["Produto"]) == {"Certificação B"}
    assert len(df_recorte) < len(planilha)  # o Estado não filtrou nada além do Produto
    assert "Produto" in chave
    avisos = payload["render_spec"]["avisos"]
    assert any("Previsão refeita para Produto: Certificação B" in a for a in avisos)
    assert any('"Estado" não se aplica à previsão' in a for a in avisos)

    # Sem filtros: base inteira, chave vazia.
    ds.build_render_items(registro, db_mode=False, df_data=planilha, active_filters=[], forecast_runner=runner)
    assert chamadas[-1][1] == "" and len(chamadas[-1][0]) == len(planilha)

    # "Ignorar filtros globais" também vale para a previsão.
    item_ignora = ds.build_forecast_item({}, filter_behavior={"ignore_global_filters": True})
    ds.build_render_items({"items": [item_ignora]}, db_mode=False, df_data=planilha, active_filters=filtros, forecast_runner=runner)
    assert chamadas[-1][1] == ""


def test_build_render_items_forecast_sem_arquivo_ou_banco(planilha):
    item = ds.build_forecast_item({})
    banco = ds.build_render_items({"items": [item]}, db_mode=True, uri="postgresql://x")[0]
    assert banco["render_spec"]["render_mode"] == "error"
    sem_runner = ds.build_render_items({"items": [item]}, db_mode=False, df_data=planilha)[0]
    assert sem_runner["render_spec"]["render_mode"] == "error"

    def runner_com_erro(df_recorte, chave):
        raise ValueError("Não foi possível prever para este recorte: série curta.")

    falha = ds.build_render_items({"items": [item]}, db_mode=False, df_data=planilha, forecast_runner=runner_com_erro)[0]
    assert falha["status"] == "error" and "recorte" in falha["render_spec"]["message"]


def test_recorte_termina_no_mesmo_dia_da_base(planilha, resultado):
    """Com fim fixo, um recorte (um produto) prevê a partir do mesmo "hoje" da base inteira."""
    meta = resultado["metadados"]["alvos"]["receita"]
    recorte = planilha[planilha["Produto"] == "Certificação B"]
    # Sem vendas do produto nos últimos dias, a série terminaria antes.
    ultimo = pd.to_datetime(recorte["Criado em"], format="%d/%m/%Y %H:%M:%S").max()
    recorte = recorte[pd.to_datetime(recorte["Criado em"], format="%d/%m/%Y %H:%M:%S") < ultimo - pd.Timedelta(days=5)]

    serie = build_daily_series(clean_orders(recorte), "pedidos", inicio=meta["inicio_treino"], fim=meta["fim_treino"])
    assert serie.index[-1] == pd.Timestamp(meta["fim_treino"])
    assert serie.iloc[-5:].sum() == 0

    res_recorte = run_predictive_pipeline(recorte, "teste|B", inicio=meta["inicio_treino"], fim=meta["fim_treino"], horizonte_previsao=28)
    assert res_recorte["metadados"]["alvos"]["receita"]["inicio_previsao"] == meta["inicio_previsao"]
    assert res_recorte["metadados"]["alvos"]["pedidos"]["total_treino"] < resultado["metadados"]["alvos"]["pedidos"]["total_treino"]


def test_planilha_real_hoje_em_07_04_2026(vendas_df):
    """Na base oficial, a previsão parte de 07/04/2026 (só B2C, data = Criado em)."""
    res = run_predictive_pipeline(vendas_df, "real", horizonte_previsao=28)
    spec = ds.build_forecast_spec(res, "receita", "dia", 28)
    previsto = _serie(spec, "Previsto")
    assert previsto["markLine"]["label"]["formatter"] == "hoje · 07/04/2026"
    assert previsto["markLine"]["data"][0]["xAxis"] == "07/04/26"
    assert res["metadados"]["apenas_b2c"] is True
