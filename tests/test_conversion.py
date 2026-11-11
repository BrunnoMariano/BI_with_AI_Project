"""Testes da Fase 6: classificador de pagamento (ml.conversion) e variáveis do pedido.

O ponto central é garantir que nenhuma coluna proibida (preenchida no checkout) entra no modelo.
"""

import numpy as np
import pandas as pd
import pytest

from ml import metrics
from ml.conversion import (
    CUPOM_OUTRO,
    COLUNAS_PROIBIDAS,
    agrupar_cupons,
    corte_temporal,
    cupons_mais_usados,
    score_pending,
    train_conversion,
)
from ml.features import COLUNAS_PEDIDO, SEM_CUPOM, build_order_features
from ml.preprocessing import clean_orders
from ml.schema import COLUMN_ALIASES

CANONICAS_PROIBIDAS = [
    "promotora", "forma_pagamento", "pagador", "data_pagamento", "uf", "cidade", "link_pagamento", "gerou_link",
]


def _pedidos(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Pedidos fictícios no formato de clean_orders; ter cupom aumenta a chance de pagar."""
    rng = np.random.default_rng(seed)
    datas = pd.Timestamp("2025-08-01 08:00") + pd.to_timedelta(np.sort(rng.integers(0, 200 * 24 * 60, n)), unit="min")
    cupom = rng.choice(["CUPOM_A", "CUPOM_B", None], n, p=[0.3, 0.2, 0.5])
    paga = rng.random(n) < np.where(pd.notna(cupom), 0.85, 0.35)
    status = np.where(paga, "pago", rng.choice(["expirado", "cancelado", "pendente"], n))
    valor = rng.choice([71.2, 89.0, 120.0], n)
    df = pd.DataFrame(
        {
            "pedido_id": range(1, n + 1),
            "data_pedido": datas,
            "data_pagamento": datas + pd.Timedelta(hours=1),
            "valor": valor,
            "status": status,
            "produto": rng.choice(["Prova A", "Prova B"], n),
            "cupom": cupom,
            "promotora": rng.choice(["bevi", "outra"], n),
            "forma_pagamento": rng.choice(["pix", "boleto"], n),
            "cliente": [f"cliente {i}" for i in rng.integers(0, 800, n)],
            "pagador": "pessoa",
            "conta": None,
            "uf": rng.choice(["SP", "RJ"], n),
            "cidade": "cidade",
            "link_pagamento": rng.choice(["Link de pagamento", "---"], n),
        }
    )
    df["gerou_link"] = df["link_pagamento"].ne("---")
    df["pago"] = df["status"].eq("pago")
    df["is_b2b"] = False
    df["valor_zero"] = df["valor"].eq(0)
    return df


@pytest.fixture(scope="module")
def pedidos():
    return _pedidos()


@pytest.fixture(scope="module")
def treino(pedidos):
    return train_conversion(pedidos)


# ---------- nenhuma coluna proibida entra no modelo ----------

def test_lista_de_variaveis_nao_tem_coluna_proibida():
    assert not set(COLUNAS_PEDIDO) & COLUNAS_PROIBIDAS
    for canonica in CANONICAS_PROIBIDAS:
        assert canonica in COLUNAS_PROIBIDAS
        assert canonica not in COLUNAS_PEDIDO
        for nome_planilha in COLUMN_ALIASES.get(canonica, []):
            assert nome_planilha in COLUNAS_PROIBIDAS


def test_build_order_features_devolve_so_as_colunas_permitidas(pedidos):
    X = build_order_features(pedidos)
    assert list(X.columns) == list(COLUNAS_PEDIDO)
    assert not set(X.columns) & COLUNAS_PROIBIDAS


def test_modelo_treinado_so_recebe_as_colunas_permitidas(treino):
    assert list(treino["modelo"].feature_names_in_) == list(COLUNAS_PEDIDO)
    assert treino["variaveis"] == list(COLUNAS_PEDIDO)
    assert set(treino["importancia"]["variavel"]) == set(COLUNAS_PEDIDO)
    nomes_internos = treino["modelo"].named_steps["preparar"].get_feature_names_out()
    for proibida in CANONICAS_PROIBIDAS:
        assert not any(proibida in nome for nome in nomes_internos)


def test_colunas_proibidas_nao_mudam_nenhuma_probabilidade(pedidos, treino):
    """Prova de comportamento: embaralhar ou apagar campos proibidos não altera o resultado."""
    rng = np.random.default_rng(0)
    embaralhado = pedidos.copy()
    for coluna in CANONICAS_PROIBIDAS:
        embaralhado[coluna] = rng.permutation(embaralhado[coluna].to_numpy())
    sem_proibidas = pedidos.drop(columns=CANONICAS_PROIBIDAS)

    original = score_pending(pedidos, treino)["pedidos"]["probabilidade"].to_numpy()
    for variante in (embaralhado, sem_proibidas):
        retreinado = train_conversion(variante)
        np.testing.assert_allclose(score_pending(variante, retreinado)["pedidos"]["probabilidade"], original)
        assert retreinado["auc"] == pytest.approx(treino["auc"])


def test_status_nao_vira_variavel(pedidos):
    assert "status" not in build_order_features(pedidos) and "pago" not in build_order_features(pedidos)


# ---------- variáveis do pedido ----------

def _historico_cliente() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data_pedido": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-01 12:00", "2026-01-03 09:00", "2026-01-01 11:00", "2026-01-02 09:00"]
            ),
            "valor": [10.0, 10.0, 10.0, 20.0, 30.0],
            "produto": ["A", "A", None, "B", "B"],
            "cupom": ["X", None, "X", None, None],
            "cliente": ["ana", "ana", "ana", "bia", None],
            "is_b2b": [False, False, False, True, False],
        }
    )


def test_pedidos_anteriores_e_tentativas_olham_so_para_tras():
    X = build_order_features(_historico_cliente())
    assert X["pedidos_anteriores"].tolist() == [0, 1, 2, 0, 0]
    assert X["tentativas_no_dia"].tolist() == [1, 2, 1, 1, 1]
    assert X["hora"].tolist() == [10, 12, 9, 11, 9]
    assert X["dia_semana"].tolist() == [3, 3, 5, 3, 4]
    assert X["is_b2b"].tolist() == [0, 0, 0, 1, 0]
    assert X["cupom"].tolist() == ["X", SEM_CUPOM, "X", SEM_CUPOM, SEM_CUPOM]
    assert X["produto"].iloc[2] == "desconhecido"


def test_pedido_futuro_nao_altera_variaveis_do_passado():
    base = _historico_cliente()
    novo = pd.Timestamp("2026-01-01 23:00")  # mesmo dia e mesmo cliente dos primeiros pedidos
    futuro = pd.concat([base, base.iloc[[0]].assign(data_pedido=novo)], ignore_index=True)
    antes_do_novo = base.index[base["data_pedido"] < novo]
    pd.testing.assert_frame_equal(build_order_features(futuro).loc[antes_do_novo], build_order_features(base).loc[antes_do_novo])
    assert build_order_features(futuro)["tentativas_no_dia"].iloc[-1] == 3  # o novo pedido enxerga os dois anteriores


def test_build_order_features_sem_coluna_da_erro():
    with pytest.raises(ValueError, match="cupom"):
        build_order_features(_historico_cliente().drop(columns="cupom"))


def test_cupons_top_e_outro():
    cupons = pd.Series([f"C{i:02d}" for i in range(20) for _ in range(20 - i)] + [SEM_CUPOM] * 100)
    top = cupons_mais_usados(cupons)
    assert top == [f"C{i:02d}" for i in range(15)]
    agrupado = agrupar_cupons(pd.DataFrame({"cupom": ["C00", "C19", SEM_CUPOM]}), top)
    assert agrupado["cupom"].tolist() == ["C00", CUPOM_OUTRO, SEM_CUPOM]


# ---------- validação e rótulos ----------

def test_corte_temporal_nunca_mistura_passado_e_futuro():
    datas = pd.Series(pd.date_range("2026-01-01", periods=100, freq="D")).sample(frac=1, random_state=1)
    treino, teste = corte_temporal(datas)
    assert len(treino) == 80 and len(teste) == 20
    assert datas.loc[treino].max() < datas.loc[teste].min()


def test_train_conversion_reporta_metricas(treino):
    validacao = treino["validacao"]
    assert pd.Timestamp(validacao["fim_treino"]) <= pd.Timestamp(validacao["inicio_teste"])
    assert treino["auc"] == pytest.approx(treino["comparacao"]["auc"].max())
    assert set(treino["comparacao"]["modelo"]) == {"regressao_logistica", "lightgbm"}
    assert treino["vencedor"] == treino["comparacao"].iloc[0]["modelo"]
    assert 0.5 < treino["auc"] <= 1 and 0 < treino["precisao_media"] <= 1
    assert treino["importancia"]["importancia"].is_monotonic_decreasing
    assert treino["importancia"].iloc[0]["variavel"] == "cupom"  # é o sinal plantado nos dados


def test_abertos_recentes_ficam_fora_do_treino_e_antigos_contam_como_nao_pagos():
    df = _pedidos()
    fim = df["data_pedido"].max()
    abertos = df["status"].eq("pendente")
    recentes = abertos & ((fim - df["data_pedido"]).dt.days <= 14)
    antigos = abertos & ~recentes
    decididos = int((~abertos).sum())
    assert recentes.sum() > 0 and antigos.sum() > 0

    resultado = train_conversion(df, dias_resolucao=14)
    total = resultado["validacao"]["n_treino"] + resultado["validacao"]["n_teste"]
    assert total == decididos + int(antigos.sum())


def test_pedidos_de_valor_zero_ficam_fora_do_treino():
    df = _pedidos()
    base = train_conversion(df)["validacao"]
    df.loc[df.index[:50], "valor"] = 0.0
    df["valor_zero"] = df["valor"].eq(0)
    assert train_conversion(df)["validacao"]["n_treino"] < base["n_treino"]


def test_train_conversion_poucos_pedidos_da_erro():
    with pytest.raises(ValueError, match="pelo menos"):
        train_conversion(_pedidos(50))


def test_metricas_de_classificacao():
    assert metrics.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert metrics.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert metrics.precisao_media([0, 1], [0.2, 0.9]) == pytest.approx(1.0)
    assert np.isnan(metrics.auc([1, 1], [0.2, 0.9]))
    assert np.isnan(metrics.precisao_media([0, 0], [0.2, 0.9]))


# ---------- score_pending ----------

def test_score_pending_ordem_e_soma(pedidos, treino):
    fila = score_pending(pedidos, treino)
    tabela = fila["pedidos"]
    assert fila["n_pedidos"] == int(pedidos["status"].eq("pendente").sum()) == len(tabela)
    assert set(tabela["status"]) <= {"pendente", "aguardando pagamento"}
    assert tabela["probabilidade"].is_monotonic_decreasing
    assert tabela["probabilidade"].between(0, 1).all()
    assert fila["receita_esperada"] == pytest.approx((tabela["valor"] * tabela["probabilidade"]).sum())
    assert fila["valor_em_aberto"] == pytest.approx(tabela["valor"].sum())
    assert fila["recentes"]["receita_esperada"] <= fila["receita_esperada"]


def test_kpi_principal_so_recentes_sem_tentativas_abandonadas(pedidos, treino):
    fila = score_pending(pedidos, treino)
    tabela = fila["pedidos"]
    pagaram = set(pedidos.loc[pedidos["pago"], "cliente"])
    por_id = pedidos.set_index("pedido_id")["cliente"]
    abandonada = tabela["pedido_id"].map(por_id).isin(pagaram)
    np.testing.assert_array_equal(tabela["tentativa_abandonada"], abandonada)

    recentes = tabela[tabela["idade_dias"] <= treino["validacao"]["dias_resolucao"]]
    validos = recentes[~recentes["tentativa_abandonada"]]
    assert 0 < fila["recentes"]["n_abandonadas"] < fila["recentes"]["n_pedidos"]
    assert fila["recuperaveis"]["n_pedidos"] == fila["recentes"]["n_pedidos"] - fila["recentes"]["n_abandonadas"]
    assert fila["recuperaveis"]["receita_esperada"] == pytest.approx((validos["valor"] * validos["probabilidade"]).sum())
    assert fila["recuperaveis"]["valor_em_aberto"] == pytest.approx(validos["valor"].sum())
    assert fila["recuperaveis"]["receita_esperada"] < fila["recentes"]["receita_esperada"] < fila["receita_esperada"]


def test_tentativa_abandonada_exige_outro_pedido_pago_do_mesmo_comprador(pedidos, treino):
    df = pedidos.copy()
    pendentes = df.index[df["status"].eq("pendente")]
    ultimo, penultimo = pendentes[-1], pendentes[-2]
    pago = df.index[df["pago"]][0]
    df.loc[df["cliente"].isin(["so_pendente", "ja_pagou"]), "cliente"] = "outro"
    df.loc[ultimo, "cliente"] = "so_pendente"
    df.loc[penultimo, "cliente"] = "ja_pagou"
    df.loc[pago, "cliente"] = "ja_pagou"

    marcados = score_pending(df, treino)["pedidos"].set_index("pedido_id")["tentativa_abandonada"]
    assert not marcados.loc[df.loc[ultimo, "pedido_id"]]
    assert marcados.loc[df.loc[penultimo, "pedido_id"]]


def test_gerou_link_vem_da_coluna_e_chega_na_fila(pedidos, treino):
    limpo = clean_orders(pd.DataFrame({"Link de pagamento": ["Link de pagamento", "---", None, " --- "]}))
    assert limpo["gerou_link"].tolist() == [True, False, False, False]
    assert "gerou_link" in score_pending(pedidos, treino)["pedidos"]
    assert "gerou_link" not in score_pending(pedidos.drop(columns="gerou_link"), treino)["pedidos"]


def test_score_pending_nao_expoe_dados_pessoais(pedidos, treino):
    assert not {"cliente", "pagador", "uf", "cidade"} & set(score_pending(pedidos, treino)["pedidos"].columns)


def test_score_pending_treina_sozinho_e_inclui_aguardando(pedidos):
    df = pedidos.copy()
    df.loc[df["status"].eq("pendente").idxmax(), "status"] = "aguardando pagamento"
    fila = score_pending(df)
    assert "aguardando pagamento" in set(fila["pedidos"]["status"])


def test_score_pending_sem_fila(pedidos, treino):
    sem_abertos = pedidos[~pedidos["status"].eq("pendente")]
    fila = score_pending(sem_abertos, treino)
    assert fila["n_pedidos"] == 0 and fila["receita_esperada"] == 0.0 and fila["pedidos"].empty


# ---------- planilha oficial ----------

@pytest.fixture(scope="module")
def pedidos_planilha(vendas_df):
    return clean_orders(vendas_df)


@pytest.fixture(scope="module")
def treino_planilha(pedidos_planilha):
    return train_conversion(pedidos_planilha)


def test_planilha_auc_proximo_do_prototipo(treino_planilha):
    assert treino_planilha["auc"] == pytest.approx(0.73, abs=0.04)


def test_planilha_fila_e_receita_esperada(pedidos_planilha, treino_planilha):
    fila = score_pending(pedidos_planilha, treino_planilha)
    assert fila["n_pedidos"] == 635  # 615 pendentes + 20 aguardando pagamento
    assert fila["valor_em_aberto"] == pytest.approx(67594.80, abs=0.01)
    assert 0 < fila["receita_esperada"] < fila["valor_em_aberto"]
    assert fila["recentes"]["n_pedidos"] == 64
    assert fila["recentes"]["n_abandonadas"] == 39
    assert fila["recuperaveis"]["n_pedidos"] == 25
    assert 0 < fila["recuperaveis"]["receita_esperada"] < fila["recentes"]["receita_esperada"]
