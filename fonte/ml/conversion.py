"""Probabilidade de pagamento dos pedidos: classificador sem vazamento e fila de pendentes.

Usa só variáveis conhecidas na criação do pedido (ml.features.COLUNAS_PEDIDO). Promotora,
endereço, forma de pagamento, nome do pagador e data de pagamento são proibidos: são
preenchidos no checkout e revelam o resultado (decisão 8 do CLAUDE.md).

Validação temporal: treina nos 80% mais antigos e testa nos 20% mais recentes.
Funções puras: não dependem de Streamlit, Firebase nem OpenAI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml import metrics
from ml.features import COLUNAS_PEDIDO, SEM_CUPOM, build_order_features

# Nomes canônicos (ml.schema) e nomes da planilha que nunca podem virar variável do modelo.
COLUNAS_PROIBIDAS = frozenset(
    {
        "promotora", "forma_pagamento", "pagador", "data_pagamento", "uf", "cidade",
        "Promotora", "Forma de pagamento", "Nome Pagamento", "Pago por", "Data Venda", "Pago em",
        "CEP", "Logradouro", "Número.1", "Bairro", "Cidade", "Estado",
        # Etapa do funil, definida depois da criação: quase todo pendente não tem link, todo expirado tem.
        "link_pagamento", "gerou_link", "Link de pagamento",
    }
)
STATUS_ABERTOS = ("pendente", "aguardando pagamento")
STATUS_PERDIDOS = ("expirado", "cancelado")
DIAS_RESOLUCAO = 14  # 99% dos pagamentos saem em até 8 dias; em aberto há mais que isso = não pagou
FRACAO_TREINO = 0.8
N_CUPONS = 15
CUPOM_OUTRO = "outro"
MIN_PEDIDOS = 100

_CATEGORICAS = ["produto", "cupom"]
_NUMERICAS = ["valor", "hora", "dia_semana", "is_b2b", "pedidos_anteriores", "tentativas_no_dia"]


def _rotulos_conhecidos(df_clean: pd.DataFrame, dias_resolucao: int) -> pd.Series:
    """Marca os pedidos com desfecho conhecido, os únicos que podem treinar o modelo.

    Pago, expirado e cancelado estão decididos. Em aberto há mais de `dias_resolucao` dias
    conta como não pago; os mais novos ficam de fora porque ainda podem pagar. Pedidos de
    R$ 0,00 saem: são testes/cortesias, não uma decisão de pagamento.
    """
    idade = (df_clean["data_pedido"].max() - df_clean["data_pedido"]).dt.days
    decidido = df_clean["status"].isin(("pago", *STATUS_PERDIDOS))
    aberto_antigo = df_clean["status"].isin(STATUS_ABERTOS) & (idade > dias_resolucao)
    return (decidido | aberto_antigo) & df_clean["data_pedido"].notna() & ~df_clean["valor_zero"].astype(bool)


def cupons_mais_usados(cupom: pd.Series, n: int = N_CUPONS) -> list[str]:
    """Os `n` cupons mais frequentes (empate em ordem alfabética), sem contar "sem cupom"."""
    contagem = cupom[cupom != SEM_CUPOM].value_counts().sort_index()
    return contagem.sort_values(ascending=False, kind="stable").index[:n].tolist()


def agrupar_cupons(X: pd.DataFrame, cupons: list[str]) -> pd.DataFrame:
    """Cupom fora da lista vira "outro"; "sem cupom" continua separado."""
    resultado = X.copy()
    resultado["cupom"] = X["cupom"].where(X["cupom"].isin([*cupons, SEM_CUPOM]), CUPOM_OUTRO)
    return resultado


def _modelos() -> dict:
    """Candidatos, ainda sem treinar, com a mesma preparação das variáveis."""
    from lightgbm import LGBMClassifier

    def preparar():
        numericas = Pipeline([("imputar", SimpleImputer(strategy="median")), ("escala", StandardScaler())])
        return ColumnTransformer(
            [
                ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CATEGORICAS),
                ("numericas", numericas, _NUMERICAS),
            ]
        )

    return {
        "regressao_logistica": Pipeline(
            [("preparar", preparar()), ("modelo", LogisticRegression(max_iter=2000))]
        ),
        "lightgbm": Pipeline(
            [
                ("preparar", preparar()),
                (
                    "modelo",
                    LGBMClassifier(
                        n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                        random_state=42, n_jobs=1, deterministic=True, verbose=-1,
                    ),
                ),
            ]
        ),
    }


def corte_temporal(datas: pd.Series, fracao_treino: float = FRACAO_TREINO) -> tuple[pd.Index, pd.Index]:
    """Índices de treino (mais antigos) e teste (mais recentes), nunca sorteados."""
    ordenado = datas.sort_values(kind="stable")
    n_treino = int(len(ordenado) * fracao_treino)
    return ordenado.index[:n_treino], ordenado.index[n_treino:]


def train_conversion(df_clean: pd.DataFrame, dias_resolucao: int = DIAS_RESOLUCAO) -> dict:
    """Treina o classificador de pagamento e devolve modelo, métricas e importância das variáveis.

    Compara regressão logística e LightGBM no corte temporal 80/20 e fica com o de maior AUC.
    O vencedor é treinado de novo com todos os pedidos de desfecho conhecido para pontuar a fila.
    Chaves: modelo, cupons, vencedor, auc, precisao_media, comparacao, importancia, validacao, variaveis.
    """
    faltando = [c for c in ("data_pedido", "status", "valor", "pago", "is_b2b", "valor_zero") if c not in df_clean]
    if faltando:
        raise ValueError("A base não tem as colunas necessárias para o modelo de pagamento: " + ", ".join(faltando))

    conhecidos = _rotulos_conhecidos(df_clean, dias_resolucao)
    X_todos = build_order_features(df_clean).loc[conhecidos]
    y_todos = df_clean.loc[conhecidos, "pago"].astype(int)
    if len(y_todos) < MIN_PEDIDOS:
        raise ValueError(f"São necessários pelo menos {MIN_PEDIDOS} pedidos com desfecho conhecido; há {len(y_todos)}.")

    datas = df_clean.loc[conhecidos, "data_pedido"]
    idx_treino, idx_teste = corte_temporal(datas)
    if y_todos.loc[idx_treino].nunique() < 2 or y_todos.loc[idx_teste].nunique() < 2:
        raise ValueError("O treino ou o teste só tem pedidos pagos, ou só não pagos; não dá para avaliar o modelo.")

    # A lista de cupons sai só do treino: o teste não pode influenciar a preparação.
    cupons = cupons_mais_usados(X_todos.loc[idx_treino, "cupom"])
    X_treino = agrupar_cupons(X_todos.loc[idx_treino], cupons)
    X_teste = agrupar_cupons(X_todos.loc[idx_teste], cupons)
    y_treino, y_teste = y_todos.loc[idx_treino], y_todos.loc[idx_teste]

    linhas, ajustados = [], {}
    for nome, modelo in _modelos().items():
        modelo.fit(X_treino, y_treino)
        probabilidade = modelo.predict_proba(X_teste)[:, 1]
        ajustados[nome] = modelo
        linhas.append(
            {
                "modelo": nome,
                "auc": metrics.auc(y_teste, probabilidade),
                "precisao_media": metrics.precisao_media(y_teste, probabilidade),
            }
        )
    comparacao = pd.DataFrame(linhas).sort_values("auc", ascending=False, kind="stable").reset_index(drop=True)
    vencedor = str(comparacao.iloc[0]["modelo"])

    # Permutação no teste: quanto o AUC cai ao embaralhar cada variável original.
    permutado = permutation_importance(
        ajustados[vencedor], X_teste, y_teste, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=1
    )
    importancia = (
        pd.DataFrame({"variavel": list(X_teste.columns), "importancia": permutado.importances_mean})
        .sort_values("importancia", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    cupons_finais = cupons_mais_usados(X_todos["cupom"])
    modelo_final = _modelos()[vencedor].fit(agrupar_cupons(X_todos, cupons_finais), y_todos)

    return {
        "modelo": modelo_final,
        "cupons": cupons_finais,
        "vencedor": vencedor,
        "auc": float(comparacao.iloc[0]["auc"]),
        "precisao_media": float(comparacao.iloc[0]["precisao_media"]),
        "comparacao": comparacao,
        "importancia": importancia,
        "validacao": {
            "n_treino": int(len(idx_treino)),
            "n_teste": int(len(idx_teste)),
            "fim_treino": datas.loc[idx_treino].max().isoformat(),
            "inicio_teste": datas.loc[idx_teste].min().isoformat(),
            "taxa_pago_treino": float(y_treino.mean()),
            "taxa_pago_teste": float(y_teste.mean()),
            "dias_resolucao": int(dias_resolucao),
        },
        "variaveis": list(COLUNAS_PEDIDO),
    }


def _totais(pedidos: pd.DataFrame) -> dict:
    """Quantidade, receita esperada e valor em aberto de um conjunto de pedidos pontuados."""
    return {
        "n_pedidos": int(len(pedidos)),
        "receita_esperada": float(pedidos["receita_esperada"].sum()),
        "valor_em_aberto": float(pedidos["valor"].sum()),
    }


def score_pending(df_clean: pd.DataFrame, treino: dict | None = None) -> dict:
    """Pontua os pedidos pendentes/aguardando pagamento, do mais provável ao menos provável.

    Devolve "pedidos" (DataFrame sem dados pessoais: pedido_id, data_pedido, idade_dias, status,
    produto, valor, gerou_link, tentativa_abandonada, probabilidade, receita_esperada) e três
    níveis de totais (n_pedidos, receita_esperada = soma de valor x probabilidade, valor_em_aberto):
    a fila inteira, no próprio dict; "recentes", só os abertos há até DIAS_RESOLUCAO dias, com
    n_abandonadas; e "recuperaveis", os recentes cujo comprador não tem outro pedido pago.
    Sem `treino`, treina o modelo antes.
    """
    treino = train_conversion(df_clean) if treino is None else treino
    dias_resolucao = treino["validacao"]["dias_resolucao"]
    abertos = df_clean["status"].isin(STATUS_ABERTOS) & df_clean["data_pedido"].notna()

    # As variáveis saem da base inteira: o histórico do cliente inclui pedidos já decididos.
    X = agrupar_cupons(build_order_features(df_clean).loc[abertos], treino["cupons"])
    probabilidade = treino["modelo"].predict_proba(X)[:, 1] if len(X) else np.empty(0)

    colunas = ("pedido_id", "data_pedido", "status", "produto", "valor", "gerou_link")
    fila = df_clean.loc[abertos, [c for c in colunas if c in df_clean]].copy()
    fila.insert(2, "idade_dias", (df_clean["data_pedido"].max() - fila["data_pedido"]).dt.days)
    # O pedido em aberto nunca é pago, então achar o comprador aqui já significa "outro pedido".
    compradores_que_pagaram = set(df_clean.loc[df_clean["pago"].astype(bool), "cliente"].dropna())
    fila["tentativa_abandonada"] = df_clean.loc[abertos, "cliente"].isin(compradores_que_pagaram)
    fila["probabilidade"] = probabilidade
    fila["receita_esperada"] = fila["valor"] * fila["probabilidade"]
    fila = fila.sort_values(["probabilidade", "valor"], ascending=False, kind="stable").reset_index(drop=True)

    recentes = fila[fila["idade_dias"] <= dias_resolucao]
    return {
        "pedidos": fila,
        **_totais(fila),
        "recentes": {**_totais(recentes), "n_abandonadas": int(recentes["tentativa_abandonada"].sum())},
        "recuperaveis": _totais(recentes[~recentes["tentativa_abandonada"]]),
    }
