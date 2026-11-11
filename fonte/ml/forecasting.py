"""Modelos de previsão da série diária, backtest temporal e escolha do melhor modelo.

Todos os modelos têm a mesma assinatura: recebem a série de treino (pd.Series com
índice diário) e o horizonte h, e devolvem um array com h previsões (nunca negativas).

Validação sempre temporal (decisão 7 do CLAUDE.md): backtest com origem deslizante,
cortes consecutivos de `horizonte` dias no fim da série; o vencedor é o de menor WAPE
médio. Os intervalos de 80%/95% vêm dos quantis dos resíduos (real - previsto) do backtest.

Funções puras: não dependem de Streamlit, Firebase nem OpenAI.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml import metrics
from ml.features import agregar_serie, calendar_features, dias_no_periodo, periodos

HORIZONTE_PADRAO = 28
N_CORTES_PADRAO = 4
MIN_DIAS_TREINO = 56  # mínimo de dias no primeiro corte do backtest

# Nível do intervalo -> (quantil inferior, quantil superior) dos resíduos.
QUANTIS_INTERVALO = {"80": (0.10, 0.90), "95": (0.025, 0.975)}

LAGS_LGBM = (28, 35, 42)
JANELA_MEDIA_LGBM = 28
COLUNAS_PREVISAO = ["data", "previsto", "lim_inf_80", "lim_sup_80", "lim_inf_95", "lim_sup_95"]


# ---------- modelos ----------

def baseline_sazonal_7d(treino: pd.Series, h: int) -> np.ndarray:
    """Repete a última semana: a previsão de cada dia é o valor do mesmo dia da semana anterior."""
    ultima_semana = np.asarray(treino, dtype=float)[-7:]
    return np.resize(ultima_semana, h)


def media_movel_28d(treino: pd.Series, h: int) -> np.ndarray:
    """Valor constante igual à média dos últimos 28 dias."""
    return np.full(h, float(np.mean(np.asarray(treino, dtype=float)[-28:])))


def holt_winters(treino: pd.Series, h: int) -> np.ndarray:
    """Holt-Winters aditivo: tendência amortecida + sazonalidade semanal (7 dias)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # avisos de convergência do otimizador
        ajuste = ExponentialSmoothing(
            np.asarray(treino, dtype=float),
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        ).fit()
    return np.asarray(ajuste.forecast(h), dtype=float)


def _features_lightgbm(y: pd.Series) -> pd.DataFrame:
    """Calendário + lags de 28/35/42 dias + média móvel de 28 dias defasada em 28 dias."""
    X = calendar_features(y.index)
    for lag in LAGS_LGBM:
        X[f"lag_{lag}"] = y.shift(lag).to_numpy()
    X["media_28_lag_28"] = y.shift(28).rolling(JANELA_MEDIA_LGBM, min_periods=JANELA_MEDIA_LGBM).mean().to_numpy()
    return X


def lightgbm(treino: pd.Series, h: int) -> np.ndarray:
    """LightGBM sobre calendário e lags >= 28 dias.

    Como o menor lag é 28, os 28 primeiros dias saem de uma vez, sem usar previsões;
    para h > 28 a previsão segue em blocos de 28 dias reaproveitando o bloco anterior.
    """
    from lightgbm import LGBMRegressor

    treino = pd.Series(np.asarray(treino, dtype=float), index=pd.DatetimeIndex(treino.index))
    X = _features_lightgbm(treino)
    validos = X.notna().all(axis=1)
    if validos.sum() < 28:
        raise ValueError("Série curta demais para o LightGBM (precisa de pelo menos 98 dias).")

    modelo = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        random_state=42,
        n_jobs=1,
        deterministic=True,
        verbose=-1,
    )
    modelo.fit(X[validos], treino[validos])

    futuro = pd.date_range(treino.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    y = pd.concat([treino, pd.Series(np.nan, index=futuro)])
    for inicio in range(0, h, min(LAGS_LGBM)):
        bloco = futuro[inicio : inicio + min(LAGS_LGBM)]
        X_bloco = _features_lightgbm(y).loc[bloco]
        y.loc[bloco] = np.maximum(modelo.predict(X_bloco), 0.0)
    return y.loc[futuro].to_numpy(dtype=float)


MODELOS = {
    "baseline_sazonal_7d": baseline_sazonal_7d,
    "media_movel_28d": media_movel_28d,
    "holt_winters": holt_winters,
    "lightgbm": lightgbm,
}


def prever(modelo: str, treino: pd.Series, h: int) -> np.ndarray:
    """Roda um modelo pelo nome e garante h valores sem negativos."""
    if modelo not in MODELOS:
        raise ValueError(f"Modelo desconhecido: {modelo!r}. Opções: {list(MODELOS)}")
    previsto = np.asarray(MODELOS[modelo](treino, h), dtype=float)
    if previsto.shape != (h,) or not np.isfinite(previsto).all():
        raise ValueError(f"O modelo {modelo} devolveu previsões inválidas.")
    return np.maximum(previsto, 0.0)


# ---------- intervalos ----------

def intervalos(previsto, residuos) -> dict[str, np.ndarray]:
    """Limites de 80% e 95%: previsto + quantis dos resíduos, cortados em zero."""
    previsto = np.asarray(previsto, dtype=float)
    residuos = np.asarray(residuos, dtype=float).ravel()
    residuos = residuos[np.isfinite(residuos)]
    limites = {}
    for nivel, (q_inf, q_sup) in QUANTIS_INTERVALO.items():
        if len(residuos) == 0:
            inf = sup = np.full_like(previsto, np.nan)
        else:
            inf = np.maximum(previsto + np.quantile(residuos, q_inf), 0.0)
            sup = np.maximum(previsto + np.quantile(residuos, q_sup), 0.0)
        limites[f"lim_inf_{nivel}"] = inf
        limites[f"lim_sup_{nivel}"] = sup
    return limites


# ---------- backtest ----------

def cortes_backtest(serie: pd.Series, horizonte: int = HORIZONTE_PADRAO, n_cortes: int = N_CORTES_PADRAO) -> list[dict]:
    """Janelas de teste consecutivas no fim da série (origem deslizante).

    Cada corte treina com tudo antes de `inicio_teste` e testa nos `horizonte` dias seguintes.
    """
    n = len(serie)
    if n_cortes < 1 or horizonte < 1:
        raise ValueError("horizonte e n_cortes devem ser >= 1.")
    if n - n_cortes * horizonte < MIN_DIAS_TREINO:
        raise ValueError(
            f"Série com {n} dias é curta demais para {n_cortes} cortes de {horizonte} dias "
            f"(mínimo {n_cortes * horizonte + MIN_DIAS_TREINO} dias)."
        )
    cortes = []
    for k in range(n_cortes):
        fim_treino = n - (n_cortes - k) * horizonte  # posição do primeiro dia de teste
        cortes.append(
            {
                "corte": k + 1,
                "fim_treino_pos": fim_treino,
                "inicio_teste": serie.index[fim_treino],
                "fim_teste": serie.index[fim_treino + horizonte - 1],
            }
        )
    return cortes


def backtest(
    serie: pd.Series,
    horizonte: int = HORIZONTE_PADRAO,
    n_cortes: int = N_CORTES_PADRAO,
    modelos=None,
) -> dict:
    """Backtest com origem deslizante para cada modelo.

    Devolve um dict com:
    - "tabela": métricas médias dos cortes por modelo (ordenada por WAPE);
    - "por_corte": métricas de cada corte;
    - "previsoes": real x previsto de cada corte (formato longo);
    - "residuos": {modelo: array (n_cortes, horizonte)} com real - previsto.

    A cobertura do intervalo em cada corte usa só os resíduos dos OUTROS cortes,
    para não medir o intervalo com os próprios erros que o geraram.
    """
    modelos = list(MODELOS) if modelos is None else list(modelos)
    cortes = cortes_backtest(serie, horizonte, n_cortes)
    valores = np.asarray(serie, dtype=float)

    previsoes, residuos, falhas = [], {}, {}
    for modelo in modelos:
        matriz = np.full((n_cortes, horizonte), np.nan)
        for i, corte in enumerate(cortes):
            pos = corte["fim_treino_pos"]
            treino = serie.iloc[:pos]
            real = valores[pos : pos + horizonte]
            try:
                previsto = prever(modelo, treino, horizonte)
            except Exception as erro:  # um modelo com problema não derruba os outros
                falhas[modelo] = str(erro)
                warnings.warn(f"Modelo {modelo} falhou no corte {corte['corte']}: {erro}")
                continue
            matriz[i] = real - previsto
            previsoes.append(
                pd.DataFrame(
                    {
                        "modelo": modelo,
                        "corte": corte["corte"],
                        "data": serie.index[pos : pos + horizonte],
                        "real": real,
                        "previsto": previsto,
                    }
                )
            )
        residuos[modelo] = matriz

    previsoes_df = (
        pd.concat(previsoes, ignore_index=True)
        if previsoes
        else pd.DataFrame(columns=["modelo", "corte", "data", "real", "previsto"])
    )

    linhas = []
    for modelo in modelos:
        for i, corte in enumerate(cortes):
            if np.isnan(residuos[modelo][i]).all():
                continue
            parte = previsoes_df[(previsoes_df["modelo"] == modelo) & (previsoes_df["corte"] == corte["corte"])]
            y, p = parte["real"].to_numpy(), parte["previsto"].to_numpy()
            outros = np.delete(residuos[modelo], i, axis=0) if n_cortes > 1 else residuos[modelo]
            lim = intervalos(p, outros)
            linhas.append(
                {
                    "modelo": modelo,
                    "corte": corte["corte"],
                    "inicio_teste": corte["inicio_teste"],
                    "fim_teste": corte["fim_teste"],
                    "wape": metrics.wape(y, p),
                    "mae": metrics.mae(y, p),
                    "rmse": metrics.rmse(y, p),
                    "mape_seguro": metrics.mape_seguro(y, p),
                    "erro_total": metrics.erro_total(y, p),
                    "cobertura_80": metrics.cobertura(y, lim["lim_inf_80"], lim["lim_sup_80"]),
                    "cobertura_95": metrics.cobertura(y, lim["lim_inf_95"], lim["lim_sup_95"]),
                }
            )
    por_corte = pd.DataFrame(linhas)

    colunas_metricas = ["wape", "mae", "rmse", "mape_seguro", "erro_total", "cobertura_80", "cobertura_95"]
    if por_corte.empty:
        tabela = pd.DataFrame(columns=["modelo", *colunas_metricas, "n_cortes"])
    else:
        tabela = por_corte.groupby("modelo", sort=False)[colunas_metricas].mean()
        tabela["n_cortes"] = por_corte.groupby("modelo", sort=False).size()
        # Só entra no ranking quem completou todos os cortes.
        tabela = tabela[tabela["n_cortes"] == n_cortes]
        ordem = {nome: i for i, nome in enumerate(modelos)}  # desempate: ordem da lista
        tabela = tabela.reset_index()
        tabela["_ordem"] = tabela["modelo"].map(ordem)
        tabela = tabela.sort_values(["wape", "_ordem"]).drop(columns="_ordem").reset_index(drop=True)

    return {
        "tabela": tabela,
        "por_corte": por_corte,
        "previsoes": previsoes_df,
        "residuos": residuos,
        "cortes": [{k: v for k, v in c.items() if k != "fim_treino_pos"} for c in cortes],
        "falhas": falhas,
        "horizonte": horizonte,
        "n_cortes": n_cortes,
    }


def faixas_backtest(resultado_backtest: dict, modelo: str) -> pd.DataFrame:
    """Real x previsto de cada dia do backtest de `modelo`, com faixas de 80%/95%.

    Cada corte usa só os resíduos dos OUTROS cortes (mesma regra da cobertura), para o
    dia não ser julgado por uma faixa construída com o próprio erro.
    Colunas: data, corte, real, previsto, lim_inf_80, lim_sup_80, lim_inf_95, lim_sup_95.
    """
    previsoes = resultado_backtest["previsoes"]
    matriz = resultado_backtest["residuos"][modelo]
    partes = []
    for corte, parte in previsoes[previsoes["modelo"] == modelo].groupby("corte"):
        outros = np.delete(matriz, corte - 1, axis=0) if matriz.shape[0] > 1 else matriz
        previsto = parte["previsto"].to_numpy()
        partes.append(parte[["data", "corte", "real", "previsto"]].assign(**intervalos(previsto, outros)))
    return pd.concat(partes, ignore_index=True)


# ---------- previsão final ----------

def forecast(
    serie: pd.Series,
    modelo: str = "holt_winters",
    h: int = HORIZONTE_PADRAO,
    residuos=None,
    n_cortes: int = N_CORTES_PADRAO,
) -> pd.DataFrame:
    """Treina `modelo` na série inteira e prevê os próximos h dias com intervalos de 80% e 95%.

    Se `residuos` não for informado, roda um backtest só desse modelo para obtê-los.
    Devolve DataFrame com data, previsto, lim_inf_80, lim_sup_80, lim_inf_95, lim_sup_95.
    """
    if residuos is None:
        residuos = backtest(serie, HORIZONTE_PADRAO, n_cortes, modelos=[modelo])["residuos"][modelo]
    previsto = prever(modelo, serie, h)
    datas = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    resultado = pd.DataFrame({"data": datas, "previsto": previsto, **intervalos(previsto, residuos)})
    return resultado[COLUNAS_PREVISAO]


@dataclass
class AjustePrevisao:
    """Resultado de fit_best: série de treino, modelo vencedor e o backtest que o escolheu."""

    serie: pd.Series
    modelo: str
    tabela: pd.DataFrame
    backtest: dict = field(repr=False)

    @property
    def residuos(self) -> np.ndarray:
        """Resíduos do backtest do modelo vencedor (base dos intervalos)."""
        return self.backtest["residuos"][self.modelo]

    def forecast(self, h: int | None = None) -> pd.DataFrame:
        """Previsão dos próximos h dias (padrão: o horizonte do backtest) com o modelo vencedor."""
        return forecast(self.serie, self.modelo, h or self.backtest["horizonte"], self.residuos)


def fit_best(
    serie: pd.Series,
    horizonte: int = HORIZONTE_PADRAO,
    n_cortes: int = N_CORTES_PADRAO,
    modelos=None,
) -> AjustePrevisao:
    """Roda o backtest de todos os modelos e escolhe o de menor WAPE médio."""
    resultado = backtest(serie, horizonte, n_cortes, modelos)
    tabela = resultado["tabela"]
    if tabela.empty or tabela["wape"].isna().all():
        raise ValueError("Nenhum modelo conseguiu completar o backtest.")
    vencedor = str(tabela.dropna(subset=["wape"]).iloc[0]["modelo"])
    return AjustePrevisao(serie=serie, modelo=vencedor, tabela=tabela, backtest=resultado)


# ---------- agregação (semana/mês/total) com intervalos ----------

N_SIMULACOES = 2000
BLOCO_SIMULACAO = 7


def matriz_residuos(residuos) -> np.ndarray:
    """Resíduos do backtest como matriz (cortes x passos), sem cortes que falharam.

    Aceita a matriz direto ou o DataFrame longo (corte, passo, residuo) salvo pelo pipeline.
    """
    if isinstance(residuos, pd.DataFrame):
        residuos = residuos.pivot(index="corte", columns="passo", values="residuo").sort_index().to_numpy()
    matriz = np.atleast_2d(np.asarray(residuos, dtype=float))
    return matriz[np.isfinite(matriz).all(axis=1)]


def residuos_longos(matriz) -> pd.DataFrame:
    """Matriz (cortes x passos) -> DataFrame corte, passo, residuo (formato aceito pelo Firestore)."""
    matriz = np.atleast_2d(np.asarray(matriz, dtype=float))
    cortes, passos = np.indices(matriz.shape)
    return pd.DataFrame({"corte": cortes.ravel() + 1, "passo": passos.ravel() + 1, "residuo": matriz.ravel()})


def simular_trajetorias(
    previsto, residuos, n_sim: int = N_SIMULACOES, bloco: int = BLOCO_SIMULACAO, seed: int = 42
) -> np.ndarray:
    """Trajetórias possíveis (n_sim x h) = previsto + erros reais do backtest, cortadas em zero.

    Block bootstrap: cada trajetória sorteia UM corte do backtest e, dentro dele, blocos
    de 7 dias seguidos. Assim mantém o padrão da semana e o viés de nível de cada corte,
    o que dá faixas realistas para totais (somar as faixas diárias exageraria a incerteza).
    """
    previsto = np.asarray(previsto, dtype=float)
    matriz = matriz_residuos(residuos)
    if matriz.size == 0:
        raise ValueError("Sem resíduos de backtest para simular.")
    rng = np.random.default_rng(seed)
    n_cortes, passos = matriz.shape
    bloco = min(bloco, passos)
    n_blocos = int(np.ceil(len(previsto) / bloco))
    cortes = rng.integers(0, n_cortes, size=n_sim)
    inicios = rng.integers(0, passos - bloco + 1, size=(n_sim, n_blocos))
    colunas = (inicios[:, :, None] + np.arange(bloco)).reshape(n_sim, -1)[:, : len(previsto)]
    erros = matriz[cortes[:, None], colunas]
    return np.maximum(previsto + erros, 0.0)


def _limites_das_somas(somas: np.ndarray) -> dict[str, np.ndarray]:
    limites = {}
    for nivel, (q_inf, q_sup) in QUANTIS_INTERVALO.items():
        limites[f"lim_inf_{nivel}"] = np.quantile(somas, q_inf, axis=0)
        limites[f"lim_sup_{nivel}"] = np.quantile(somas, q_sup, axis=0)
    return limites


def total_com_intervalo(previsao: pd.DataFrame, residuos, dias: int | None = None) -> dict:
    """Total previsto dos primeiros `dias` dias com faixas de 80%/95% (via simular_trajetorias)."""
    parte = previsao if dias is None else previsao.iloc[:dias]
    total = {"previsto": float(parte["previsto"].sum()), "dias": int(len(parte))}
    try:
        somas = simular_trajetorias(parte["previsto"].to_numpy(), residuos).sum(axis=1)
        total.update({chave: float(valor) for chave, valor in _limites_das_somas(somas).items()})
    except ValueError:
        total.update({f"lim_{lado}_{nivel}": float("nan") for nivel in QUANTIS_INTERVALO for lado in ("inf", "sup")})
    return total


def agregar_previsao(
    previsao: pd.DataFrame, residuos, granularidade: str = "dia", historico: pd.Series | None = None
) -> pd.DataFrame:
    """Previsão somada por dia/semana/mês com faixas de 80%/95%.

    - "dia": devolve as faixas diárias da própria previsão;
    - "semana"/"mes": faixas pelos quantis das somas das trajetórias simuladas;
    - no mês em que a previsão começa, soma o que já foi realizado (`historico`) para
      que o mês apareça inteiro (realizado + previsto).
    Colunas: periodo, previsto, lim_inf_80, lim_sup_80, lim_inf_95, lim_sup_95, n_dias, realizado, parcial.
    """
    origem = previsao["data"].iloc[0]
    chaves = periodos(previsao["data"], granularidade, origem)
    rotulos, posicoes = np.unique(chaves, return_inverse=True)
    rotulos = pd.DatetimeIndex(rotulos)

    if granularidade == "dia":
        resultado = previsao.rename(columns={"data": "periodo"}).reset_index(drop=True)
    else:
        trajetorias = simular_trajetorias(previsao["previsto"].to_numpy(), residuos)
        somas = np.zeros((trajetorias.shape[0], len(rotulos)))
        for i in range(len(rotulos)):
            somas[:, i] = trajetorias[:, posicoes == i].sum(axis=1)
        resultado = pd.DataFrame(
            {
                "periodo": rotulos,
                "previsto": np.bincount(posicoes, weights=previsao["previsto"].to_numpy(), minlength=len(rotulos)),
                **_limites_das_somas(somas),
            }
        )
    resultado["n_dias"] = np.bincount(posicoes, minlength=len(rotulos))
    resultado["realizado"] = 0.0

    if granularidade == "mes" and historico is not None and len(historico):
        ja_realizado = historico[periodos(historico.index, "mes") == rotulos[0]]
        if len(ja_realizado):
            colunas_valor = ["previsto", "lim_inf_80", "lim_sup_80", "lim_inf_95", "lim_sup_95"]
            resultado.loc[0, colunas_valor] += float(ja_realizado.sum())
            resultado.loc[0, "realizado"] = float(ja_realizado.sum())
            resultado.loc[0, "n_dias"] += len(ja_realizado)

    resultado["parcial"] = resultado["n_dias"].to_numpy() < dias_no_periodo(resultado["periodo"], granularidade)
    colunas = ["periodo", "previsto", "lim_inf_80", "lim_sup_80", "lim_inf_95", "lim_sup_95", "n_dias", "realizado", "parcial"]
    return resultado[colunas]


def agregar_historico(historico: pd.Series, granularidade: str = "dia", origem=None) -> pd.DataFrame:
    """Histórico somado por período, alinhado à previsão.

    O primeiro período, se incompleto (a série começa no meio da semana/mês), sai para
    não parecer uma queda de vendas. No modo "mes", o mês em que a previsão começa
    também sai (ele aparece na previsão como realizado + previsto).
    """
    resultado = agregar_serie(historico, granularidade, origem)
    if granularidade != "dia" and len(resultado) > 1 and resultado["parcial"].iloc[0]:
        resultado = resultado.iloc[1:].reset_index(drop=True)
    if granularidade == "mes" and origem is not None:
        mes_origem = periodos([origem], "mes")[0]
        resultado = resultado[resultado["periodo"] != mes_origem].reset_index(drop=True)
    return resultado


# ---------- modo simulação ----------

MODELOS_RESERVA = ("holt_winters", "baseline_sazonal_7d")


def simulate(serie: pd.Series, corte, modelo: str, horizonte: int = HORIZONTE_PADRAO) -> dict:
    """Simula a previsão feita no passado: treina só com dados até `corte` (inclusive)
    e compara com o real dos `horizonte` dias seguintes.

    As faixas vêm de um backtest (mesma função da Fase 2) rodado só com os dados até o corte.
    Se o modelo pedido não rodar com o treino curto, usa Holt-Winters ou o baseline.
    """
    corte = pd.Timestamp(corte).normalize()
    treino = serie.loc[:corte]
    teste = serie.loc[corte + pd.Timedelta(days=1) : corte + pd.Timedelta(days=horizonte)]
    if len(teste) < horizonte:
        raise ValueError(f"Depois do corte é preciso ter {horizonte} dias de dados reais; há {len(teste)}.")
    if len(treino) < MIN_DIAS_TREINO:
        raise ValueError(f"O treino até o corte tem {len(treino)} dias; o mínimo é {MIN_DIAS_TREINO}.")

    avisos, usado, previsto = [], None, None
    for candidato in dict.fromkeys([modelo, *MODELOS_RESERVA]):
        try:
            previsto = prever(candidato, treino, horizonte)
            usado = candidato
            break
        except Exception as erro:
            avisos.append(f"{candidato} não rodou com o treino até o corte ({erro}).")
    if usado is None:
        raise ValueError("Nenhum modelo conseguiu prever a partir dessa data de corte.")

    n_cortes = min(N_CORTES_PADRAO, (len(treino) - MIN_DIAS_TREINO) // horizonte)
    residuos = None
    if n_cortes >= 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            residuos = backtest(treino, horizonte, n_cortes, modelos=[usado])["residuos"][usado]
        if matriz_residuos(residuos).size == 0:
            residuos = None
    if residuos is None:
        n_cortes = 0
        avisos.append(
            "Faixas indisponíveis: o treino até o corte é curto demais para medir o erro "
            f"(precisa de {MIN_DIAS_TREINO + horizonte} dias)."
        )

    real = teste.to_numpy(dtype=float)
    limites = intervalos(previsto, residuos if residuos is not None else [])
    previsao = pd.DataFrame({"data": teste.index, "real": real, "previsto": previsto, **limites})
    return {
        "previsao": previsao,
        "corte": corte,
        "modelo": usado,
        "modelo_pedido": modelo,
        "n_dias_treino": int(len(treino)),
        "n_cortes_intervalo": int(n_cortes),
        "wape": metrics.wape(real, previsto),
        "mae": metrics.mae(real, previsto),
        "erro_total": metrics.erro_total(real, previsto),
        "total_real": float(real.sum()),
        "total_previsto": float(previsto.sum()),
        "cobertura_80": (
            metrics.cobertura(real, limites["lim_inf_80"], limites["lim_sup_80"]) if residuos is not None else float("nan")
        ),
        "avisos": avisos,
    }
