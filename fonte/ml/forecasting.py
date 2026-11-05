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
from ml.features import calendar_features

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
