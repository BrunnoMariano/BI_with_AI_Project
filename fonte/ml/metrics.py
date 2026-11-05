"""Métricas de erro para avaliar previsões (funções puras sobre arrays)."""

from __future__ import annotations

import numpy as np


def _arrays(real, previsto) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(real, dtype=float)
    p = np.asarray(previsto, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"Tamanhos diferentes: real {y.shape} x previsto {p.shape}.")
    return y, p


def mae(real, previsto) -> float:
    """Erro absoluto médio."""
    y, p = _arrays(real, previsto)
    return float(np.mean(np.abs(y - p)))


def rmse(real, previsto) -> float:
    """Raiz do erro quadrático médio (pune mais os erros grandes)."""
    y, p = _arrays(real, previsto)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wape(real, previsto) -> float:
    """Erro absoluto ponderado: soma |erro| / soma |real|. Métrica principal (0,5 = 50%)."""
    y, p = _arrays(real, previsto)
    total = np.sum(np.abs(y))
    if total == 0:
        return float("nan")
    return float(np.sum(np.abs(y - p)) / total)


def mape_seguro(real, previsto) -> float:
    """MAPE só nos dias com valor real diferente de zero (evita divisão por zero)."""
    y, p = _arrays(real, previsto)
    nao_zero = y != 0
    if not nao_zero.any():
        return float("nan")
    return float(np.mean(np.abs((y[nao_zero] - p[nao_zero]) / y[nao_zero])))


def erro_total(real, previsto) -> float:
    """Erro relativo no total do período: (soma previsto - soma real) / soma real."""
    y, p = _arrays(real, previsto)
    total = np.sum(y)
    if total == 0:
        return float("nan")
    return float((np.sum(p) - total) / total)


def cobertura(real, lim_inf, lim_sup) -> float:
    """Fração dos valores reais que caíram dentro do intervalo [lim_inf, lim_sup]."""
    y = np.asarray(real, dtype=float)
    lo = np.asarray(lim_inf, dtype=float)
    hi = np.asarray(lim_sup, dtype=float)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((y >= lo) & (y <= hi)))
