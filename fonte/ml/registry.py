"""Salvar e carregar resultados do pipeline preditivo.

- Local: arquivo joblib em models/{source_key}.joblib (fora do Git pelo *.joblib).
- Nuvem (opcional): Firestore em users/{uid}/models/{source_key}, reutilizando
  firebase_service.set_document/get_document. Só é usado se uid e id_token forem
  informados; o import do Firebase é feito dentro da função para que este módulo
  continue testável offline.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PASTA_MODELOS = Path(__file__).resolve().parents[2] / "models"

# Chaves do resultado que guardam {alvo: DataFrame}.
CHAVES_TABELAS = ("previsoes", "modelos", "historico", "residuos", "backtest")


def chave_segura(source_key: str) -> str:
    """Converte a source_key num nome válido para arquivo e para ID de documento do Firestore."""
    texto = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source_key)).strip("._")
    if not texto:
        raise ValueError("source_key vazia.")
    return texto[:200]


def para_json(valor):
    """Converte o resultado em tipos simples (dict/list/str/int/float/None) aceitos pelo Firestore.

    DataFrame vira lista de registros; datas viram texto ISO; NaN vira None.
    """
    if isinstance(valor, pd.DataFrame):
        return [para_json(registro) for registro in valor.to_dict(orient="records")]
    if isinstance(valor, pd.Series):
        return para_json(valor.tolist())
    if isinstance(valor, dict):
        return {str(chave): para_json(item) for chave, item in valor.items()}
    if isinstance(valor, (list, tuple, np.ndarray)):
        return [para_json(item) for item in valor]
    if isinstance(valor, pd.Timestamp):
        return None if pd.isna(valor) else valor.isoformat()
    if isinstance(valor, (datetime, date)):
        return valor.isoformat()
    if isinstance(valor, (bool, np.bool_)):
        return bool(valor)
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        return None if math.isnan(valor) or math.isinf(valor) else float(valor)
    if valor is None or valor is pd.NaT or valor is pd.NA:
        return None
    return valor


def de_json(payload: dict) -> dict:
    """Operação inversa de para_json para o resultado do pipeline: listas de registros voltam a ser DataFrames."""
    resultado = {chave: valor for chave, valor in payload.items() if not str(chave).startswith("_")}
    for chave in CHAVES_TABELAS:
        tabelas = resultado.get(chave)
        if not isinstance(tabelas, dict):
            continue
        convertidas = {}
        for alvo, registros in tabelas.items():
            df = pd.DataFrame(registros or [])
            if "data" in df:
                df["data"] = pd.to_datetime(df["data"])
            convertidas[alvo] = df
        resultado[chave] = convertidas
    return resultado


def _caminho_local(source_key: str, pasta=None) -> Path:
    return Path(pasta or PASTA_MODELOS) / f"{chave_segura(source_key)}.joblib"


def _caminho_firestore(uid: str, source_key: str) -> str:
    return f"users/{uid}/models/{chave_segura(source_key)}"


def salvar_resultado(
    resultado: dict,
    source_key: str,
    uid: str | None = None,
    id_token: str | None = None,
    pasta=None,
) -> dict:
    """Salva o resultado em joblib local e, se uid e id_token vierem, também no Firestore.

    Devolve {"local": caminho do arquivo, "firestore": caminho do documento ou None}.
    """
    caminho = _caminho_local(source_key, pasta)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(resultado, caminho)

    documento = None
    if uid and id_token:
        from firebase_service import set_document

        documento = _caminho_firestore(uid, source_key)
        set_document(documento, para_json(resultado), id_token)
    return {"local": str(caminho), "firestore": documento}


def carregar_resultado(
    source_key: str,
    uid: str | None = None,
    id_token: str | None = None,
    pasta=None,
) -> dict | None:
    """Carrega o resultado salvo: primeiro o joblib local, senão o Firestore. None se não existir."""
    caminho = _caminho_local(source_key, pasta)
    if caminho.exists():
        return joblib.load(caminho)

    if uid and id_token:
        from firebase_service import get_document

        documento = get_document(_caminho_firestore(uid, source_key), id_token)
        if documento:
            return de_json(documento)
    return None
