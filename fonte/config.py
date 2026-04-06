# Configuracoes do Aplicativo
import streamlit as st

APP_NAME = "Plataforma de Análise de Dados"

REQUIRED_FIREBASE_KEYS = (
    "apiKey",
    "authDomain",
    "projectId",
    "storageBucket",
    "messagingSenderId",
    "appId",
)


def get_firebase_config():
    """Retorna as configuracoes do Firebase a partir do `.streamlit/secrets.toml`."""
    firebase_secrets = st.secrets.get("firebase")
    if not firebase_secrets:
        raise RuntimeError(
            "Bloco [firebase] nao encontrado em `.streamlit/secrets.toml`."
        )

    missing_keys = [key for key in REQUIRED_FIREBASE_KEYS if key not in firebase_secrets]
    if missing_keys:
        raise RuntimeError(
            "Chaves obrigatorias do Firebase ausentes em `.streamlit/secrets.toml`: "
            + ", ".join(missing_keys)
        )

    return {
        "apiKey": firebase_secrets["apiKey"],
        "authDomain": firebase_secrets["authDomain"],
        "projectId": firebase_secrets["projectId"],
        "storageBucket": firebase_secrets["storageBucket"],
        "messagingSenderId": firebase_secrets["messagingSenderId"],
        "appId": firebase_secrets["appId"],
        "measurementId": firebase_secrets.get("measurementId", ""),
        "databaseURL": firebase_secrets.get("databaseURL", ""),
    }
