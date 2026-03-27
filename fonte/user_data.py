# user_data.py
import os
import json
import streamlit as st

def encrypt_credentials(credentials_dict: dict) -> str:
    """Retorna o dicionário de credenciais como string JSON (sem criptografia)."""
    return json.dumps(credentials_dict)

def decrypt_credentials(encrypted_string: str) -> dict:
    """Converte a string JSON de volta para um dicionário."""
    if not encrypted_string: return None
    try:
        return json.loads(encrypted_string)
    except:
        return None

# --- Configurações Locais ---

def get_user_config(user_id: str):
    """Busca a configuração (em modo local, retorna vazio por enquanto)."""
    return None

def save_db_config(user_id: str, credentials_dict: dict):
    """Simula o salvamento de credenciais localmente."""
    return True
