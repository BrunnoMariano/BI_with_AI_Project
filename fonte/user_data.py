# user_data.py
import os
import json
import streamlit as st
from cryptography.fernet import Fernet

# --- Criptografia ---
# Carrega a chave de criptografia da variável de ambiente
try:
    ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY", "chave_padrao_para_dev_apenas_12345=").encode()
    cipher_suite = Fernet(ENCRYPTION_KEY)
except Exception as e:
    st.error("ERRO: Chave de criptografia inválida ou não encontrada.")
    cipher_suite = None

def encrypt_credentials(credentials_dict: dict) -> str:
    """Converte o dicionário de credenciais para JSON e o criptografa."""
    if not cipher_suite: return None
    credentials_json = json.dumps(credentials_dict)
    encrypted_data = cipher_suite.encrypt(credentials_json.encode())
    return encrypted_data.decode()

def decrypt_credentials(encrypted_string: str) -> dict:
    """Descriptografa a string e a converte de volta para um dicionário."""
    if not cipher_suite or not encrypted_string: return None
    decrypted_data = cipher_suite.decrypt(encrypted_string.encode())
    return json.loads(decrypted_data.decode())

# --- Configurações Locais ---

def get_user_config(user_id: str):
    """Busca a configuração (em modo local, retorna vazio por enquanto)."""
    return None

def save_db_config(user_id: str, credentials_dict: dict):
    """Simula o salvamento de credenciais localmente."""
    return True
