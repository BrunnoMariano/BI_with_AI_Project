# user_data.py
import os
import json
import streamlit as st
from cryptography.fernet import Fernet
from auth import supabase # Reutilizamos o cliente supabase de auth.py

# --- Criptografia ---
# Carrega a chave de criptografia da variável de ambiente que configuramos
try:
    ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY").encode()
    cipher_suite = Fernet(ENCRYPTION_KEY)
except Exception as e:
    # Este erro só deve aparecer se a variável de ambiente não estiver configurada
    st.error("ERRO CRÍTICO: Chave de criptografia não encontrada. O salvamento de credenciais está desativado.")
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

# --- Interação com a Tabela user_profiles ---

def get_user_config(user_id: str):
    """Busca a configuração salva (fonte, credenciais, etc.) para um usuário."""
    try:
        response = supabase.table('user_profiles').select('*').eq('id', user_id).single().execute()
        return response.data
    except Exception:
        return None

def save_db_config(user_id: str, credentials_dict: dict):
    """Criptografa e salva as credenciais do banco de dados para um usuário."""
    encrypted_creds = encrypt_credentials(credentials_dict)
    if encrypted_creds:
        try:
            supabase.table('user_profiles').update({
                'data_source_type': 'database',
                'db_credentials': encrypted_creds,
                'file_path': None # Limpa o caminho do arquivo se estiver mudando para DB
            }).eq('id', user_id).execute()
            return True
        except Exception as e:
            st.error(f"Erro ao salvar configuração do banco de dados: {e}")
            return False
    return False

# A lógica de salvar arquivos seria adicionada aqui no futuro