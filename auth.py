import streamlit as st
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY

# Inicializar o cliente Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def login_user(email: str, password: str):
    """Função para fazer login do usuário"""
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return response
    except Exception as e:
        return None

def register_user(email: str, password: str):
    """Função para registrar novo usuário"""
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        return response
    except Exception as e:
        return None

def logout_user():
    """Função para fazer logout do usuário"""
    try:
        supabase.auth.sign_out()
        return True
    except Exception as e:
        return False

def get_current_user():
    """Função para obter o usuário atual"""
    try:
        user = supabase.auth.get_user()
        return user
    except Exception as e:
        return None

def is_user_logged_in():
    """Verifica se o usuário está logado"""
    user = get_current_user()
    return user is not None and user.user is not None