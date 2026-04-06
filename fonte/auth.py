import streamlit as st
from firebase_service import (
    FirebaseError,
    lookup_user_account,
    register_user_account,
    send_email_verification,
    sign_in_user,
)

def login_user(email: str, password: str):
    """Faz login do usuario via Firebase Authentication REST."""
    try:
        user = sign_in_user(email, password)
        profile = lookup_user_account(user["idToken"])
        user.update(profile)
        return user
    except (FirebaseError, RuntimeError) as e:
        st.error(f"Erro ao fazer login: {e}")
        return None

def register_user(email: str, password: str):
    """Registra um novo usuario via Firebase Authentication REST."""
    try:
        user = register_user_account(email, password)
        send_email_verification(user["idToken"])
        profile = lookup_user_account(user["idToken"])
        user.update(profile)
        return user
    except (FirebaseError, RuntimeError) as e:
        st.error(f"Erro ao registrar: {e}")
        return None


def resend_verification_email(email: str, password: str):
    """Reenvia o e-mail de verificacao para o usuario informado."""
    try:
        user = sign_in_user(email, password)
        send_email_verification(user["idToken"])
        return True
    except (FirebaseError, RuntimeError) as e:
        st.error(f"Erro ao reenviar e-mail de verificacao: {e}")
        return False

def logout_user():
    """Limpa o estado da sessao do usuario."""
    keys_to_clear = [
        "user",
        "logged_in",
        "history",
        "charts",
        "db_creds",
        "connected_uri",
        "schema_text",
        "schema_dict",
        "df_data",
        "user_files",
        "selected_file_id",
        "loaded_file_id",
        "last_uploaded_file_signature",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.session_state["logged_in"] = False

def get_current_user():
    """Retorna o usuario logado no `st.session_state`."""
    user = st.session_state.get("user")
    if not user:
        return None
    if user.get("email"):
        return user
    id_token = user.get("idToken")
    if not id_token:
        return None
    try:
        profile = lookup_user_account(id_token)
    except FirebaseError:
        return user
    user.update(profile)
    st.session_state["user"] = user
    return user

def is_user_logged_in():
    """Verifica se o usuario esta autenticado."""
    user = st.session_state.get("user", {})
    return bool(st.session_state.get("logged_in", False) and user.get("idToken"))
