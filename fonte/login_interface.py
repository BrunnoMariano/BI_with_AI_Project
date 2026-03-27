import streamlit as st
from auth import login_user, register_user, logout_user, is_user_logged_in, get_current_user


def show_login_page():
    """Mostra a página de login/cadastro"""
    st.title("🔐 Login - Plataforma de Análise de Dados")

    # Criar abas para Login e Cadastro
    tab1, tab2 = st.tabs(["Login", "Cadastro"])

    with tab1:
        st.header("Fazer Login")
        with st.form("login_form"):
            email = st.text_input("E-mail")
            password = st.text_input("Senha", type="password")
            submit_button = st.form_submit_button("Entrar")

            if submit_button:
                if email and password:
                    response = login_user(email, password)
                    if response and response.user:
                        st.success("Login realizado com sucesso!")
                        st.session_state['user'] = response.user
                        st.session_state['logged_in'] = True
                        st.rerun()
                    else:
                        st.error("E-mail ou senha incorretos!")
                else:
                    st.error("Por favor, preencha todos os campos!")

    with tab2:
        st.header("Criar Conta")
        with st.form("register_form"):
            new_email = st.text_input("E-mail", key="reg_email")
            new_password = st.text_input("Senha", type="password", key="reg_password")
            confirm_password = st.text_input("Confirmar Senha", type="password")
            register_button = st.form_submit_button("Cadastrar")

            if register_button:
                if new_email and new_password and confirm_password:
                    if new_password == confirm_password:
                        if len(new_password) >= 6:
                            response = register_user(new_email, new_password)
                            if response and response.user:
                                st.success("Conta criada com sucesso! Verifique seu e-mail para confirmar.")
                            else:
                                st.error("Erro ao criar conta. Tente novamente.")
                        else:
                            st.error("A senha deve ter pelo menos 6 caracteres!")
                    else:
                        st.error("As senhas não coincidem!")
                else:
                    st.error("Por favor, preencha todos os campos!")


def show_logout_button():
    """Mostra o botão de logout na barra lateral"""
    with st.sidebar:
        user = get_current_user()
        if user and user.user:
            st.write(f"👤 Logado como: {user.user.email}")
            if st.button("Sair"):
                logout_user()
                st.session_state['logged_in'] = False
                if 'user' in st.session_state:
                    del st.session_state['user']
                st.rerun()
