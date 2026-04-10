import streamlit as st
from auth import get_current_user, login_user, logout_user, register_user, resend_verification_email
from user_data import restore_user_session
from ui_styles import render_app_hero, render_section_intro


def show_login_page():
    """Mostra a página de login/cadastro"""
    render_app_hero(
        "Plataforma de Análise de Dados",
    )

    col_left, col_center, col_right = st.columns([1.1, 1.6, 1.1])
    with col_center:
        render_section_intro(
            "Autenticação",
            "",
            "Faça login ou crie sua conta.",
        )

        tab1, tab2 = st.tabs(["Login", "Cadastro"])

        with tab1:
            render_section_intro(
                "Login",
                "Fazer login",
                "",
            )
            with st.form("login_form"):
                email = st.text_input("E-mail")
                password = st.text_input("Senha", type="password")
                submit_button = st.form_submit_button("Entrar")

                if submit_button:
                    if email and password:
                        response = login_user(email, password)
                        if response and response.get("idToken"):
                            if response.get("emailVerified"):
                                st.success("Login realizado com sucesso!")
                                restore_user_session(response)
                                st.rerun()
                            else:
                                st.warning("Seu e-mail ainda não foi verificado. Verifique sua caixa de entrada antes de entrar.")
                        else:
                            st.error("E-mail ou senha incorretos!")
                    else:
                        st.error("Por favor, preencha todos os campos!")

            if email and password and st.button("Reenviar e-mail de verificação", key="resend_verification"):
                if resend_verification_email(email, password):
                    st.success("E-mail de verificação reenviado.")

        with tab2:
            render_section_intro(
                "Cadastro",
                "Criar conta",
                "",
            )
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
                                if response and response.get("idToken"):
                                    st.success("Conta criada com sucesso! Enviamos um e-mail de verificação para você.")
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
        if user and user.get("email"):
            st.markdown(
                f"<p class='ui-section-label' style='margin-bottom:0.35rem;'>Sessão</p><p style='margin:0 0 0.8rem;color:#d7deea;font-weight:600;'>{user['email']}</p>",
                unsafe_allow_html=True,
            )
            if st.button("Sair"):
                logout_user()
                st.rerun()
