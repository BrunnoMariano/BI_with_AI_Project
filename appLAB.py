import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_openai import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from login_interface import show_login_page, show_logout_button
from auth import is_user_logged_in

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Plataforma de Dados", layout="wide")

# --- VERIFICAR SE O USUÁRIO ESTÁ LOGADO ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Se não estiver logado, mostrar página de login
if not st.session_state['logged_in']:
    show_login_page()
else:
    # Se estiver logado, mostrar a aplicação principal
    st.title("Análise Inteligente de Dados 🚀")

    # Mostrar botão de logout
    show_logout_button()

    # --- Barra Lateral para Inserir Chaves ---
    with st.sidebar:
        st.header("🔑 Configurações")
        openai_api_key = st.text_input("Sua chave da API OpenAI", type="password")
        modelo_selecionado = st.selectbox(
            "Escolha o modelo de IA:",
            ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4")
        )

        st.subheader("⚙ Conexão com Banco de Dados")
        st.info(
            "O código já está conectado a um banco PostgreSQL público de exemplo. Substitua abaixo pelos dados do seu banco.")

        db_user_exemplo = "user-name"
        db_password_exemplo = "password"
        db_host_exemplo = "host-name"
        db_port_exemplo = "5432"
        db_name_exemplo = "database-name"

        db_user = st.text_input("Usuário do Banco", db_user_exemplo)
        db_password = st.text_input("Senha do Banco", db_password_exemplo, type="password")
        db_host = st.text_input("Host do Banco", db_host_exemplo)
        db_port = st.text_input("Porta do Banco", db_port_exemplo)
        db_name = st.text_input("Nome do Banco", db_name_exemplo)

    # --- CRIAÇÃO DAS ABAS ---
    tab1, tab2 = st.tabs(["💬 Conversa com a IA", "📊 Dashboard Interativo"])

    # --- CONTEÚDO DA ABA 1: CHAT ---
    with tab1:
        st.header("Fale com seus Dados")
        if not openai_api_key:
            st.warning("Por favor, insira sua chave da API OpenAI na barra lateral para ativar o chat.")
        else:
            try:
                uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                db = SQLDatabase.from_uri(uri)
                llm = ChatOpenAI(
                    temperature=0,
                    api_key=openai_api_key,
                    model_name=modelo_selecionado,
                    verbose=True
                )
                db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

                if 'history' not in st.session_state:
                    st.session_state.history = []

                for message in st.session_state.history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Pergunte algo sobre os dados... Ex: 'Quantos filmes existem?'"):
                    st.session_state.history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Analisando os dados..."):
                            try:
                                response = db_chain.run(prompt)
                                st.markdown(response)
                                st.session_state.history.append({"role": "assistant", "content": response})
                            except Exception as e:
                                error_message = f"Ocorreu um erro ao consultar a IA: {e}"
                                st.error(error_message)
                                st.session_state.history.append({"role": "assistant", "content": error_message})

            except Exception as e:
                st.error(f"Falha ao conectar ao banco de dados na aba de Chat. Verifique as credenciais. Erro: {e}")

    # --- CONTEÚDO DA ABA 2: DASHBOARD ---
    with tab2:
        st.header("Dashboard de Análise")
        st.write("Este dashboard usa dados de exemplo para demonstração. Adapte para carregar dados do seu banco.")

        df_dashboard = pd.DataFrame({
            "Região": ["Sudeste", "Sudeste", "Nordeste", "Nordeste", "Sul", "Sudeste"],
            "Categoria": ["Eletrônicos", "Vestuário", "Eletrônicos", "Alimentos", "Vestuário", "Alimentos"],
            "Vendas": [5200, 4300, 3100, 2500, 3800, 4100],
            "Ano": [2024, 2025, 2024, 2025, 2025, 2025]
        })

        st.sidebar.header("Filtros do Dashboard")
        ano_selecionado = st.sidebar.slider("Selecione o Ano:", 2024, 2025, 2025)
        df_filtrado = df_dashboard[df_dashboard["Ano"] == ano_selecionado]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Vendas por Categoria")
            fig_bar = px.bar(df_filtrado, x="Categoria", y="Vendas", color="Categoria")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("Vendas por Região")
            fig_pie = px.pie(df_filtrado, names="Região", values="Vendas")
            st.plotly_chart(fig_pie, use_container_width=True)
