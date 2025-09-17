# appLAB.py
import streamlit as st
import pandas as pd
import plotly.express as px
import re
from sqlalchemy import create_engine, text
from langchain.sql_database import SQLDatabase
from login_interface import show_login_page, show_logout_button
from auth import is_user_logged_in
from db_utils import introspect_schema
from openai import OpenAI
import pandasql as ps  # para rodar SQL em DataFrames

st.set_page_config(page_title="Plataforma de Dados", layout="wide")


def build_history_text(history, max_turns=6):
    if not history:
        return ""
    max_messages = max_turns * 2
    recent = history[-max_messages:]
    lines = []
    for msg in recent:
        role = "Usuário" if msg["role"] == "user" else "Assistente"
        content = msg["content"].replace("```", "'`'`")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def extract_sql(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(?i)(\bselect\b|\bwith\b|\binsert\b|\bupdate\b|\bdelete\b)(.*)", text, re.S)
    if m2:
        return (m2.group(1) + m2.group(2)).strip()
    return text.strip()


def generate_sql_via_openai(client, model_name, system_prompt, schema_text, history_text, question):
    messages = [
        {
            "role": "system",
            "content": system_prompt + "\n\n---\nEsquema do conjunto de dados:\n" + schema_text,
        },
        {
            "role": "user",
            "content": (
                "Histórico:\n"
                + (history_text or "(sem histórico)\n")
                + "\nPergunta:\n"
                + question
                + "\n\nINSTRUÇÕES:\n"
                "- Retorne apenas a SQL.\n"
                "- Se não precisar de SQL, retorne: NO_SQL\n"
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def generate_humanized_answer(client, model_name, system_prompt, schema_text, history_text, question, sql_text, results_preview, show_sql=True):
    sql_section = f"\n\nSQL executado:\n```sql\n{sql_text}\n```" if show_sql else ""
    instructions = "- Explique em linguagem simples.\n"
    if show_sql:
        instructions += "- Mostre a SQL em bloco de código.\n"
    instructions += "- Sugira gráfico se fizer sentido.\n"

    messages = [
        {
            "role": "system",
            "content": system_prompt + "\n\n---\nEsquema do conjunto de dados:\n" + schema_text,
        },
        {
            "role": "user",
            "content": (
                "Histórico:\n"
                + (history_text or "(sem histórico)\n")
                + "\nPergunta:\n"
                + question
                + sql_section
                + "\n\nResultados:\n"
                + results_preview
                + "\n\nINSTRUÇÕES:\n"
                + instructions
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def generate_answer_no_sql(client, model_name, system_prompt, schema_text, history_text, question):
    messages = [
        {
            "role": "system",
            "content": system_prompt + "\n\n---\nEsquema do conjunto de dados:\n" + schema_text,
        },
        {
            "role": "user",
            "content": (
                "Histórico:\n"
                + (history_text or "(sem histórico)\n")
                + "\nPergunta:\n"
                + question
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


def describe_dataframe_schema(df: pd.DataFrame) -> str:
    """Gera descrição textual das colunas do DataFrame"""
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        # Mostramos colunas entre aspas duplas
        lines.append(f'- "{col}" ({dtype})')
    return "\n".join(lines)


# --- Login ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    show_login_page()
else:
    st.title("Análise Inteligente de Dados 🚀")
    show_logout_button()

    with st.sidebar:
        st.header("🔑 Configurações")
        openai_api_key = st.text_input("Sua chave da API OpenAI", type="password")
        modelo_selecionado = st.selectbox(
            "Escolha o modelo de IA:",
            ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4")
        )

        st.subheader("📂 Fonte de Dados")
        fonte_dados = st.selectbox("Selecione a fonte:", ["Banco de Dados", "Arquivo CSV/Excel"])

        db_user = db_password = db_host = db_port = db_name = None
        uploaded_file = None

        if fonte_dados == "Banco de Dados":
            db_user = st.text_input("Usuário do Banco", "user-name")
            db_password = st.text_input("Senha do Banco", "password", type="password")
            db_host = st.text_input("Host do Banco", "host-name")
            db_port = st.text_input("Porta do Banco", "5432")
            db_name = st.text_input("Nome do Banco", "database-name")
        else:
            uploaded_file = st.file_uploader("Faça upload de um arquivo CSV ou Excel", type=["csv", "xlsx"])

    # 3 abas
    tab1, tab2, tab3 = st.tabs(["💬 Conversa com a IA", "📊 Dashboard Interativo", "📝 Histórico de Conversas"])

    # --- Chat ---
    with tab1:
        st.header("Fale com seus Dados")
        if not openai_api_key:
            st.warning("Por favor, insira sua chave da API OpenAI na barra lateral para ativar o chat.")
        else:
            try:
                schema_text = "(sem esquema)"
                df_data = None
                db_mode = (fonte_dados == "Banco de Dados")

                if db_mode:
                    uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                    db = SQLDatabase.from_uri(uri)

                    need_introspect = (
                        'connected_uri' not in st.session_state
                        or st.session_state['connected_uri'] != uri
                    )
                    if need_introspect:
                        with st.spinner("Introspectando o esquema do banco..."):
                            schema_text, schema_dict = introspect_schema(uri)
                            st.session_state['schema_text'] = schema_text
                            st.session_state['schema_dict'] = schema_dict
                            st.session_state['connected_uri'] = uri

                    schema_text = st.session_state.get('schema_text', "(sem esquema)")

                    system_prompt = (
                        "Você é um assistente de BI que analisa bancos de dados relacionais "
                        "e explica resultados em linguagem simples. "
                        "Use SQL sempre que necessário e humanize a resposta."
                    )

                else:  # CSV/Excel
                    if uploaded_file is not None:
                        if uploaded_file.name.endswith(".csv"):
                            df_data = pd.read_csv(uploaded_file)
                        else:
                            df_data = pd.read_excel(uploaded_file)

                        st.session_state['df_data'] = df_data
                        schema_text = describe_dataframe_schema(df_data)
                        st.session_state['schema_text_file'] = schema_text

                        system_prompt = (
                            "Você é um assistente de BI que analisa arquivos tabulares (CSV/Excel). "
                            "O arquivo foi carregado em uma tabela chamada 'df'. "
                            "Sempre escreva suas queries SQL usando a tabela 'df'. "
                            "Use exatamente os nomes das colunas como aparecem no esquema, incluindo espaços, acentuação e maiúsculas/minúsculas. "
                            "Explique resultados em linguagem simples e sugira gráficos quando fizer sentido."
                        )
                    else:
                        st.info("Faça upload de um arquivo CSV/Excel para começar.")

                if 'history' not in st.session_state:
                    st.session_state.history = []

                for message in st.session_state.history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Pergunte algo sobre os dados..."):
                    st.session_state.history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Analisando os dados..."):
                            try:
                                client = OpenAI(api_key=openai_api_key)
                                history_text = build_history_text(st.session_state.history, max_turns=6)

                                sql_candidate = generate_sql_via_openai(
                                    client, modelo_selecionado,
                                    system_prompt, schema_text, history_text, prompt
                                )

                                if sql_candidate.strip().upper() == "NO_SQL":
                                    final_text = generate_answer_no_sql(
                                        client, modelo_selecionado,
                                        system_prompt, schema_text, history_text, prompt
                                    )
                                    st.markdown(final_text)
                                    st.session_state.history.append({"role": "assistant", "content": final_text})
                                else:
                                    sql_text = extract_sql(sql_candidate)
                                    try:
                                        if db_mode:
                                            engine = create_engine(uri)
                                            with engine.connect() as conn:
                                                result = conn.execute(text(sql_text))
                                                rows = result.fetchall()
                                                if rows:
                                                    df_result = pd.DataFrame(rows, columns=result.keys())
                                                else:
                                                    df_result = pd.DataFrame()
                                        else:
                                            if 'df_data' not in st.session_state:
                                                raise ValueError("Nenhum arquivo foi carregado.")
                                            df_result = ps.sqldf(sql_text, {"df": st.session_state['df_data']})
                                    except Exception as e:
                                        error_message = f"Erro ao executar a query: {e}\nQuery gerada:\n{sql_text}"
                                        st.error(error_message)
                                        st.session_state.history.append({"role": "assistant", "content": error_message})
                                    else:
                                        if df_result.empty:
                                            results_preview = "(consulta retornou 0 linhas)"
                                        else:
                                            preview_df = df_result.head(10).copy()
                                            try:
                                                results_preview = preview_df.to_markdown(index=False)
                                            except Exception:
                                                results_preview = preview_df.to_csv(index=False)

                                        final_text = generate_humanized_answer(
                                            client, modelo_selecionado,
                                            system_prompt, schema_text, history_text,
                                            prompt, sql_text, results_preview,
                                            show_sql=db_mode  # só mostra SQL se for banco
                                        )
                                        st.markdown(final_text)
                                        if not df_result.empty:
                                            st.subheader("Preview dos dados")
                                            st.dataframe(df_result.head(100))
                                        st.session_state.history.append({"role": "assistant", "content": final_text})
                            except Exception as e:
                                error_message = f"Ocorreu um erro: {e}"
                                st.error(error_message)
                                st.session_state.history.append({"role": "assistant", "content": error_message})

            except Exception as e:
                st.error(f"Falha ao conectar ou carregar os dados: {e}")

    # --- Dashboard exemplo ---
    with tab2:
        st.header("Dashboard de Análise")
        st.write("Este dashboard usa dados de exemplo.")

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

    # --- Histórico de Conversas ---
    with tab3:
        st.header("Histórico de Conversas 📝")

        if 'history' not in st.session_state or not st.session_state.history:
            st.info("Nenhuma conversa registrada ainda.")
        else:
            for i, msg in enumerate(st.session_state.history):
                role = "👤 Usuário" if msg["role"] == "user" else "🤖 Assistente"
                with st.expander(f"{role} - Mensagem {i+1}"):
                    st.markdown(msg["content"])

            if st.button("🗑️ Limpar Histórico"):
                st.session_state.history = []
                st.success("Histórico apagado!")
