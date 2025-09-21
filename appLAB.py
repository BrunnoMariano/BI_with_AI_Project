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


# ----------------------
# Utilitários
# ----------------------
def build_history_text(history, max_turns=6):
    """Converte histórico em texto compacto para enviar à IA."""
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
    """Extrai SQL de um texto (procura por bloco ```sql ... ``` ou SELECT)."""
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
    """Pede à OpenAI para retornar apenas SQL ou NO_SQL."""
    messages = [
        {
            "role": "system",
            "content": system_prompt + "\n\n---\nEsquema:\n" + schema_text,
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


def generate_chart_instructions(client, model_name, schema_text, question, df_preview=None, db_mode=True):
    """Pede instruções de gráfico para a IA."""
    preview_text = ""
    if df_preview is not None:
        try:
            preview_text = df_preview.head(5).to_markdown(index=False)
        except Exception:
            preview_text = df_preview.head(5).to_csv(index=False)

    system_content = (
        "Você é um assistente que cria dashboards.\n"
        "IMPORTANTE:\n"
        "- Use exatamente os nomes das colunas e tabelas do esquema.\n"
        "- Nunca invente nomes de tabelas.\n"
        "- Se a fonte for CSV/Excel, SEMPRE use a tabela chamada df.\n"
        "- Se a fonte for banco, use apenas tabelas do esquema introspectado.\n"
        "- Coloque nomes de colunas entre aspas duplas na SQL.\n"
        "Retorne no formato:\n"
        "SQL: <query>\n"
        "Tipo: <bar|line|pie>\n"
        "X: <coluna X>\n"
        "Y: <coluna Y>\n"
        "Título: <título>\n"
    )

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": "Esquema:\n" + schema_text + "\nPreview:\n" + preview_text + "\n\nPedido:\n" + question,
        },
    ]
    resp = client.chat.completions.create(
        model=model_name, messages=messages, temperature=0.0, max_tokens=500
    )
    return resp.choices[0].message.content.strip()


def parse_chart_instructions(text):
    """Extrai SQL, tipo, eixos e título da resposta da IA."""
    chart = {"sql": None, "type": "bar", "x": None, "y": None, "title": "Gráfico"}
    for line in text.splitlines():
        if line.strip().lower().startswith("sql:"):
            chart["sql"] = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("tipo:"):
            chart["type"] = line.split(":", 1)[1].strip().lower()
        elif line.strip().lower().startswith("x:"):
            chart["x"] = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("y:"):
            chart["y"] = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("título:") or line.strip().lower().startswith("titulo:"):
            chart["title"] = line.split(":", 1)[1].strip()
    return chart


def _strip_quotes(val: str) -> str:
    """Remove aspas extras de nomes de colunas."""
    if not isinstance(val, str):
        return val
    return val.strip().strip('"').strip("'")


def render_chart(chart_config):
    """Renderiza gráfico Plotly."""
    df = chart_config.get("df", pd.DataFrame())
    if df is None or df.empty:
        st.warning("Sem dados para exibir neste gráfico.")
        return

    tipo = chart_config.get("type", "bar")
    x = _strip_quotes(chart_config.get("x")) if chart_config.get("x") else None
    y = _strip_quotes(chart_config.get("y")) if chart_config.get("y") else None

    if x and x not in df.columns:
        st.error(f"Eixo X inválido '{x}'. Colunas: {list(df.columns)}")
        return
    if y and y not in df.columns and tipo != "pie":
        st.error(f"Eixo Y inválido '{y}'. Colunas: {list(df.columns)}")
        return

    try:
        if tipo == "bar":
            fig = px.bar(df, x=x, y=y, title=chart_config.get("title", "Gráfico"))
        elif tipo == "line":
            fig = px.line(df, x=x, y=y, title=chart_config.get("title", "Gráfico"))
        elif tipo == "pie":
            fig = px.pie(df, names=x, values=y, title=chart_config.get("title", "Gráfico"))
        else:
            st.warning(f"Tipo de gráfico '{tipo}' não suportado.")
            return
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao gerar gráfico: {e}")


def validate_sql_tables(sql_text, db_mode, schema_dict=None):
    """Valida se as tabelas da SQL existem."""
    if not sql_text:
        return False, "Consulta SQL vazia."
    sql_lower = sql_text.lower()
    if db_mode:
        if not schema_dict:
            return True, None
        available = list(schema_dict.keys())
        found = []
        for t in available:
            pattern = r"\b" + re.escape(t.lower()) + r"\b"
            if re.search(pattern, sql_lower):
                found.append(t)
        if not found:
            return False, f"Tabela não encontrada. Disponíveis: {', '.join(available)}"
        return True, None
    else:
        if re.search(r"\bfrom\s+df\b", sql_lower) or re.search(r"\bfrom\s+\"df\"\b", sql_lower):
            return True, None
        return False, "Para CSV/Excel use a tabela chamada 'df'."


def generate_answer_no_sql(client, model_name, system_prompt, schema_text, history_text, question):
    """Resposta quando não precisa de SQL."""
    messages = [
        {"role": "system", "content": system_prompt + "\n\n---\nEsquema:\n" + schema_text},
        {"role": "user", "content": "Histórico:\n" + (history_text or "(sem histórico)") + "\nPergunta:\n" + question},
    ]
    resp = client.chat.completions.create(
        model=model_name, messages=messages, temperature=0.0, max_tokens=500
    )
    return resp.choices[0].message.content.strip()


def generate_humanized_answer(client, model_name, question, df_result, sql_text=None, db_mode=True):
    """Resposta humanizada baseada em dados."""
    preview_text = ""
    if df_result is not None and not df_result.empty:
        try:
            preview_text = df_result.head(10).to_markdown(index=False)
        except Exception:
            preview_text = df_result.head(10).to_csv(index=False)

    instructions = "Você é um assistente de BI que explica resultados de forma clara e amigável."
    if db_mode:
        instructions += " Inclua a consulta SQL no final."
    else:
        instructions += " Não inclua SQL, pois os dados são de arquivo."

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"Pergunta:\n{question}\n\nPreview:\n{preview_text}\n\n" + (f"SQL:\n{sql_text}" if (db_mode and sql_text) else "")},
    ]
    resp = client.chat.completions.create(
        model=model_name, messages=messages, temperature=0.2, max_tokens=700
    )
    return resp.choices[0].message.content.strip()


def describe_dataframe_schema(df: pd.DataFrame) -> str:
    """Retorna esquema textual de um DataFrame."""
    lines = []
    for col in df.columns:
        lines.append(f'- "{col}" ({str(df[col].dtype)})')
    return "\n".join(lines)


# ----------------------
# App principal
# ----------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    show_login_page()
else:
    st.title("Análise Inteligente de Dados 🚀")
    show_logout_button()

    with st.sidebar:
        st.header("🔑 Configurações")
        openai_api_key = st.text_input("Chave OpenAI", type="password")
        modelo_selecionado = st.selectbox("Modelo:", ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4"))

        st.subheader("📂 Fonte de Dados")
        fonte_dados = st.selectbox("Fonte:", ["Banco de Dados", "Arquivo CSV/Excel"])

        db_user = db_password = db_host = db_port = db_name = None
        uploaded_file = None

        if fonte_dados == "Banco de Dados":
            db_user = st.text_input("Usuário", "user")
            db_password = st.text_input("Senha", "password", type="password")
            db_host = st.text_input("Host", "localhost")
            db_port = st.text_input("Porta", "5432")
            db_name = st.text_input("Banco", "meubanco")
        else:
            uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

    tab1, tab2, tab3 = st.tabs(["💬 Conversa", "📊 Dashboard", "📝 Histórico"])

    # ----------------------
    # Aba 1: Conversa
    # ----------------------
    with tab1:
        st.header("Converse com seus Dados")
        if not openai_api_key:
            st.warning("Insira a chave da API.")
        else:
            try:
                schema_text = "(sem esquema)"
                df_data = None
                db_mode = (fonte_dados == "Banco de Dados")

                if db_mode:
                    uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                    need = ('connected_uri' not in st.session_state) or (st.session_state.get('connected_uri') != uri)
                    if need:
                        with st.spinner("Introspectando esquema..."):
                            schema_text, schema_dict = introspect_schema(uri)
                            st.session_state['schema_text'] = schema_text
                            st.session_state['schema_dict'] = schema_dict
                            st.session_state['connected_uri'] = uri
                    schema_text = st.session_state.get('schema_text', "(sem esquema)")
                    system_prompt = "Você é um assistente de BI que analisa bancos de dados."
                else:
                    if uploaded_file is not None:
                        if uploaded_file.name.endswith(".csv"):
                            df_data = pd.read_csv(uploaded_file)
                        else:
                            df_data = pd.read_excel(uploaded_file)
                        st.session_state['df_data'] = df_data
                        schema_text = describe_dataframe_schema(df_data)
                        st.session_state['schema_text_file'] = schema_text
                        system_prompt = "Você é um assistente de BI que analisa arquivos CSV/Excel (dados estão em 'df')."
                    else:
                        st.info("Faça upload de um arquivo para começar.")

                if 'history' not in st.session_state:
                    st.session_state.history = []

                for message in st.session_state.history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Pergunte algo..."):
                    st.session_state.history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Analisando..."):
                            try:
                                client = OpenAI(api_key=openai_api_key)
                                history_text = build_history_text(st.session_state.history, max_turns=6)
                                sql_candidate = generate_sql_via_openai(client, modelo_selecionado, system_prompt, schema_text, history_text, prompt)
                                if sql_candidate.strip().upper() == "NO_SQL":
                                    final_text = generate_answer_no_sql(client, modelo_selecionado, system_prompt, schema_text, history_text, prompt)
                                    st.markdown(final_text)
                                    st.session_state.history.append({"role": "assistant", "content": final_text})
                                else:
                                    sql_text = extract_sql(sql_candidate)
                                    schema_dict = st.session_state.get('schema_dict', None)
                                    ok, vmsg = validate_sql_tables(sql_text, db_mode, schema_dict)
                                    if not ok:
                                        st.error(vmsg)
                                        st.session_state.history.append({"role": "assistant", "content": vmsg})
                                    else:
                                        try:
                                            if db_mode:
                                                engine = create_engine(uri)
                                                with engine.connect() as conn:
                                                    result = conn.execute(text(sql_text))
                                                    rows = result.fetchall()
                                                    df_result = pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()
                                            else:
                                                df_result = ps.sqldf(sql_text, {"df": st.session_state['df_data']})
                                        except Exception as e:
                                            error_message = f"Erro ao executar query: {e}\nQuery:\n{sql_text}"
                                            st.error(error_message)
                                            st.session_state.history.append({"role": "assistant", "content": error_message})
                                        else:
                                            final_text = generate_humanized_answer(client, modelo_selecionado, prompt, df_result, sql_text, db_mode)
                                            st.markdown(final_text)
                                            if not df_result.empty:
                                                st.subheader("Preview dos dados")
                                                st.dataframe(df_result.head(100))
                                            st.session_state.history.append({"role": "assistant", "content": final_text})
                            except Exception as e:
                                st.error(f"Ocorreu um erro: {e}")
            except Exception as e:
                st.error(f"Falha: {e}")

    # ----------------------
    # Aba 2: Dashboard
    # ----------------------
    with tab2:
        st.header("Dashboard Interativo 📊")

        if 'charts' not in st.session_state:
            st.session_state.charts = []

        schema_text = st.session_state.get('schema_text', st.session_state.get('schema_text_file', "(sem esquema)"))
        schema_dict = st.session_state.get('schema_dict', None)
        df_data = st.session_state.get('df_data', None)
        db_mode = (fonte_dados == "Banco de Dados")
        uri = None
        if db_mode:
            uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        for idx, chart in enumerate(st.session_state.charts):
            st.subheader(chart.get("title", f"Gráfico {idx+1}"))
            render_chart(chart)

            edit_prompt = st.text_input(f"Editar gráfico {idx+1}", key=f"edit_{idx}")
            if st.button(f"Atualizar gráfico {idx+1}"):
                if edit_prompt.strip():
                    try:
                        client = OpenAI(api_key=openai_api_key)
                        instr_text = generate_chart_instructions(client, modelo_selecionado, schema_text, edit_prompt, df_data, db_mode)
                        new_chart = parse_chart_instructions(instr_text)
                        valid, msg = validate_sql_tables(new_chart["sql"], db_mode, schema_dict)
                        if not valid:
                            st.error(msg)
                        else:
                            if db_mode:
                                engine = create_engine(uri)
                                with engine.connect() as conn:
                                    result = conn.execute(text(new_chart["sql"]))
                                    rows = result.fetchall()
                                    df_result = pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()
                            else:
                                df_result = ps.sqldf(new_chart["sql"], {"df": st.session_state['df_data']})
                            new_chart["df"] = df_result
                            st.session_state.charts[idx] = new_chart
                            st.success("Gráfico atualizado!")
                            if f"edit_{idx}" in st.session_state:
                                del st.session_state[f"edit_{idx}"]
                            st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao editar gráfico: {e}")

        st.subheader("Adicionar novo gráfico")
        new_chart_prompt = st.text_input("Descreva o gráfico:", key="new_chart")
        if st.button("Gerar novo gráfico"):
            if new_chart_prompt.strip():
                try:
                    client = OpenAI(api_key=openai_api_key)
                    instr_text = generate_chart_instructions(client, modelo_selecionado, schema_text, new_chart_prompt, df_data, db_mode)
                    chart_cfg = parse_chart_instructions(instr_text)
                    valid, msg = validate_sql_tables(chart_cfg["sql"], db_mode, schema_dict)
                    if not valid:
                        st.error(msg)
                    else:
                        if db_mode:
                            engine = create_engine(uri)
                            with engine.connect() as conn:
                                result = conn.execute(text(chart_cfg["sql"]))
                                rows = result.fetchall()
                                df_result = pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()
                        else:
                            df_result = ps.sqldf(chart_cfg["sql"], {"df": st.session_state['df_data']})
                        chart_cfg["df"] = df_result
                        st.session_state.charts.append(chart_cfg)
                        st.success("Novo gráfico adicionado!")
                        if "new_chart" in st.session_state:
                            del st.session_state["new_chart"]
                        st.rerun()
                except Exception as e:
                    st.error(f"Erro ao adicionar gráfico: {e}")

    # ----------------------
    # Aba 3: Histórico
    # ----------------------
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
