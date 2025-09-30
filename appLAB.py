# appLAB.py
import streamlit as st
import pandas as pd
import plotly.express as px
import re
from sqlalchemy import create_engine, text
from login_interface import show_login_page, show_logout_button
from db_utils import introspect_schema
from openai import OpenAI
import pandasql as ps
from user_data import get_user_config, save_db_config, decrypt_credentials

st.set_page_config(page_title="Plataforma de Dados", layout="wide")


# ----------------------
# Utilitários e Funções da IA
# ----------------------

def build_history_text(history, max_turns=6):
    """Converte histórico em texto compacto para enviar à IA."""
    if not history: return ""
    max_messages = max_turns * 2;
    recent = history[-max_messages:]
    lines = [f"{'Usuário' if msg['role'] == 'user' else 'Assistente'}: {msg['content'].replace('```', ' `` ')}" for msg
             in recent]
    return "\n".join(lines)


def extract_sql(text: str) -> str:
    """Extrai SQL de um texto."""
    if not text: return ""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if m: return m.group(1).strip()
    m2 = re.search(r"(?i)(\bselect\b|\bwith\b|\binsert\b|\bupdate\b|\bdelete\b)(.*)", text, re.S)
    if m2: return (m2.group(1) + m2.group(2)).strip()
    return text.strip()


def describe_dataframe_schema(df: pd.DataFrame) -> str:
    """Retorna esquema textual de um DataFrame."""
    lines = [f"A tabela se chama 'df' e possui as seguintes colunas:"]
    for col in df.columns: lines.append(f'- "{col}" (tipo: {str(df[col].dtype)})')
    return "\n".join(lines)


# --- FUNÇÕES DE IA ESPECIALIZADAS ---

def classify_intent(client, model_name, question, history_text):
    """Classifica a intenção do usuário em uma de três categorias."""
    system_prompt = (
        "Você é um especialista em interpretar a intenção do usuário. Classifique a pergunta do usuário em uma das seguintes categorias:\n"
        "- `data_query`: O usuário está pedindo por dados específicos, totais, médias, listas ou qualquer análise que exija uma consulta SQL.\n"
        "- `schema_info`: O usuário está perguntando sobre a estrutura dos dados, como 'quais tabelas existem?', 'descreva a tabela X', 'quais as colunas da tabela Y?'.\n"
        "- `suggestion_request`: O usuário está pedindo sugestões, ideias, ou fazendo uma pergunta geral/aberta (ex: 'olá', 'o que posso perguntar?', 'me ajude').\n"
        "Responda APENAS com a categoria, nada mais."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Histórico da conversa:\n{history_text}\n\nPergunta do usuário:\n{question}"}
    ]
    try:
        resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.0, max_tokens=20)
        return resp.choices[0].message.content.strip().lower().replace('`', '')
    except Exception:
        return 'data_query'  # Default seguro


def generate_sql(client, model_name, system_prompt, schema_text, history_text, question):
    """Função genérica para pedir SQL à OpenAI."""
    messages = [{"role": "system", "content": system_prompt + "\n\n---\nEsquema dos Dados:\n" + schema_text},
                {"role": "user", "content": f"Histórico:\n{history_text or '(vazio)'}\n\nPergunta:\n{question}"}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.0, max_tokens=800)
    return resp.choices[0].message.content.strip()


def generate_humanized_answer(client, model_name, question, df_result, sql_text=None, show_sql=False):
    """Gera resposta humanizada. Inclui SQL apenas se show_sql=True."""
    preview_text = df_result.head(10).to_markdown(
        index=False) if df_result is not None and not df_result.empty else "(A consulta não retornou dados)"
    system_prompt = "Você é um assistente de BI que explica resultados de forma clara e amigável para um usuário de negócios.\nSua resposta DEVE ser estruturada e informativa. Use tópicos, negrito e tabelas em markdown."
    user_content = f"Com base na pergunta '{question}' e nos dados retornados, gere uma análise completa e informativa.\n\nPreview dos Dados:\n{preview_text}\n\n"
    if show_sql and sql_text:
        system_prompt += "\nSua resposta DEVE terminar com a consulta SQL utilizada."
        user_content += f"Consulta SQL Executada:\n```sql\n{sql_text}\n```"
    else:
        system_prompt += "\nNUNCA inclua o código SQL na sua resposta."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.2, max_tokens=1000)
    return resp.choices[0].message.content.strip()


def generate_suggestions(client, model_name, schema_text, question):
    """Gera sugestões de análise."""
    system_prompt = "Você é um analista de dados sênior e proativo. Seu objetivo é ajudar o usuário a descobrir insights em seus dados.\nO usuário fez uma pergunta aberta. Com base no esquema dos dados, sugira de 3 a 5 perguntas ou tipos de gráficos interessantes que ele poderia fazer para explorar melhor seus dados.\nApresente as sugestões em formato de lista (tópicos)."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user",
                                                               "content": f"A minha pergunta foi: '{question}'.\n\nO esquema dos dados disponíveis é:\n{schema_text}\n\nCom base nisso, quais análises você me sugere?"}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.5, max_tokens=500)
    return resp.choices[0].message.content.strip()


# --- FUNÇÕES DO DASHBOARD ---
def generate_chart_instructions(client, model_name, schema_text, question, db_mode, df_preview=None):
    """Pede instruções de gráfico para a IA, com hierarquia de regras."""
    preview_text = df_preview.head(5).to_markdown(
        index=False) if df_preview is not None and not df_preview.empty else ""

    if db_mode:
        dialect_instructions = "3. A consulta deve ser compatível com o dialeto PostgreSQL.\n4. Para agrupar por mês, use a função `to_char(\"nome_da_coluna_data\", 'YYYY-MM')`."
    else:  # Modo Arquivo (pandasql/SQLite)
        dialect_instructions = "3. A consulta deve ser compatível com o dialeto SQLite.\n4. Para agrupar por mês, use a função `strftime('%Y-%m', \"nome_da_coluna_data\")`."

    system_content = (
            "Você é uma especialista sênior em visualização de dados. Sua missão é traduzir o pedido do usuário em uma configuração de gráfico completa e precisa.\n"
            "\n--- HIERARQUIA DE REGRAS ---\n"
            "1. **PRIORIDADE MÁXIMA: OBEDECER AO USUÁRIO.** Se o usuário pedir um tipo de gráfico específico (ex: 'gráfico de pizza', 'gráfico de barras'), VOCÊ DEVE usar esse tipo no campo 'Tipo'. Não mude a escolha dele.\n"
            "2. **MODO ESPECIALISTA (FALLBACK):** SOMENTE SE o pedido for vago ou não especificar um tipo de gráfico, use sua expertise para escolher a melhor visualização, seguindo as diretrizes abaixo:\n"
            "   - Para comparações entre categorias -> use 'bar'.\n"
            "   - Para tendências ao longo do tempo -> use 'line'.\n"
            "   - Para a relação entre duas variáveis numéricas -> use 'scatter'.\n"
            "   - Para a distribuição de uma variável -> use 'histogram'.\n"
            "   - Para composição de um todo -> use 'pie'.\n"
            "\nREGRAS CRÍTICAS DE SINTAXE SQL:\n"
            "1. Use EXCLUSIVAMENTE os nomes de colunas fornecidos no esquema. NÃO INVENTE nomes de colunas.\n"
            "2. Se um nome de coluna contiver espaços ou caracteres especiais, você DEVE envolvê-lo em aspas duplas (ex: `\"Receita Total (R$)\"`).\n"
            + dialect_instructions +
            "\nREGRAS CRÍTICAS DE SAÍDA:\n"
            "1. Gere um título inteligente e descritivo que corresponda ao pedido do usuário.\n"
            "2. Se apropriado e se adicionar valor, use o parâmetro 'Color' para incluir uma dimensão extra na análise.\n"
            "3. Retorne a resposta APENAS no formato especificado abaixo, sem nenhuma outra explicação.\n"
            "\n--- FORMATO OBRIGATÓRIO ---\n"
            "SQL: <query SQL para extrair os dados>\n"
            "Tipo: <bar|line|pie|scatter|histogram|area>\n"
            "X: <coluna para o eixo X>\n"
            "Y: <coluna para o eixo Y>\n"
            "Color: <(opcional) coluna para as cores>\n"
            "Título: <título descritivo do gráfico>\n"
            "Justificativa: <(opcional) por que este é o melhor gráfico para o pedido>"
    )

    messages = [{"role": "system", "content": system_content}, {"role": "user",
                                                                "content": f"Esquema:\n{schema_text}\n\nPreview dos Dados:\n{preview_text}\n\nPedido do Usuário:\n{question}"}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.0, max_tokens=500)
    return resp.choices[0].message.content.strip()


def parse_chart_instructions(text):
    """Extrai SQL, tipo, eixos e título da resposta da IA."""
    chart = {"sql": None, "type": "bar", "x": None, "y": None, "color": None, "title": "Gráfico"}
    for line in text.splitlines():
        if line.strip().lower().startswith("sql:"):
            chart["sql"] = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("tipo:"):
            chart["type"] = line.split(":", 1)[1].strip().lower()
        elif line.strip().lower().startswith("x:"):
            chart["x"] = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("y:"):
            chart["y"] = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("color:"):
            chart["color"] = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("título:") or line.strip().lower().startswith("titulo:"):
            chart["title"] = line.split(":", 1)[1].strip()
    return chart


def _strip_quotes(val: str):
    """Remove aspas extras de nomes de colunas."""
    if not isinstance(val, str): return val
    return val.strip().strip('"').strip("'")


def render_chart(chart_config):
    """Renderiza gráfico Plotly."""
    df = chart_config.get("df", pd.DataFrame())
    if df is None or df.empty: st.warning("Sem dados para exibir neste gráfico."); return

    tipo = chart_config.get("type", "bar")
    x = _strip_quotes(chart_config.get("x")) if chart_config.get("x") else None
    y = _strip_quotes(chart_config.get("y")) if chart_config.get("y") else None
    color = _strip_quotes(chart_config.get("color")) if chart_config.get("color") else None

    if x and x not in df.columns: st.error(f"Eixo X inválido '{x}'. Colunas: {list(df.columns)}"); return
    if y and y not in df.columns and tipo not in ["pie", "histogram"]: st.error(
        f"Eixo Y inválido '{y}'. Colunas: {list(df.columns)}"); return
    if color and color not in df.columns: st.error(
        f"Coluna de cor inválida '{color}'. Colunas: {list(df.columns)}"); return

    try:
        title = chart_config.get("title", "Gráfico")
        if tipo == "bar":
            fig = px.bar(df, x=x, y=y, color=color, title=title)
        elif tipo == "line":
            fig = px.line(df, x=x, y=y, color=color, title=title)
        elif tipo == "pie":
            fig = px.pie(df, names=x, values=y, color=color, title=title)
        elif tipo == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, title=title)
        elif tipo == "histogram":
            fig = px.histogram(df, x=x, y=y, color=color, title=title)
        elif tipo == "area":
            fig = px.area(df, x=x, y=y, color=color, title=title)
        else:
            st.warning(f"Tipo de gráfico '{tipo}' não suportado."); return
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao gerar gráfico: {e}")


def validate_sql_tables(sql_text, db_mode, schema_dict=None):
    if not sql_text: return False, "Consulta SQL vazia."
    sql_lower = sql_text.lower()
    if db_mode:
        if not schema_dict: return True, None
        available = list(schema_dict.keys())
        found = any(re.search(r"\b" + re.escape(t.lower()) + r"\b", sql_lower) for t in available)
        if not found: return False, f"Tabela não encontrada na consulta. Disponíveis: {', '.join(available)}"
        return True, None
    else:
        if re.search(r"\bfrom\s+df\b", sql_lower) or re.search(r"\bfrom\s+\"df\"\b", sql_lower): return True, None
        return False, "Para CSV/Excel use a tabela chamada 'df'."


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

    if 'config_loaded' not in st.session_state:
        user_id = st.session_state.user.id
        with st.spinner("Carregando configurações..."):
            user_config = get_user_config(user_id)
            if user_config:
                st.session_state.fonte_dados = user_config.get('data_source_type', 'Banco de Dados')
                if user_config.get('data_source_type') == 'database' and user_config.get('db_credentials'):
                    decrypted_creds = decrypt_credentials(user_config['db_credentials'])
                    if decrypted_creds:
                        st.session_state.db_creds = decrypted_creds
            st.session_state.config_loaded = True

    with st.sidebar:
        st.header("🔑 Configurações")
        openai_api_key = st.text_input("Chave OpenAI", type="password")
        modelo_selecionado = st.selectbox("Modelo:", ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4"))
        st.subheader("📂 Fonte de Dados")
        fonte_dados_index = 0 if st.session_state.get('fonte_dados', 'Banco de Dados') == 'Banco de Dados' else 1
        fonte_dados = st.selectbox("Fonte:", ["Banco de Dados", "Arquivo CSV/Excel"], index=fonte_dados_index)
        db_creds_saved = st.session_state.get('db_creds', {})
        uploaded_file = None
        if fonte_dados == "Banco de Dados":
            with st.form("db_config_form"):
                st.write("Insira e salve as credenciais do seu banco de dados.")
                db_user = st.text_input("Usuário", value=db_creds_saved.get('user', ''))
                db_password = st.text_input("Senha", value=db_creds_saved.get('password', ''), type="password")
                db_host = st.text_input("Host", value=db_creds_saved.get('host', ''))
                db_port = st.text_input("Porta", value=db_creds_saved.get('port', ''))
                db_name = st.text_input("Banco", value=db_creds_saved.get('dbname', ''))
                if st.form_submit_button("Salvar Conexão"):
                    credentials_to_save = {"user": db_user, "password": db_password, "host": db_host, "port": db_port,
                                           "dbname": db_name}
                    if all(credentials_to_save.values()):
                        with st.spinner("Salvando..."):
                            if save_db_config(st.session_state.user.id, credentials_to_save):
                                st.session_state.db_creds = credentials_to_save;
                                st.session_state.fonte_dados = 'database';
                                st.success("Conexão salva com sucesso!");
                                st.rerun()
                    else:
                        st.warning("Preencha todos os campos do banco de dados.")
        else:
            uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

    tab1, tab2, tab3 = st.tabs(["💬 Conversa", "📊 Dashboard", "📝 Histórico"])

    with tab1:
        st.header("Converse com seus Dados")
        if not openai_api_key:
            st.warning("Insira a chave da API OpenAI para começar.")
        else:
            client = OpenAI(api_key=openai_api_key)
            db_mode = (fonte_dados == "Banco de Dados")
            schema_text = "(sem esquema)";
            df_data = None;
            uri = None
            try:
                if db_mode and 'db_creds' in st.session_state and all(st.session_state.db_creds.values()):
                    creds = st.session_state.db_creds
                    uri = f"postgresql+psycopg2://{creds['user']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['dbname']}"
                    if st.session_state.get('connected_uri') != uri:
                        with st.spinner("Analisando esquema do banco de dados..."):
                            schema_text, schema_dict = introspect_schema(uri)
                            st.session_state['schema_text'] = schema_text;
                            st.session_state['schema_dict'] = schema_dict;
                            st.session_state['connected_uri'] = uri
                    schema_text = st.session_state.get('schema_text', "(sem esquema)")
                elif not db_mode:
                    if uploaded_file is not None:
                        df_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(
                            uploaded_file)
                        st.session_state['df_data'] = df_data;
                        schema_text = describe_dataframe_schema(df_data)
                    elif 'df_data' in st.session_state:
                        df_data = st.session_state['df_data'];
                        schema_text = describe_dataframe_schema(df_data)
                    else:
                        st.info("Faça upload de um arquivo CSV ou Excel para começar a análise.")
            except Exception as e:
                st.error(f"Falha ao carregar a fonte de dados: {e}")

            if 'history' not in st.session_state: st.session_state.history = []
            for message in st.session_state.history:
                with st.chat_message(message["role"]): st.markdown(message["content"])

            if prompt := st.chat_input("Pergunte algo sobre seus dados..."):
                st.session_state.history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analisando..."):
                        if (db_mode and schema_text == "(sem esquema)") or (not db_mode and df_data is None):
                            st.warning("Fonte de dados não configurada.");
                            st.stop()

                        history_text = build_history_text(st.session_state.history, max_turns=6)
                        intent = classify_intent(client, modelo_selecionado, prompt, history_text)
                        final_text = "";
                        response_handled_internally = False

                        if 'schema_info' in intent:
                            final_text = f"Aqui estão as informações sobre a estrutura dos seus dados:\n\n```\n{schema_text}\n```"
                        elif 'suggestion_request' in intent:
                            final_text = generate_suggestions(client, modelo_selecionado, schema_text, prompt)
                        else:
                            if db_mode:
                                system_prompt = "Você é um especialista em PostgreSQL. Gere SQL para responder a perguntas sobre o esquema fornecido."
                            else:
                                system_prompt = "Você é um especialista em pandasql para consultar um DataFrame 'df'.\nREGRAS CRÍTICAS:\n1. Use EXCLUSIVAMENTE os nomes de colunas do esquema.\n2. A tabela é SEMPRE 'df'.\n3. Envolva colunas com espaços/caracteres especiais em aspas duplas."

                            sql_candidate = generate_sql(client, modelo_selecionado, system_prompt, schema_text,
                                                         history_text, prompt)
                            sql_text = extract_sql(sql_candidate)

                            if not sql_text or "NO_SQL" in sql_candidate.upper():
                                final_text = generate_suggestions(client, modelo_selecionado, schema_text, prompt)
                            else:
                                schema_dict = st.session_state.get('schema_dict', {})
                                ok, vmsg = validate_sql_tables(sql_text, db_mode, schema_dict)
                                if not ok:
                                    final_text = f"Erro de validação: {vmsg}"
                                else:
                                    try:
                                        if db_mode:
                                            engine = create_engine(uri);
                                            df_result = pd.read_sql(text(sql_text), engine)
                                        else:
                                            df_result = ps.sqldf(sql_text, {"df": df_data})
                                        final_text = generate_humanized_answer(client, modelo_selecionado, prompt,
                                                                               df_result, sql_text, show_sql=db_mode)
                                        st.markdown(final_text)
                                        if not df_result.empty:
                                            st.subheader("Preview dos dados retornados");
                                            st.dataframe(df_result.head(100))
                                        st.session_state.history.append({"role": "assistant", "content": final_text})
                                        response_handled_internally = True
                                    except Exception as e:
                                        final_text = f"Ocorreu um erro ao executar a análise: {e}"
                        if not response_handled_internally:
                            st.markdown(final_text)
                            st.session_state.history.append({"role": "assistant", "content": final_text})

    with tab2:
        st.header("Dashboard Interativo 📊")
        if 'charts' not in st.session_state: st.session_state.charts = []
        db_mode_dash = (fonte_dados == "Banco de Dados")
        schema_text_ctx = st.session_state.get(
            'schema_text') if db_mode_dash and 'schema_text' in st.session_state else describe_dataframe_schema(
            st.session_state.get('df_data', pd.DataFrame()))
        schema_dict_ctx = st.session_state.get('schema_dict') if db_mode_dash else None
        df_data_ctx = st.session_state.get('df_data') if not db_mode_dash else None
        uri_ctx = f"postgresql+psycopg2://{st.session_state.db_creds['user']}:{st.session_state.db_creds['password']}@{st.session_state.db_creds['host']}:{st.session_state.db_creds['port']}/{st.session_state.db_creds['dbname']}" if db_mode_dash and 'db_creds' in st.session_state else None
        for idx, chart in enumerate(st.session_state.charts):
            st.subheader(chart.get("title", f"Gráfico {idx + 1}"));
            render_chart(chart)
            with st.expander(f"Opções para Gráfico {idx + 1}"):
                edit_prompt = st.text_input("Refine ou altere este gráfico:", key=f"edit_{idx}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Atualizar Gráfico", key=f"update_{idx}"):
                        if edit_prompt.strip() and openai_api_key:
                            with st.spinner("Atualizando gráfico..."):
                                try:
                                    client = OpenAI(api_key=openai_api_key)
                                    instr_text = generate_chart_instructions(client, modelo_selecionado,
                                                                             schema_text_ctx, edit_prompt, db_mode_dash,
                                                                             df_data_ctx)
                                    new_chart_cfg = parse_chart_instructions(instr_text);
                                    valid, msg = validate_sql_tables(new_chart_cfg["sql"], db_mode_dash,
                                                                     schema_dict_ctx)
                                    if not valid:
                                        st.error(msg)
                                    else:
                                        if db_mode_dash and uri_ctx:
                                            engine = create_engine(uri_ctx);
                                            df_result = pd.read_sql(text(new_chart_cfg["sql"]), engine)
                                        elif not db_mode_dash and df_data_ctx is not None:
                                            df_result = ps.sqldf(new_chart_cfg["sql"], {"df": df_data_ctx})
                                        else:
                                            df_result = pd.DataFrame()
                                        if not df_result.empty:
                                            new_chart_cfg["df"] = df_result;
                                            st.session_state.charts[idx] = new_chart_cfg;
                                            st.rerun()
                                        else:
                                            st.warning("A nova consulta para o gráfico não retornou dados.")
                                except Exception as e:
                                    st.error(f"Erro ao atualizar gráfico: {e}")
                with col2:
                    if st.button("Remover Gráfico", type="primary", key=f"remove_{idx}"):
                        st.session_state.charts.pop(idx);
                        st.rerun()
        st.subheader("Adicionar novo gráfico")
        new_chart_prompt = st.text_input("Descreva o gráfico que você deseja criar:", key="new_chart_prompt_input")
        if st.button("Gerar novo gráfico", key="new_chart_btn"):
            if new_chart_prompt.strip() and openai_api_key:
                with st.spinner("Gerando gráfico..."):
                    try:
                        client = OpenAI(api_key=openai_api_key)
                        instr_text = generate_chart_instructions(client, modelo_selecionado, schema_text_ctx,
                                                                 new_chart_prompt, db_mode_dash, df_data_ctx)
                        chart_cfg = parse_chart_instructions(instr_text);
                        valid, msg = validate_sql_tables(chart_cfg["sql"], db_mode_dash, schema_dict_ctx)
                        if not valid:
                            st.error(msg)
                        else:
                            if db_mode_dash and uri_ctx:
                                engine = create_engine(uri_ctx);
                                df_result = pd.read_sql(text(chart_cfg["sql"]), engine)
                            elif not db_mode_dash and df_data_ctx is not None:
                                df_result = ps.sqldf(chart_cfg["sql"], {"df": df_data_ctx})
                            else:
                                st.warning("Fonte de dados não está pronta.");
                                df_result = pd.DataFrame()
                            if not df_result.empty:
                                chart_cfg["df"] = df_result;
                                st.session_state.charts.append(chart_cfg);
                                st.rerun()
                            else:
                                st.warning("A consulta para o gráfico não retornou dados.")
                    except Exception as e:
                        st.error(f"Erro ao adicionar gráfico: {e}")

    with tab3:
        st.header("Histórico de Conversas 📝")
        if 'history' not in st.session_state or not st.session_state.history:
            st.info("Nenhuma conversa registrada ainda.")
        else:
            for i, msg in enumerate(st.session_state.history):
                role = "👤 Usuário" if msg["role"] == "user" else "🤖 Assistente"
                with st.expander(f"{role} - Mensagem {i + 1}"): st.markdown(msg["content"])
            if st.button("🗑️ Limpar Histórico"):
                st.session_state.history = [];
                st.rerun()