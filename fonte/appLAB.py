# appLAB.py
import streamlit as st
import pandas as pd
import plotly.express as px
import re
from sqlalchemy import create_engine, text
# from login_interface import show_login_page, show_logout_button
from db_utils import introspect_schema
from openai import OpenAI
import pandasql as ps
from user_data import get_user_config, save_db_config, decrypt_credentials
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Plataforma de Dados", page_icon="⚙️", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
/* Google Font: Inter */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Esconder rodapé e botão "Deploy" do Streamlit, preservando o botão de abrir/fechar sidebar! */
footer {visibility: hidden;}
.stDeployButton {display:none;}
#MainMenu {visibility: hidden;}

/* Customizar Alertas/Avisos (st.warning, st.info) */
div[data-testid="stAlert"] {
    border-radius: 8px;
    border: 1px solid rgba(128, 128, 128, 0.2);
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
}

/* Botões com bordas arredondadas e efeito hover */
div[data-testid="stButton"] button {
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* Arredondar bordas de selectbox e inputs */
div[data-baseweb="select"] > div, 
input[data-baseweb="base-input"] {
    border-radius: 8px !important;
}

/* Melhorar espaçamento do Título Principal */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 20px !important;
}
</style>
""", unsafe_allow_html=True)
# ----------------------
# Usuário e Sessão Local (Substituindo Supabase)
# ----------------------
if 'user' not in st.session_state:
    class LocalUser:
        id = "local_dev_user"
        email = "dev@local.user"
    st.session_state.user = LocalUser()
    st.session_state['logged_in'] = True

# ----------------------
# Utilitários e Funções da IA
# ----------------------

def build_history_text(history, max_turns=6):
    if not history: return ""
    max_messages = max_turns * 2;
    recent = history[-max_messages:]
    lines = [f"{'Usuário' if msg['role'] == 'user' else 'Assistente'}: {msg['content'].replace('```', ' `` ')}" for msg
             in recent]
    return "\n".join(lines)


def extract_sql(text: str) -> str:
    if not text: return ""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if m: return m.group(1).strip()
    m2 = re.search(r"(?i)(\bselect\b|\bwith\b|\binsert\b|\bupdate\b|\bdelete\b)(.*)", text, re.S)
    if m2: return (m2.group(1) + m2.group(2)).strip()
    return text.strip()


def describe_dataframe_schema(df: pd.DataFrame) -> str:
    lines = [f"A tabela se chama 'df' e possui as seguintes colunas:"]
    for col in df.columns: lines.append(f'- "{col}" (tipo: {str(df[col].dtype)})')
    return "\n".join(lines)


# --- FUNÇÕES DE IA ESPECIALIZADAS ---

def classify_intent(client, model_name, question, history_text):
    """Classifica a intenção do usuário em uma de quatro categorias."""
    system_prompt = (
        "Você é um especialista em interpretar a intenção do usuário. Classifique a pergunta em uma das categorias:\n"
        "- `data_query`: Pedido de dados específicos, totais, médias, que exija uma consulta SQL.\n"
        "- `schema_info`: Pedido sobre a estrutura dos dados ('quais tabelas existem?', 'descreva a tabela X').\n"
        "- `schema_summary`: Pedido de uma explicação ou resumo sobre o propósito do banco de dados ('o que é esse banco?', 'qual o contexto dos dados?').\n"
        "- `suggestion_request`: Pedido de sugestões, ideias, ou uma pergunta geral/aberta ('olá', 'me ajude').\n"
        "Responda APENAS com a categoria."
    )
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Histórico:\n{history_text}\n\nPergunta do usuário:\n{question}"}]
    try:
        resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.0, max_completion_tokens=50)
        return resp.choices[0].message.content.strip().lower().replace('`', '')
    except Exception:
        return 'data_query'


def generate_sql(client, model_name, system_prompt, schema_text, history_text, question):
    messages = [{"role": "system", "content": system_prompt + "\n\n---\nEsquema:\n" + schema_text},
                {"role": "user", "content": f"Histórico:\n{history_text or '(vazio)'}\n\nPergunta:\n{question}"}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.0, max_completion_tokens=1000)
    return resp.choices[0].message.content.strip()


def generate_humanized_answer(client, model_name, question, df_result, sql_text=None, show_sql=False):
    preview_text = df_result.head(10).to_markdown(
        index=False) if df_result is not None and not df_result.empty else "(A consulta não retornou dados)"
    system_prompt = "Você é um assistente de BI que explica resultados de forma clara e amigável. Use tópicos, negrito e tabelas em markdown."
    user_content = f"Com base na pergunta '{question}' e nos dados retornados, gere uma análise completa.\n\nPreview:\n{preview_text}\n\n"
    if show_sql and sql_text:
        system_prompt += "\nSua resposta DEVE terminar com a consulta SQL utilizada."
        user_content += f"SQL Executada:\n```sql\n{sql_text}\n```"
    else:
        system_prompt += "\nNUNCA inclua o código SQL na sua resposta."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.2, max_completion_tokens=1500)
    return resp.choices[0].message.content.strip()


def generate_suggestions(client, model_name, schema_text, question):
    system_prompt = "Você é um analista de dados sênior e proativo. O usuário fez uma pergunta aberta. Com base no esquema, sugira de 3 a 5 perguntas ou gráficos interessantes que ele poderia fazer. Apresente em tópicos."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user",
                                                               "content": f"Minha pergunta foi: '{question}'.\n\nO esquema é:\n{schema_text}\n\nQuais análises você sugere?"}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.5, max_completion_tokens=800)
    return resp.choices[0].message.content.strip()


def generate_schema_summary(client, model_name, schema_text, question):
    system_prompt = "Você é um arquiteto de dados e analista de negócios sênior. Sua tarefa é analisar o esquema de um banco de dados e fornecer um resumo executivo sobre seu propósito.\nINSTRUÇÕES:\n1. Infira o tema principal do banco (ex: 'Vendas', 'Bioinformática').\n2. Descreva as prováveis relações entre as tabelas.\n3. Apresente a resposta em seções como 'Tema Principal', 'Principais Entidades' e 'Possíveis Análises'.\n4. Baseie sua análise unicamente nos nomes fornecidos no esquema."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user",
                                                               "content": f"A pergunta do usuário foi: '{question}'.\n\nO esquema do banco de dados é:\n{schema_text}\n\nPor favor, gere o resumo executivo sobre este banco de dados."}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.3, max_completion_tokens=1500)
    return resp.choices[0].message.content.strip()


# <<< NOVA FUNÇÃO PARA EXTRAIR NOME DA TABELA >>>
def extract_table_name(client, model_name, question, schema_dict):
    """Extrai o nome de uma tabela da pergunta do usuário."""
    system_prompt = (
        "Sua tarefa é extrair um único nome de tabela da pergunta do usuário. As tabelas disponíveis são listadas abaixo.\n"
        "Responda APENAS com o nome exato da tabela em minúsculas. Se nenhuma tabela específica for mencionada ou se o usuário pedir uma lista de todas as tabelas, responda com a palavra 'NONE'."
    )
    table_list = "\n".join([f"- {t}" for t in schema_dict.keys()])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"As tabelas disponíveis são:\n{table_list}\n\nPergunta do usuário: '{question}'\n\nNome da tabela mencionada:"}
    ]
    try:
        resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.0, max_completion_tokens=50)
        result = resp.choices[0].message.content.strip().lower().splitlines()[0]
        return result if result in schema_dict.keys() else 'NONE'
    except Exception:
        return 'NONE'


# <<< NOVA FUNÇÃO PARA FORMATAR DETALHES DA TABELA >>>
def format_table_details(table_name, schema_dict):
    """Formata os detalhes de uma tabela específica em Markdown."""
    if table_name in schema_dict:
        columns = schema_dict[table_name]
        if not columns:
            return f"A tabela `{table_name}` existe, mas não foi possível ler suas colunas."

        header = f"### Detalhes da Tabela: `{table_name}`\n\n| Nome da Coluna | Tipo de Dado |\n|---|---|\n"
        rows = [f"| `{col[0]}` | `{col[1]}` |" for col in columns]
        return header + "\n".join(rows)
    else:
        return f"Desculpe, não encontrei uma tabela chamada `{table_name}`."


# --- FUNÇÕES DO DASHBOARD E UTILITÁRIOS GERAIS ---
def generate_chart_instructions(client, model_name, schema_text, question, db_mode, df_preview=None):
    preview_text = df_preview.head(5).to_markdown(
        index=False) if df_preview is not None and not df_preview.empty else ""
    if db_mode:
        dialect_instructions = "3. A consulta deve ser compatível com o dialeto PostgreSQL.\n4. Para agrupar por mês, use `to_char(\"nome_da_coluna_data\", 'YYYY-MM')`."
    else:
        dialect_instructions = "3. A consulta deve ser compatível com o dialeto SQLite.\n4. Para agrupar por mês, use `strftime('%Y-%m', \"nome_da_coluna_data\")`."
    system_content = (
            "Você é uma especialista sênior em visualização de dados. Sua missão é traduzir o pedido do usuário em uma configuração de gráfico completa e precisa.\n"
            "\n--- HIERARQUIA DE REGRAS ---\n"
            "1. **PRIORIDADE MÁXIMA: OBEDECER AO USUÁRIO.** Se o usuário pedir um tipo de gráfico específico (ex: 'gráfico de pizza'), VOCÊ DEVE usar esse tipo.\n"
            "2. **MODO ESPECIALISTA (FALLBACK):** SOMENTE SE o pedido for vago, use sua expertise para escolher a melhor visualização.\n"
            "\nREGRAS CRÍTICAS DE SINTAXE SQL:\n"
            "1. Use EXCLUSIVAMENTE os nomes de colunas do esquema.\n2. Envolva colunas com espaços/caracteres especiais em aspas duplas.\n" + dialect_instructions +
            "\nREGRAS CRÍTICAS DE SAÍDA:\n"
            "1. Gere um título inteligente.\n2. Se apropriado, use 'Color' para adicionar uma dimensão.\n3. Retorne APENAS no formato especificado abaixo.\n"
            "\n--- FORMATO OBRIGATÓRIO ---\nSQL: <query>\nTipo: <bar|line|pie|scatter|histogram|area>\nX: <coluna X>\nY: <coluna Y>\nColor: <(opcional) coluna cor>\nTítulo: <título>"
    )
    messages = [{"role": "system", "content": system_content}, {"role": "user",
                                                                "content": f"Esquema:\n{schema_text}\n\nPreview:\n{preview_text}\n\nPedido:\n{question}"}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.0, max_completion_tokens=800)
    return resp.choices[0].message.content.strip()


def parse_chart_instructions(text):
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


AVATAR_USER = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='24' height='24'%3E%3Cpath fill='%23E0E0E0' d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/%3E%3C/svg%3E"
AVATAR_BOT = "fonte/imagens/icone_branco.png"

def _strip_quotes(val: str):
    if not isinstance(val, str): return val
    return val.strip().strip('"').strip("'")


def render_chart(chart_config):
    df = chart_config.get("df", pd.DataFrame())
    if df is None or df.empty: st.warning("Sem dados para exibir."); return
    tipo = chart_config.get("type", "bar");
    x = _strip_quotes(chart_config.get("x"));
    y = _strip_quotes(chart_config.get("y"));
    color = _strip_quotes(chart_config.get("color"))
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
    if not sql_text: return False, "Consulta SQL vazia.", sql_text
    
    if not db_mode:
        sql_text = re.sub(r"(?i)\b(?:from|join)\s+[`'\"]?[a-zA-Z0-9_]+[`'\"]?\b", "FROM df", sql_text)
        
    sql_lower = sql_text.lower()
    if db_mode:
        if not schema_dict: return True, None, sql_text
        available = list(schema_dict.keys())
        found = any(re.search(r"\b" + re.escape(t.lower()) + r"\b", sql_lower) for t in available)
        if not found: return False, f"Tabela não encontrada. Disponíveis: {', '.join(available)}", sql_text
        return True, None, sql_text
    else:
        if re.search(r"\bfrom\s+[`'\"]?df[`'\"]?\b", sql_lower): return True, None, sql_text
        return False, "Erro na tabela para pandasql.", sql_text


# ----------------------
# App principal
# ----------------------
# Bypass Auth:
st.session_state['logged_in'] = True
if 'user' not in st.session_state:
    class MockUser:
        id = "dev_local_user_id"
        email = "dev@local.user"
    st.session_state.user = MockUser()

if not st.session_state['logged_in']:
    show_login_page()
else:
    st.title("Análise Inteligente de Dados")
    # show_logout_button()

    if 'config_loaded' not in st.session_state:
        user_id = st.session_state.user.id
        with st.spinner("Carregando configurações..."):
            user_config = get_user_config(user_id)
            if user_config:
                st.session_state.fonte_dados = user_config.get('data_source_type', 'Banco de Dados')
                if user_config.get('data_source_type') == 'database' and user_config.get('db_credentials'):
                    decrypted_creds = decrypt_credentials(user_config['db_credentials'])
                    if decrypted_creds: st.session_state.db_creds = decrypted_creds
            st.session_state.config_loaded = True

    uploaded_file = None
    with st.sidebar:
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Painel de Controle</h3>", unsafe_allow_html=True)
        
        menu_selecionado = option_menu(
            menu_title=None, 
            options=["Inteligência Artificial", "Conexão de Dados"], 
            icons=["cpu", "database"], 
            menu_icon="cast", 
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent", "border": "none"},
                "icon": {"color": "var(--text-color)", "font-size": "16px"}, 
                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "rgba(255,255,255,0.05)"},
                "nav-link-selected": {"background-color": "var(--primary-color)", "border-radius": "8px", "color": "white", "font-weight": "600"},
            }
        )
        
        st.divider()

        # Modelos
        model_options = {
            "o4-mini (Recomendado - Rápido)": "o4-mini",
            "GPT-5.4 (Máximo Desempenho)": "gpt-5.4",
            "o3 (Especialista em Lógica)": "o3",
            "GPT-5.4-mini (Alta Performance)": "gpt-5.4-mini"
        }
        
        # Recuperar estado atual
        openai_api_key = st.session_state.get('openai_api_key', '')
        modelo_selecionado = st.session_state.get('modelo_selecionado', 'o4-mini')
        fonte_dados = st.session_state.get('fonte_dados', 'Banco de Dados')
        db_creds_saved = st.session_state.get('db_creds', {})

        if menu_selecionado == "Inteligência Artificial":
            st.markdown("#### Configurações da IA")
            nova_chave = st.text_input("Chave OpenAI", type="password", value=openai_api_key)
            st.session_state['openai_api_key'] = nova_chave
            openai_api_key = nova_chave
            
            # Tentar achar o índice correto do selectbox
            default_ix = 0
            for i, val in enumerate(model_options.values()):
                if val == modelo_selecionado:
                    default_ix = i
                    break
                    
            modelo_label = st.selectbox("Selecione o Modelo:", list(model_options.keys()), index=default_ix)
            modelo_selecionado = model_options[modelo_label]
            st.session_state['modelo_selecionado'] = modelo_selecionado

        elif menu_selecionado == "Conexão de Dados":
            st.markdown("#### Fonte de Dados")
            fonte_dados_index = 0 if fonte_dados == 'Banco de Dados' else 1
            novo_fonte = st.radio("Origem:", ["Banco de Dados", "Arquivo CSV/Excel"], index=fonte_dados_index)
            st.session_state['fonte_dados'] = novo_fonte
            fonte_dados = novo_fonte
            
            if fonte_dados == "Banco de Dados":
                with st.form("db_config_form"):
                    st.write("Insira as credenciais do seu banco.")
                    db_user = st.text_input("Usuário", value=db_creds_saved.get('user', ''));
                    db_password = st.text_input("Senha", value=db_creds_saved.get('password', ''), type="password")
                    db_host = st.text_input("Host", value=db_creds_saved.get('host', ''));
                    db_port = st.text_input("Porta", value=db_creds_saved.get('port', ''));
                    db_name = st.text_input("Banco", value=db_creds_saved.get('dbname', ''))
                    if st.form_submit_button("Salvar Conexão"):
                        credentials_to_save = {"user": db_user, "password": db_password, "host": db_host, "port": db_port,
                                               "dbname": db_name}
                        if all(credentials_to_save.values()):
                            with st.spinner("Salvando..."):
                                if save_db_config(st.session_state.user.id, credentials_to_save):
                                    st.session_state.db_creds = credentials_to_save;
                                    st.session_state.fonte_dados = 'Banco de Dados';
                                    st.success("Conexão salva!");
                                    st.rerun()
                        else:
                            st.warning("Preencha todos os campos.")
            else:
                uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    tab1, tab2, tab3 = st.tabs(["Conversa", "Dashboard", "Histórico"])

    with tab1:
        st.header("Converse com seus Dados")
        if not openai_api_key:
            st.warning("Insira a chave da API OpenAI para começar.")
        else:
            client = OpenAI(api_key=openai_api_key)
            db_mode = (fonte_dados == "Banco de Dados")
            schema_text = "";
            schema_dict = {};
            df_data = None;
            uri = None
            try:
                if db_mode:
                    if 'db_creds' in st.session_state and all(st.session_state.db_creds.values()):
                        creds = st.session_state.db_creds
                        uri = f"postgresql+psycopg2://{creds['user']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['dbname']}"
                        if st.session_state.get('connected_uri') != uri:
                            with st.spinner("Analisando esquema do banco de dados..."):
                                schema_text, schema_dict = introspect_schema(uri)
                                st.session_state['schema_text'] = schema_text;
                                st.session_state['schema_dict'] = schema_dict;
                                st.session_state['connected_uri'] = uri
                        schema_text = st.session_state.get('schema_text', "")
                        schema_dict = st.session_state.get('schema_dict', {})
                else:
                    if uploaded_file is not None:
                        df_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(
                            uploaded_file)
                        st.session_state['df_data'] = df_data;
                        schema_text = describe_dataframe_schema(df_data)
                    elif 'df_data' in st.session_state:
                        df_data = st.session_state['df_data'];
                        schema_text = describe_dataframe_schema(df_data)
                    else:
                        st.info("Faça upload de um arquivo para começar.")
            except Exception as e:
                st.error(f"Falha ao carregar fonte de dados: {e}")

            if 'history' not in st.session_state: st.session_state.history = []
            for message in st.session_state.history:
                av = AVATAR_USER if message["role"] == "user" else AVATAR_BOT
                with st.chat_message(message["role"], avatar=av): st.markdown(message["content"])

            if prompt := st.chat_input("Pergunte algo sobre seus dados..."):
                st.session_state.history.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar=AVATAR_USER):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar=AVATAR_BOT):
                    with st.spinner("Analisando..."):
                        if (db_mode and not schema_text) or (not db_mode and df_data is None):
                            st.warning("Fonte de dados não configurada.");
                            st.stop()

                        history_text = build_history_text(st.session_state.history, max_turns=6)
                        intent = classify_intent(client, modelo_selecionado, prompt, history_text)
                        final_text = "";
                        response_handled_internally = False

                        if 'schema_summary' in intent:
                            final_text = generate_schema_summary(client, modelo_selecionado, schema_text, prompt)
                        elif 'schema_info' in intent:
                            current_schema_dict = st.session_state.get('schema_dict', {}) if db_mode else {
                                'df': [(c, str(t)) for c, t in zip(df_data.columns, df_data.dtypes)]}
                            table_name = extract_table_name(client, modelo_selecionado, prompt, current_schema_dict)
                            if table_name != 'none':
                                final_text = format_table_details(table_name, current_schema_dict)
                            else:
                                table_list = "\n".join([f"- `{t}`" for t in current_schema_dict.keys()])
                                final_text = f"Aqui estão as tabelas/arquivos que encontrei:\n\n{table_list}"
                        elif 'suggestion_request' in intent:
                            final_text = generate_suggestions(client, modelo_selecionado, schema_text, prompt)
                        else:  # intent == 'data_query'
                            if db_mode:
                                system_prompt = "Você é um especialista em PostgreSQL. Gere SQL para responder a perguntas sobre o esquema."
                            else:
                                system_prompt = "Você é um especialista em pandasql para consultar 'df'.\nREGRAS:\n1. Use EXCLUSIVAMENTE nomes de colunas do esquema.\n2. A tabela é SEMPRE 'df'.\n3. Envolva colunas com espaços/caracteres especiais em aspas duplas."
                            sql_candidate = generate_sql(client, modelo_selecionado, system_prompt, schema_text,
                                                         history_text, prompt)
                            sql_text = extract_sql(sql_candidate)
                            if not sql_text or "NO_SQL" in sql_candidate.upper():
                                final_text = generate_suggestions(client, modelo_selecionado, schema_text, prompt)
                            else:
                                schema_dict_ctx = st.session_state.get('schema_dict', {})
                                ok, vmsg, sql_text = validate_sql_tables(sql_text, db_mode, schema_dict_ctx)
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
                                        final_text = f"Ocorreu um erro na análise: {e}"
                        if not response_handled_internally:
                            st.markdown(final_text)
                            st.session_state.history.append({"role": "assistant", "content": final_text})

    with tab2:
        st.header("Dashboard Interativo")
        if 'charts' not in st.session_state: st.session_state.charts = []
        db_mode_dash = (fonte_dados == "Banco de Dados")
        schema_text_ctx = st.session_state.get('schema_text', '') if db_mode_dash else describe_dataframe_schema(
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
                                    valid, msg, new_sql = validate_sql_tables(new_chart_cfg["sql"], db_mode_dash, schema_dict_ctx)
                                    new_chart_cfg["sql"] = new_sql
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
                        valid, msg, new_sql = validate_sql_tables(chart_cfg["sql"], db_mode_dash, schema_dict_ctx)
                        chart_cfg["sql"] = new_sql
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
        st.header("Histórico de Conversas")
        if 'history' not in st.session_state or not st.session_state.history:
            st.info("Nenhuma conversa registrada ainda.")
        else:
            for i, msg in enumerate(st.session_state.history):
                role = "Usuário" if msg["role"] == "user" else "Assistente"
                with st.expander(f"{role} - Mensagem {i + 1}"): st.markdown(msg["content"])
            if st.button("Limpar Histórico"):
                st.session_state.history = [];
                st.rerun()