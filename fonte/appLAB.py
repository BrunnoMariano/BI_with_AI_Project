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
from user_data import (
    append_chat_message,
    clear_chat_history,
    load_dataframe_from_storage,
    save_db_config,
    save_uploaded_file,
)
from ui_styles import inject_global_styles, render_app_hero, render_section_intro

st.set_page_config(page_title="Plataforma de Dados", page_icon="⚙️", layout="wide")
inject_global_styles()
# ----------------------
# Estado de Autenticação Inicial
# ----------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

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
    lines = [f"A tabela se chama 'df' e possui {len(df)} registros."]
    lines.append("Colunas disponíveis:")
    for col in df.columns:
        series = df[col]
        sample_str = _series_examples(series)
        lines.append(
            f'- "{col}" (tipo: {str(series.dtype)}, nulos: {int(series.isna().sum())}, '
            f'unique: {int(series.dropna().nunique())}, exemplos: {sample_str})'
        )
    return "\n".join(lines)


SPECIALIZED_FILE_REQUIRED_COLUMNS = {
    "produto",
    "valor pago",
    "status venda",
    "forma de pagamento",
    "data venda",
    "criado em",
}


def _normalize_label(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _series_examples(series: pd.Series, limit: int = 3) -> str:
    examples = []
    seen = set()
    for value in series.dropna().astype(str):
        clean = value.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        examples.append(repr(clean[:80]))
        if len(examples) >= limit:
            break
    return ", ".join(examples) if examples else "(sem exemplos não nulos)"


def _top_value_summary(series: pd.Series, limit: int = 5, normalize: bool = False) -> str:
    clean = series.dropna().astype(str).map(lambda value: re.sub(r"\s+", " ", value).strip())
    clean = clean[clean != ""]
    if clean.empty:
        return "(sem valores não nulos)"
    if normalize:
        clean = clean.str.lower()
    counts = clean.value_counts().head(limit)
    return ", ".join([f"{idx} ({int(count)})" for idx, count in counts.items()])


def _count_case_variants(series: pd.Series) -> int:
    clean = series.dropna().astype(str).map(lambda value: re.sub(r"\s+", " ", value).strip())
    clean = clean[clean != ""]
    if clean.empty:
        return 0
    canonical = {}
    for value in clean.unique():
        key = value.lower()
        canonical.setdefault(key, set()).add(value)
    return sum(1 for variants in canonical.values() if len(variants) > 1)


def build_schema_dict_from_dataframe(df: pd.DataFrame) -> dict:
    return {"df": [(column, str(df[column].dtype)) for column in df.columns]}


def detect_dataset_profile(df: pd.DataFrame, file_name: str = "") -> str:
    normalized_columns = {_normalize_label(column) for column in df.columns}
    if SPECIALIZED_FILE_REQUIRED_COLUMNS.issubset(normalized_columns):
        return "client_sales_profile_v1"
    return "generic_file_v1"


def build_dataset_profile(df: pd.DataFrame, file_name: str = "") -> dict:
    profile_id = detect_dataset_profile(df, file_name)
    profile = {
        "profile_id": profile_id,
        "file_name": file_name or "",
        "business_defaults": {},
        "column_semantics": {},
        "query_hints": [],
        "data_quality_notes": [],
        "summary_stats": {},
    }

    if profile_id == "client_sales_profile_v1":
        profile["business_defaults"] = {
            "sales_definition": "Trate 'vendas' sem qualificador como linhas com Status Venda = pago.",
            "time_reference": "Use Data Venda como data principal; use Criado em para cadastro/criação ou como fallback quando fizer sentido.",
            "normalization": "Ao agrupar texto em Promotora, Bairro, Cidade e campos semelhantes, normalize com trim + lowercase quando isso evitar duplicidades artificiais.",
        }
        profile["column_semantics"] = {
            "Número": "Identificador principal do registro/transação.",
            "Número.1": "Número do endereço; não confundir com o identificador principal.",
            "Valor Pago": "Valor monetário textual no formato brasileiro; converta para número antes de somar, ordenar ou calcular médias.",
            "Status Venda": "Status operacional da transação; pago é o padrão para perguntas genéricas de vendas.",
            "Data Venda": "Momento da efetivação da venda; prefira esta coluna em análises de período.",
            "Criado em": "Data de criação/cadastro do registro; use para análises de origem/cadastro ou fallback controlado.",
        }
        profile["query_hints"] = [
            "Se o usuário pedir faturamento, receita ou valor vendido, some Valor Pago após converter de texto monetário para número.",
            "Se o usuário pedir vendas por período sem especificar status, considere apenas linhas pagas e use Data Venda.",
            "Se o usuário pedir agrupamentos por promotora, bairro ou cidade, normalize caixa e espaços para reduzir duplicidades artificiais.",
            "Se a pergunta mencionar cadastro, criação ou origem do registro, use Criado em como referência temporal.",
        ]

        if "Status Venda" in df.columns:
            profile["summary_stats"]["status_counts"] = _top_value_summary(df["Status Venda"])
        if "Forma de pagamento" in df.columns:
            profile["summary_stats"]["payment_counts"] = _top_value_summary(df["Forma de pagamento"])
        if "Produto" in df.columns:
            profile["summary_stats"]["product_counts"] = _top_value_summary(df["Produto"])

        if "Valor Pago" in df.columns and df["Valor Pago"].dtype == object:
            profile["data_quality_notes"].append(
                "Valor Pago está textual; consultas de soma/média precisam converter moeda brasileira para número."
            )
        if "Data Venda" in df.columns and "Criado em" in df.columns:
            sales_dates = pd.to_datetime(df["Data Venda"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
            created_dates = pd.to_datetime(df["Criado em"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
            profile["data_quality_notes"].append(
                f"Data Venda válida em {int(sales_dates.notna().sum())} linhas; Criado em válida em {int(created_dates.notna().sum())} linhas."
            )
        if "Número.1" in df.columns:
            profile["data_quality_notes"].append(
                "O arquivo possui cabeçalho duplicado para Número; pandas renomeou a segunda ocorrência para Número.1."
            )
        if "Promotora" in df.columns:
            promoter_variants = _count_case_variants(df["Promotora"])
            if promoter_variants:
                profile["data_quality_notes"].append(
                    f"Promotora tem {promoter_variants} grupos com variações de caixa/espaços; agrupar sem normalização pode fragmentar resultados."
                )
        if "Bairro" in df.columns:
            district_variants = _count_case_variants(df["Bairro"])
            if district_variants:
                profile["data_quality_notes"].append(
                    f"Bairro tem {district_variants} grupos com variações de caixa/espaços; use normalização em comparações agregadas."
                )
    else:
        profile["business_defaults"] = {
            "generic_mode": "Respeite o esquema recebido e não assuma regras de negócio específicas."
        }
        profile["query_hints"] = [
            "Use apenas colunas existentes no esquema e peça consultas simples quando o pedido do usuário for vago."
        ]

    return profile


def build_prompt_context(df: pd.DataFrame, profile: dict) -> str:
    lines = [describe_dataframe_schema(df)]
    lines.append("")
    lines.append("Contexto operacional do arquivo:")
    lines.append(f'- profile_id: {profile.get("profile_id", "generic_file_v1")}')

    business_defaults = profile.get("business_defaults", {})
    if business_defaults:
        lines.append("- defaults de negócio:")
        for key, value in business_defaults.items():
            lines.append(f"  - {key}: {value}")

    summary_stats = profile.get("summary_stats", {})
    if summary_stats:
        lines.append("- resumos úteis:")
        for key, value in summary_stats.items():
            lines.append(f"  - {key}: {value}")

    data_quality_notes = profile.get("data_quality_notes", [])
    if data_quality_notes:
        lines.append("- alertas de qualidade dos dados:")
        for note in data_quality_notes:
            lines.append(f"  - {note}")

    semantics = profile.get("column_semantics", {})
    if semantics:
        lines.append("- semântica de colunas críticas:")
        for column, description in semantics.items():
            if column in df.columns:
                lines.append(f"  - {column}: {description}")

    query_hints = profile.get("query_hints", [])
    if query_hints:
        lines.append("- dicas para interpretar perguntas:")
        for hint in query_hints:
            lines.append(f"  - {hint}")

    return "\n".join(lines)


def build_file_context(df: pd.DataFrame, file_name: str = ""):
    profile = build_dataset_profile(df, file_name)
    prompt_context = build_prompt_context(df, profile)
    schema_dict = build_schema_dict_from_dataframe(df)
    return prompt_context, schema_dict, profile


def build_sql_system_prompt(db_mode, profile):
    if db_mode:
        return (
            "Você gera EXCLUSIVAMENTE queries SQL compatíveis com PostgreSQL.\n\n"
            "REGRAS ABSOLUTAS:\n"
            "1. Use APENAS tabelas e colunas que existem no esquema fornecido. NUNCA invente.\n"
            "2. Envolva nomes com caracteres especiais ou espaços em aspas duplas.\n"
            "3. Use sintaxe PostgreSQL válida.\n"
            "4. Retorne APENAS o SQL puro, sem explicações, sem markdown, sem ```.\n"
        )

    generic_prompt = (
        "Você gera EXCLUSIVAMENTE queries SQL compatíveis com SQLite para consultar um DataFrame chamado 'df'.\n\n"
        "REGRAS ABSOLUTAS - VIOLAR QUALQUER UMA DELAS E UM ERRO:\n"
        "1. A tabela é SEMPRE 'df'. NUNCA use outro nome de tabela.\n"
        "2. Use APENAS colunas que existem no esquema fornecido. NUNCA invente colunas.\n"
        "3. Envolva TODOS os nomes de colunas em aspas duplas.\n"
        "4. Gere UMA única consulta SELECT. Não use múltiplas instruções SQL, UNION, código Python ou comentários.\n"
        "5. Use recursos SQLite simples: SUM, AVG, COUNT, MIN, MAX, GROUP BY, ORDER BY, DESC, ASC, WHERE, LIKE, IN, BETWEEN, LOWER, UPPER, TRIM, LIMIT, CASE, substr, REPLACE, CAST, COALESCE.\n"
        "6. NUNCA use ILIKE, to_char, EXTRACT, DATE_TRUNC, ::, sintaxe PostgreSQL ou window functions.\n"
        "7. Retorne APENAS o SQL puro. NUNCA inclua explicações, markdown ou ```.\n"
    )

    if profile.get("profile_id") != "client_sales_profile_v1":
        return generic_prompt + (
            "8. Para filtrar texto, prefira WHERE LOWER(TRIM(\"coluna\")) LIKE '%valor%'.\n"
            "9. Se a pergunta for vaga, escolha a consulta mais direta possível.\n"
        )

    return generic_prompt + (
        "8. Nesta estrutura especializada, quando o usuário falar genericamente em vendas, resultado, desempenho, faturamento ou receita sem pedir outro status, considere APENAS linhas com LOWER(TRIM(\"Status Venda\")) = 'pago'.\n"
        "9. Para converter \"Valor Pago\" em número, use exatamente este padrão de expressão: "
        "CAST(REPLACE(REPLACE(REPLACE(REPLACE(COALESCE(\"Valor Pago\", '0'), 'R$', ''), ' ', ''), '.', ''), ',', '.') AS REAL)\n"
        "10. Para análises por período, use \"Data Venda\" como padrão. Só use \"Criado em\" quando a pergunta for sobre criação/cadastro ou quando um fallback fizer parte da lógica pedida.\n"
        "11. Para agrupar texto em \"Promotora\", \"Bairro\", \"Cidade\" e similares, normalize com LOWER(TRIM(\"coluna\")) sempre que a pergunta pedir agregação/comparação por categoria.\n"
        "12. \"Número\" é o identificador principal do registro. \"Número.1\" é o número do endereço.\n"
        "13. Para agrupar por mês em colunas de data textual dd/mm/aaaa hh:mm:ss, use substr(\"Data Venda\", 7, 4) || '-' || substr(\"Data Venda\", 4, 2). Se precisar usar \"Criado em\", aplique a mesma lógica nessa coluna.\n"
        "14. Se a pergunta pedir quantidade por categoria e também o maior/menor valor, retorne a agregação já ordenada. Não use subconsulta se uma única agregação resolver.\n\n"
        "EXEMPLO CORRETO:\n"
        "Pergunta: 'qual o faturamento total?'\n"
        "Resposta: SELECT SUM(CAST(REPLACE(REPLACE(REPLACE(REPLACE(COALESCE(\"Valor Pago\", '0'), 'R$', ''), ' ', ''), '.', ''), ',', '.') AS REAL)) AS faturamento_total FROM df WHERE LOWER(TRIM(\"Status Venda\")) = 'pago'\n\n"
        "EXEMPLO CORRETO:\n"
        "Pergunta: 'vendas pagas por mês'\n"
        "Resposta: SELECT substr(\"Data Venda\", 7, 4) || '-' || substr(\"Data Venda\", 4, 2) AS mes, COUNT(*) AS total_vendas FROM df WHERE LOWER(TRIM(\"Status Venda\")) = 'pago' AND \"Data Venda\" IS NOT NULL GROUP BY mes ORDER BY mes\n\n"
        "EXEMPLO CORRETO:\n"
        "Pergunta: 'quantas vendas por promotora'\n"
        "Resposta: SELECT LOWER(TRIM(\"Promotora\")) AS promotora_normalizada, COUNT(*) AS total_vendas FROM df WHERE LOWER(TRIM(\"Status Venda\")) = 'pago' AND \"Promotora\" IS NOT NULL GROUP BY LOWER(TRIM(\"Promotora\")) ORDER BY total_vendas DESC\n"
    )


def build_sql_fix_prompt(error_text, sql_text, question, db_mode, profile):
    if db_mode:
        correction_guidance = "Corrija a query mantendo compatibilidade com PostgreSQL e retorne APENAS o SQL corrigido."
    elif profile.get("profile_id") == "client_sales_profile_v1":
        correction_guidance = (
            "Corrija a query mantendo compatibilidade com SQLite para a tabela df. "
            "Preserve as regras do perfil especializado: vendas genéricas usam status pago, "
            "Valor Pago precisa ser convertido de texto monetário para número, "
            "Data Venda é a data principal e agrupamentos textuais podem usar LOWER(TRIM(...)). "
            "Retorne APENAS o SQL corrigido."
        )
    else:
        correction_guidance = (
            "Corrija a query usando UMA única instrução SELECT compatível com SQLite para a tabela df. "
            "Retorne APENAS o SQL corrigido."
        )

    return (
        f"A query SQL abaixo falhou com erro: {error_text}\n\n"
        f"Query que falhou:\n{sql_text}\n\n"
        f"Pergunta original: {question}\n\n"
        f"{correction_guidance}"
    )


# --- FUNÇÕES DE IA ESPECIALIZADAS ---

def classify_intent(client, model_name, question, history_text, prompt_context=""):
    """Classifica a intenção do usuário em uma de quatro categorias."""
    system_prompt = (
        "Classifique a intenção do usuário em UMA das categorias abaixo. "
        "Responda APENAS com o nome da categoria, sem explicação, sem aspas, sem pontuação.\n\n"
        "Categorias:\n"
        "- data_query: Pedido de dados, totais, médias, filtros, listas, comparações. "
        "Ex: 'qual o total de vendas?', 'quais produtos venderam mais?', 'mostre os dados de outubro', "
        "'tem dados de setembro?', 'quais foram os dados?', 'quanto vendeu em janeiro?'\n"
        "- schema_info: Pedido sobre estrutura dos dados. "
        "Ex: 'quais colunas tem?', 'descreva a tabela X', 'quais tabelas existem?'\n"
        "- schema_summary: Pedido sobre contexto geral do banco. "
        "Ex: 'o que é esse banco?', 'que tipo de dados são esses?', 'qual o contexto?'\n"
        "- suggestion_request: Saudação, pedido aberto ou vago. "
        "Ex: 'olá', 'me ajude', 'o que posso perguntar?', 'boa tarde'\n"
        "\nUse o contexto do arquivo/banco para diferenciar perguntas de negócio, estrutura e pedidos vagos."
    )
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contexto disponível:\n{prompt_context or '(não informado)'}\n\nHistórico:\n{history_text}\n\nPergunta do usuário:\n{question}"}]
    try:
        resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.0, max_completion_tokens=20)
        return resp.choices[0].message.content.strip().lower().replace('`', '').strip()
    except Exception:
        return 'data_query'


def generate_sql(client, model_name, system_prompt, schema_text, history_text, question):
    messages = [{"role": "system", "content": system_prompt + "\n\n---\nEsquema:\n" + schema_text},
                {"role": "user", "content": f"Histórico:\n{history_text or '(vazio)'}\n\nPergunta:\n{question}"}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.0, max_completion_tokens=1000)
    return resp.choices[0].message.content.strip()


def generate_humanized_answer(client, model_name, question, df_result, profile=None, sql_text=None, show_sql=False):
    preview_text = df_result.head(15).to_markdown(
        index=False) if df_result is not None and not df_result.empty else "(A consulta não retornou dados)"
    system_prompt = (
        "Você é um assistente de dados. Responda de forma DIRETA e CONCISA.\n\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. Responda EXATAMENTE o que foi perguntado. Nada mais, nada menos.\n"
        "2. Se a pergunta é simples (ex: 'qual o total?', 'quais os dados?'), responda em 1-3 frases curtas.\n"
        "3. Se houver múltiplas linhas de dados, apresente como tabela markdown.\n"
        "4. NUNCA adicione seções como: 'Insights', 'Observações', 'Recomendações', 'Próximos passos', 'Resumo', 'Conclusão'.\n"
        "5. Use **negrito** para destacar números importantes.\n"
        "6. Seja natural e amigável, mas objetivo.\n"
        "7. Se os dados já respondem a pergunta claramente, apenas apresente-os sem elaborar."
    )
    if profile and profile.get("profile_id") == "client_sales_profile_v1":
        system_prompt += (
            "\n8. Se a pergunta falar genericamente de vendas, assuma que a consulta já considerou vendas pagas."
            "\n9. Só mencione normalização textual ou fallback de datas se isso afetar a interpretação do resultado."
        )
    user_content = f"Pergunta: '{question}'\n\nDados retornados:\n{preview_text}\n\n"
    if profile and profile.get("profile_id") == "client_sales_profile_v1" and sql_text:
        sql_lower = sql_text.lower()
        notes = []
        if 'lower(trim(' in sql_lower:
            notes.append("A consulta normalizou texto com LOWER(TRIM(...)) para consolidar variações de escrita.")
        if '"criado em"' in sql_lower and '"data venda"' not in sql_lower:
            notes.append("A consulta usou Criado em como referência temporal.")
        elif '"data venda"' in sql_lower:
            notes.append("A consulta usou Data Venda como referência temporal principal.")
        if notes:
            user_content += "Notas da consulta:\n- " + "\n- ".join(notes) + "\n\n"
    if show_sql and sql_text:
        system_prompt += "\nAo final da resposta, inclua a query SQL usada em um bloco de código."
        user_content += f"SQL Executada:\n```sql\n{sql_text}\n```"
    else:
        system_prompt += "\nNUNCA inclua código SQL na resposta."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.2, max_completion_tokens=600)
    return resp.choices[0].message.content.strip()


def generate_suggestions(client, model_name, schema_text, question, profile=None):
    system_prompt = (
        "Sugira exatamente 3 perguntas úteis que o usuário pode fazer sobre esses dados. "
        "Seja breve - uma frase por sugestão. Formato:\n"
        "1. [pergunta]\n2. [pergunta]\n3. [pergunta]\n\n"
        "Não adicione introdução nem conclusão."
    )
    if profile and profile.get("profile_id") == "client_sales_profile_v1":
        system_prompt += (
            "\nPriorize perguntas úteis para uma base de vendas: período, produto, status, forma de pagamento, promotora, cupom e localização."
        )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user",
                                                                "content": f"Pergunta do usuário: '{question}'\n\nEsquema dos dados:\n{schema_text}"}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.5, max_completion_tokens=300)
    return resp.choices[0].message.content.strip()


def generate_schema_summary(client, model_name, schema_text, question, profile=None):
    system_prompt = (
        "Descreva esta fonte de dados em NO MÁXIMO 3 parágrafos curtos:\n"
        "1. Qual o tema principal.\n"
        "2. Quais são os campos ou tabelas mais importantes e seus propósitos.\n"
        "3. Que tipos de análises podem ser feitas.\n\n"
        "Seja direto e objetivo. Não use seções com títulos, apenas texto corrido."
    )
    if profile and profile.get("profile_id") == "client_sales_profile_v1":
        system_prompt += (
            "\nO contexto aponta para uma base operacional de vendas, pagamentos, status, cupons e localização."
        )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user",
                                                                "content": f"Pergunta do usuário: '{question}'\n\nEsquema:\n{schema_text}"}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.3, max_completion_tokens=500)
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
def format_table_details(table_name, schema_dict, profile=None):
    """Formata os detalhes de uma tabela específica em Markdown."""
    if table_name in schema_dict:
        columns = schema_dict[table_name]
        if not columns:
            return f"A tabela `{table_name}` existe, mas não foi possível ler suas colunas."

        header = f"### Detalhes da Tabela: `{table_name}`\n\n| Nome da Coluna | Tipo de Dado |\n|---|---|\n"
        rows = [f"| `{col[0]}` | `{col[1]}` |" for col in columns]
        response = header + "\n".join(rows)
        semantics = profile.get("column_semantics", {}) if profile else {}
        semantic_lines = []
        for column, description in semantics.items():
            if any(col_name == column for col_name, _ in columns):
                semantic_lines.append(f"- `{column}`: {description}")
        if semantic_lines:
            response += "\n\nObservações úteis:\n" + "\n".join(semantic_lines)
        return response
    else:
        return f"Desculpe, não encontrei uma tabela chamada `{table_name}`."


# --- FUNÇÕES DO DASHBOARD E UTILITÁRIOS GERAIS ---
def generate_chart_instructions(client, model_name, schema_text, question, db_mode, profile=None, df_preview=None):
    preview_text = df_preview.head(5).to_markdown(
        index=False) if df_preview is not None and not df_preview.empty else ""
    if db_mode:
        dialect_instructions = (
            "3. A consulta deve ser compatível com o dialeto PostgreSQL.\n"
            "4. Para agrupar por mês, use `to_char(\"nome_da_coluna_data\", 'YYYY-MM')`."
        )
    else:
        dialect_instructions = (
            "3. A consulta deve ser compatível com o dialeto SQLite.\n"
            "4. A tabela para consulta é SEMPRE `df`. Nunca use outro nome de tabela.\n"
            "5. Para agrupar por mês, use `strftime('%Y-%m', \"nome_da_coluna_data\")`."
        )
        if profile and profile.get("profile_id") == "client_sales_profile_v1":
            dialect_instructions += (
                "\n6. Se o pedido envolver valor/faturamento/receita, converta \"Valor Pago\" de texto monetário para número."
                "\n7. Para análises temporais, prefira \"Data Venda\"; use \"Criado em\" para cadastro/criação."
                "\n8. Para agrupar categorias textuais como promotora, bairro ou cidade, normalize com LOWER(TRIM(\"coluna\")) quando isso evitar duplicidades."
                "\n9. Em pedidos genéricos sobre vendas, considere status pago como padrão."
                "\n10. Para agrupar por mês em datas textuais dd/mm/aaaa hh:mm:ss, use substr(\"Data Venda\", 7, 4) || '-' || substr(\"Data Venda\", 4, 2)."
            )
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
    if profile and profile.get("profile_id") == "client_sales_profile_v1":
        system_content += (
            "\n4. Se o pedido for vago, priorize gráficos de vendas por período, produto, status, forma de pagamento, promotora, cupom, estado ou cidade."
        )
    messages = [{"role": "system", "content": system_content}, {"role": "user",
                                                                "content": f"Esquema:\n{schema_text}\n\nPreview:\n{preview_text}\n\nPedido:\n{question}"}]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=1.0 if 'o' in model_name else 0.0, max_completion_tokens=800)
    return resp.choices[0].message.content.strip()


def parse_chart_instructions(text):
    chart = {"sql": None, "type": "bar", "x": None, "y": None, "color": None, "title": "Gráfico"}
    current_field = None
    sql_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()

        if lower.startswith("sql:"):
            current_field = "sql"
            sql_value = raw_line.split(":", 1)[1].strip()
            if sql_value:
                sql_lines.append(sql_value)
        elif lower.startswith("tipo:"):
            current_field = None
            chart["type"] = raw_line.split(":", 1)[1].strip().lower()
        elif lower.startswith("x:"):
            current_field = None
            chart["x"] = raw_line.split(":", 1)[1].strip()
        elif lower.startswith("y:"):
            current_field = None
            chart["y"] = raw_line.split(":", 1)[1].strip()
        elif lower.startswith("color:"):
            current_field = None
            chart["color"] = raw_line.split(":", 1)[1].strip()
        elif lower.startswith("título:") or lower.startswith("titulo:"):
            current_field = None
            chart["title"] = raw_line.split(":", 1)[1].strip()
        elif current_field == "sql":
            sql_lines.append(raw_line.rstrip())

    if sql_lines:
        chart["sql"] = "\n".join(sql_lines).strip().strip("`")

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
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,23,0.22)",
            font={"family": "Plus Jakarta Sans, Segoe UI, sans-serif", "color": "#eef2f7"},
            title={"font": {"size": 22, "color": "#f5f7fb"}, "x": 0.02},
            margin={"l": 20, "r": 20, "t": 64, "b": 20},
            legend={
                "bgcolor": "rgba(13,17,23,0.34)",
                "bordercolor": "rgba(255,255,255,0.08)",
                "borderwidth": 1,
                "font": {"color": "#d7deea"},
            },
            xaxis={
                "gridcolor": "rgba(255,255,255,0.06)",
                "zerolinecolor": "rgba(255,255,255,0.07)",
            },
            yaxis={
                "gridcolor": "rgba(255,255,255,0.06)",
                "zerolinecolor": "rgba(255,255,255,0.07)",
            },
        )
        fig.update_traces(marker_line_color="rgba(255,255,255,0.16)", marker_line_width=0.6)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao gerar gráfico: {e}")


def validate_sql_tables(sql_text, db_mode, schema_dict=None):
    if not sql_text: return False, "Consulta SQL vazia.", sql_text
    
    if not db_mode:
        # Regex mais robusta para pegar nomes de tabela com espaços, aspas ou nomes simples
        sql_text = re.sub(r"(?i)\b(FROM|JOIN)\s+([\"'`][^\"'`]+[\"'`]|[\w]+)", r"\1 df", sql_text)
        # Garantia final: Se por algum motivo o regex falhou e não temos 'FROM df', forçamos uma substituição básica
        if "from df" not in sql_text.lower():
            sql_text = re.sub(r"(?i)\bFROM\s+[\s\w\"'`]+", "FROM df", sql_text)
        
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


def has_complete_db_credentials(credentials):
    required_keys = ("user", "password", "host", "port", "dbname")
    return isinstance(credentials, dict) and all(credentials.get(key) for key in required_keys)


# Verificação de Autenticação
if not st.session_state.get('logged_in', False):
    show_login_page()
else:
    render_app_hero("Análise Inteligente de Dados")
    show_logout_button()

    if 'config_loaded' not in st.session_state:
        st.session_state.config_loaded = True

    uploaded_file = None
    with st.sidebar:
        st.markdown(
            "<p class='ui-section-label' style='margin-bottom:0.5rem;'>Controle</p><h3 style='margin-top:0;'>Painel de Operação</h3>",
            unsafe_allow_html=True,
        )
        
        menu_selecionado = st.radio(
            "",
            ["Inteligência Artificial", "Conexão de Dados"],
            label_visibility="collapsed",
            key="sidebar_control_menu",
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
            st.markdown(
                "<p class='ui-section-label'>Assistente</p><h3 style='margin-top:0;'>Configurações da IA</h3>",
                unsafe_allow_html=True,
            )
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
            st.markdown(
                "<p class='ui-section-label'>Fonte</p><h3 style='margin-top:0;'>Conexão de Dados</h3>",
                unsafe_allow_html=True,
            )
            fonte_dados_index = 0 if fonte_dados == 'Banco de Dados' else 1
            novo_fonte = st.radio(
                "Origem:",
                ["Banco de Dados", "Arquivo CSV/Excel"],
                index=fonte_dados_index,
                label_visibility="collapsed",
            )
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
                                user = st.session_state.get("user", {})
                                save_db_config(user.get("user_id", ""), credentials_to_save)
                                st.session_state.db_creds = credentials_to_save;
                                st.session_state.fonte_dados = 'Banco de Dados';
                                st.success("Conexão salva!");
                                st.rerun()
                        else:
                            st.warning("Preencha todos os campos.")
            else:
                user_files = st.session_state.get("user_files", [])
                if user_files:
                    file_labels = ["Selecione um arquivo salvo"] + [
                        f"{file['name']} ({file.get('uploaded_at', '')[:19]})" for file in user_files
                    ]
                    default_index = 0
                    current_file_id = st.session_state.get("selected_file_id")
                    for idx, file in enumerate(user_files, start=1):
                        if file["file_id"] == current_file_id:
                            default_index = idx
                            break
                    selected_label = st.selectbox("Arquivos salvos", file_labels, index=default_index)
                    if selected_label == file_labels[0]:
                        st.session_state["selected_file_id"] = None
                    else:
                        selected_index = file_labels.index(selected_label) - 1
                        st.session_state["selected_file_id"] = user_files[selected_index]["file_id"]

                uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
                if uploaded_file is not None:
                    upload_signature = f"{uploaded_file.name}:{uploaded_file.size}"
                    if st.session_state.get("last_uploaded_file_signature") != upload_signature:
                        with st.spinner("Enviando arquivo para o Firebase..."):
                            saved_file = save_uploaded_file(uploaded_file)
                            st.session_state["last_uploaded_file_signature"] = upload_signature
                            st.session_state["selected_file_id"] = saved_file["file_id"]
                            st.success("Arquivo enviado e vinculado a sua conta.")
                            st.rerun()
    tab1, tab2, tab3 = st.tabs(["Conversa", "Dashboard", "Histórico"])

    with tab1:
        render_section_intro(
            "Conversa",
            "Converse com seus dados",
            "",
        )
        if not openai_api_key:
            st.warning("Insira a chave da API OpenAI para começar.")
        else:
            client = OpenAI(api_key=openai_api_key)
            db_mode = (fonte_dados == "Banco de Dados")
            schema_text = "";
            schema_dict = {};
            current_profile = {"profile_id": "database_profile_v1"}
            df_data = None;
            uri = None
            try:
                if db_mode:
                    if has_complete_db_credentials(st.session_state.get('db_creds')):
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
                    selected_file_id = st.session_state.get("selected_file_id")
                    user_files = st.session_state.get("user_files", [])
                    selected_file = next((file for file in user_files if file["file_id"] == selected_file_id), None)
                    selected_file_name = selected_file.get("name", "") if selected_file else ""

                    if selected_file is not None and st.session_state.get("loaded_file_id") != selected_file_id:
                        with st.spinner("Carregando arquivo salvo..."):
                            df_data = load_dataframe_from_storage(selected_file)
                            prompt_context, file_schema_dict, current_profile = build_file_context(df_data, selected_file_name)
                            st.session_state["df_data"] = df_data
                            st.session_state["loaded_file_id"] = selected_file_id
                            st.session_state["file_prompt_context"] = prompt_context
                            st.session_state["file_schema_dict"] = file_schema_dict
                            st.session_state["dataset_profile"] = current_profile
                        schema_text = prompt_context
                        schema_dict = file_schema_dict
                    elif selected_file is not None and 'df_data' in st.session_state:
                        df_data = st.session_state['df_data'];
                        current_profile = st.session_state.get("dataset_profile") or build_dataset_profile(df_data, selected_file_name)
                        schema_dict = st.session_state.get("file_schema_dict") or build_schema_dict_from_dataframe(df_data)
                        schema_text = st.session_state.get("file_prompt_context") or build_prompt_context(df_data, current_profile)
                    elif 'df_data' in st.session_state and st.session_state.get("loaded_file_id"):
                        df_data = st.session_state['df_data'];
                        current_profile = st.session_state.get("dataset_profile") or build_dataset_profile(df_data, selected_file_name)
                        schema_dict = st.session_state.get("file_schema_dict") or build_schema_dict_from_dataframe(df_data)
                        schema_text = st.session_state.get("file_prompt_context") or build_prompt_context(df_data, current_profile)
                    else:
                        st.info("Faça upload de um arquivo ou selecione um salvo para começar.")
            except Exception as e:
                st.error(f"Falha ao carregar fonte de dados: {e}")

            if 'history' not in st.session_state: st.session_state.history = []
            for message in st.session_state.history:
                av = AVATAR_USER if message["role"] == "user" else AVATAR_BOT
                with st.chat_message(message["role"], avatar=av): st.markdown(message["content"])

            st.markdown("<div class='ui-chat-composer-spacer'></div>", unsafe_allow_html=True)
            prompt = None
            with st.form("chat_prompt_form", clear_on_submit=True):
                prompt_col, submit_col = st.columns([24, 1])
                with prompt_col:
                    prompt_value = st.text_input(
                        "Pergunte algo sobre seus dados...",
                        label_visibility="collapsed",
                        placeholder="Pergunte algo sobre seus dados...",
                    )
                with submit_col:
                    submitted = st.form_submit_button("↑")
                if submitted and prompt_value.strip():
                    prompt = prompt_value.strip()

            if prompt:
                append_chat_message("user", prompt)
                with st.chat_message("user", avatar=AVATAR_USER):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar=AVATAR_BOT):
                    with st.spinner("Analisando..."):
                        if (db_mode and not schema_text) or (not db_mode and df_data is None):
                            st.warning("Fonte de dados não configurada.");
                            st.stop()

                        history_text = build_history_text(st.session_state.history, max_turns=6)
                        intent = classify_intent(client, modelo_selecionado, prompt, history_text, schema_text)
                        final_text = "";
                        response_handled_internally = False

                        if 'schema_summary' in intent:
                            final_text = generate_schema_summary(client, modelo_selecionado, schema_text, prompt, current_profile)
                        elif 'schema_info' in intent:
                            current_schema_dict = st.session_state.get('schema_dict', {}) if db_mode else schema_dict
                            table_name = extract_table_name(client, modelo_selecionado, prompt, current_schema_dict)
                            if table_name != 'none':
                                final_text = format_table_details(table_name, current_schema_dict, current_profile)
                            else:
                                table_list = "\n".join([f"- `{t}`" for t in current_schema_dict.keys()])
                                final_text = f"Aqui estão as tabelas/arquivos que encontrei:\n\n{table_list}"
                        elif 'suggestion_request' in intent:
                            final_text = generate_suggestions(client, modelo_selecionado, schema_text, prompt, current_profile)
                        else:  # intent == 'data_query'
                            system_prompt = build_sql_system_prompt(db_mode, current_profile)
                            sql_candidate = generate_sql(client, modelo_selecionado, system_prompt, schema_text,
                                                         history_text, prompt)
                            sql_text = extract_sql(sql_candidate)
                            if not sql_text or "NO_SQL" in sql_candidate.upper():
                                final_text = generate_suggestions(client, modelo_selecionado, schema_text, prompt, current_profile)
                            else:
                                schema_dict_ctx = st.session_state.get('schema_dict', {}) if db_mode else schema_dict
                                ok, vmsg, sql_text = validate_sql_tables(sql_text, db_mode, schema_dict_ctx)
                                if not ok:
                                    final_text = f"Não consegui gerar uma consulta válida para sua pergunta. Tente reformular de forma mais específica."
                                else:
                                    # Mecanismo de retry: tenta executar, se falhar, pede correção à IA
                                    max_retries = 2
                                    sql_succeeded = False
                                    df_result = None
                                    last_error = None
                                    for attempt in range(max_retries):
                                        try:
                                            if db_mode:
                                                engine = create_engine(uri)
                                                df_result = pd.read_sql(text(sql_text), engine)
                                            else:
                                                df_result = ps.sqldf(sql_text, {"df": df_data})
                                            sql_succeeded = True
                                            break
                                        except Exception as e:
                                            last_error = str(e)
                                            if attempt < max_retries - 1:
                                                fix_prompt = build_sql_fix_prompt(str(e), sql_text, prompt, db_mode, current_profile)
                                                sql_candidate = generate_sql(client, modelo_selecionado, system_prompt,
                                                                             schema_text, "", fix_prompt)
                                                sql_text = extract_sql(sql_candidate)
                                                ok, vmsg, sql_text = validate_sql_tables(sql_text, db_mode, schema_dict_ctx)
                                                if not ok:
                                                    sql_succeeded = False
                                                    break
                                            else:
                                                sql_succeeded = False

                                    if sql_succeeded and df_result is not None:
                                        final_text = generate_humanized_answer(
                                            client,
                                            modelo_selecionado,
                                            prompt,
                                            df_result,
                                            profile=current_profile,
                                            sql_text=sql_text,
                                            show_sql=db_mode,
                                        )
                                        st.markdown(final_text)
                                        if not df_result.empty:
                                            st.markdown(
                                                "<p class='ui-section-label'>Resultado</p><h3 style='margin-top:0;'>Preview dos dados retornados</h3>",
                                                unsafe_allow_html=True,
                                            )
                                            st.dataframe(df_result.head(100))
                                        append_chat_message("assistant", final_text)
                                        response_handled_internally = True
                                    else:
                                        final_text = "Desculpe, não consegui processar essa consulta. Tente reformular sua pergunta de forma mais específica."
                                        if last_error:
                                            st.caption(f"Detalhe técnico: {last_error}")
                        if not response_handled_internally:
                            st.markdown(final_text)
                            append_chat_message("assistant", final_text)

    with tab2:
        render_section_intro(
            "DATA ANALYTICS",
            "",
            "",
        )
        if 'charts' not in st.session_state: st.session_state.charts = []
        db_mode_dash = (fonte_dados == "Banco de Dados")
        file_profile_ctx = st.session_state.get("dataset_profile", {"profile_id": "generic_file_v1"})
        file_df_ctx = st.session_state.get('df_data', pd.DataFrame())
        schema_text_ctx = st.session_state.get('schema_text', '') if db_mode_dash else st.session_state.get(
            "file_prompt_context",
            build_prompt_context(file_df_ctx, file_profile_ctx),
        )
        schema_dict_ctx = st.session_state.get('schema_dict') if db_mode_dash else st.session_state.get(
            "file_schema_dict",
            build_schema_dict_from_dataframe(file_df_ctx),
        )
        df_data_ctx = st.session_state.get('df_data') if not db_mode_dash else None
        db_creds_dash = st.session_state.get("db_creds", {})
        uri_ctx = None
        if db_mode_dash and has_complete_db_credentials(db_creds_dash):
            uri_ctx = (
                f"postgresql+psycopg2://{db_creds_dash['user']}:{db_creds_dash['password']}"
                f"@{db_creds_dash['host']}:{db_creds_dash['port']}/{db_creds_dash['dbname']}"
            )
        for idx, chart in enumerate(st.session_state.charts):
            st.markdown(
                f"<p class='ui-section-label'>Visualização {idx + 1}</p><h3 style='margin-top:0;'>{chart.get('title', f'Gráfico {idx + 1}')}</h3>",
                unsafe_allow_html=True,
            )
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
                                                                             file_profile_ctx, df_data_ctx)
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
        st.markdown("<h3 style='margin-top:0;'>Adicionar novo gráfico</h3>", unsafe_allow_html=True)
        new_chart_prompt = st.text_input("Descreva o gráfico que você deseja criar:", key="new_chart_prompt_input")
        if st.button("Gerar novo gráfico", key="new_chart_btn"):
            if new_chart_prompt.strip() and openai_api_key:
                with st.spinner("Gerando gráfico..."):
                    try:
                        client = OpenAI(api_key=openai_api_key)
                        instr_text = generate_chart_instructions(client, modelo_selecionado, schema_text_ctx,
                                                                 new_chart_prompt, db_mode_dash, file_profile_ctx, df_data_ctx)
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
        render_section_intro(
            "Memória",
            "Histórico de conversas",
            "Revise perguntas e respostas em um layout mais organizado, com melhor separação visual e leitura mais confortável.",
        )
        if 'history' not in st.session_state or not st.session_state.history:
            st.info("Nenhuma conversa registrada ainda.")
        else:
            for i, msg in enumerate(st.session_state.history):
                role = "Usuário" if msg["role"] == "user" else "Assistente"
                with st.expander(f"{role} - Mensagem {i + 1}"): st.markdown(msg["content"])
            if st.button("Limpar Histórico"):
                clear_chat_history()
                st.rerun()
