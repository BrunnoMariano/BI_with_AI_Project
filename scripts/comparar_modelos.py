"""Compara os modelos da OpenAI do seletor do app: teste de fumaça + qualidade, custo e tempo.

Uso (na raiz do projeto, com a planilha em data/vendas_cliente.xlsx):

    $env:OPENAI_API_KEY = "sk-..."          # PowerShell (a chave nunca é impressa)
    python scripts/comparar_modelos.py      # todos os modelos de ml/llm_utils.MODELOS
    python scripts/comparar_modelos.py --modelos gpt-6.1-terra,gpt-6-luna
    python scripts/comparar_modelos.py --so-fumaca

O que é medido, por modelo:
- fumaça: 1 chamada mínima com os parâmetros de chat_params (confirma que a API aceita);
- intenção: classify_intent acerta a categoria de 4 perguntas;
- SQL: generate_sql + pandasql respondem 5 perguntas com gabarito calculado da planilha
  (primeira tentativa, sem o "retry" do app; só números são comparados, nunca linhas — LGPD);
- previsão: explain_forecast responde 2 perguntas e TODO número citado precisa existir no
  JSON de contexto_previsao (fidelidade: número inventado = falha);
- tokens (entrada, saída, raciocínio), custo estimado em US$ e tempo médio por chamada.
Resultado impresso e salvo em data/comparacao_modelos.csv (pasta fora do Git).

As funções do chat moram em fonte/appLAB.py, que não pode ser importado (ele roda o app).
Por isso o script copia só as definições necessárias, lendo o arquivo com `ast`.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pandasql as ps

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "fonte"))

from ml.explain import contexto_previsao, explain_forecast, parece_pergunta_de_previsao, tabela_previsao  # noqa: E402
from ml.llm_utils import MODELOS, chat_params  # noqa: E402
from ml.pipeline import run_predictive_pipeline  # noqa: E402
from ml.preprocessing import parse_brl  # noqa: E402
from ml.schema import detect_dataset_profile  # noqa: E402

# US$ por 1M tokens (entrada, saída), tier padrão — developers.openai.com/api/docs/pricing, 07/11/2026.
PRECOS = {
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.4-mini": (0.75, 4.50),
}
# Números de conferência do CLAUDE.md (só para avisar se a planilha mudou).
CONFERENCIA = {"total_pedidos": 4558, "pedidos_pagos": 2919, "receita_paga": 308783.48, "receita_b2b": 88410.08}

FUNCOES_CHAT = (
    "_series_examples", "_top_value_summary", "_count_case_variants", "describe_dataframe_schema",
    "build_schema_dict_from_dataframe", "build_dataset_profile", "build_prompt_context", "build_file_context",
    "build_sql_system_prompt", "extract_sql", "validate_sql_tables", "classify_intent", "generate_sql",
)
CONSTANTES_CHAT = ("INTENT_CATEGORIES",)

PERGUNTAS_INTENCAO = (
    ("quanto vou vender mês que vem?", "forecast_query"),
    ("quais colunas tem?", "schema_info"),
    ("olá", "suggestion_request"),
    ("quanto vendeu em janeiro de 2026?", "data_query"),
)
PERGUNTAS_SQL = (
    ("quantos pedidos existem na base no total, considerando todos os status?", "total_pedidos"),
    ("quantas vendas foram pagas?", "pedidos_pagos"),
    ("qual a receita total paga?", "receita_paga"),
    ("quanto somam os pedidos pagos de vendas corporativas (B2B), ou seja, com a coluna Conta preenchida?", "receita_b2b"),
    ("qual produto teve mais vendas pagas?", "produto_mais_vendido"),
)
PERGUNTAS_PREVISAO = (
    "quanto vou faturar nos próximos 28 dias?",
    "vamos bater a meta de R$ 50.000 nos próximos 28 dias?",
)
# Números "de texto" que a resposta pode citar sem vir do JSON (níveis das faixas, horizontes...).
NUMEROS_LIVRES = {80.0, 95.0, 7.0, 28.0, 90.0, 100.0}


# ---------- funções do chat (copiadas do appLAB.py via ast) ----------

def carregar_funcoes_chat(caminho_app: Path = RAIZ / "fonte" / "appLAB.py") -> dict:
    """Executa só as definições listadas em FUNCOES_CHAT/CONSTANTES_CHAT e devolve o namespace."""
    arvore = ast.parse(Path(caminho_app).read_text(encoding="utf-8"))
    nos = []
    for no in arvore.body:
        if isinstance(no, ast.FunctionDef) and no.name in FUNCOES_CHAT:
            nos.append(no)
        elif isinstance(no, ast.Assign) and any(getattr(t, "id", None) in CONSTANTES_CHAT for t in no.targets):
            nos.append(no)
    ns = {
        "re": re, "pd": pd, "detect_dataset_profile": detect_dataset_profile,
        "chat_params": chat_params, "parece_pergunta_de_previsao": parece_pergunta_de_previsao,
    }
    exec(compile(ast.Module(body=nos, type_ignores=[]), str(caminho_app), "exec"), ns)
    faltando = [nome for nome in FUNCOES_CHAT + CONSTANTES_CHAT if nome not in ns]
    if faltando:
        raise RuntimeError(f"Não achei no appLAB.py: {', '.join(faltando)}")
    return ns


# ---------- gabarito ----------

def calcular_gabarito(df: pd.DataFrame) -> dict:
    """Respostas certas das PERGUNTAS_SQL, calculadas com pandas a partir da planilha."""
    status = df["Status Venda"].astype(str).str.strip().str.lower()
    pagos = df.loc[status == "pago"]
    valores = pagos["Valor Pago"].map(parse_brl).fillna(0.0)
    conta = pagos["Conta"] if "Conta" in pagos else pd.Series(index=pagos.index, dtype=object)
    b2b = conta.notna() & conta.astype(str).str.strip().ne("")
    produtos = pagos["Produto"].dropna().astype(str).str.strip()
    return {
        "total_pedidos": int(len(df)),
        "pedidos_pagos": int(len(pagos)),
        "receita_paga": round(float(valores.sum()), 2),
        "receita_b2b": round(float(valores[b2b].sum()), 2),
        "produto_mais_vendido": produtos.value_counts().idxmax(),
    }


def resultado_contem(df_resultado: pd.DataFrame | None, esperado) -> bool:
    """True se alguma célula do resultado do SQL bate com o gabarito (número ±0,01 ou texto igual)."""
    if df_resultado is None or df_resultado.empty:
        return False
    for valor in df_resultado.to_numpy().ravel():
        if isinstance(esperado, str):
            if isinstance(valor, str) and valor.strip().lower() == esperado.strip().lower():
                return True
            continue
        try:
            if abs(float(valor) - float(esperado)) <= 0.01:
                return True
        except (TypeError, ValueError):
            continue
    return False


# ---------- fidelidade numérica das respostas de previsão ----------

_DATAS = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
_NUMERO = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(\s*mil\b)?")


def numeros_citados(texto: str) -> list[float]:
    """Números de um texto em pt-BR ("R$ 23.935,27" -> 23935.27; "24 mil" -> 24000), sem datas."""
    numeros = []
    for inteiro, decimal, mil in _NUMERO.findall(_DATAS.sub(" ", str(texto))):
        valor = float(inteiro.replace(".", "") + ("." + decimal if decimal else ""))
        numeros.append(valor * 1000 if mil else valor)
    return numeros


def numeros_do_json(objeto) -> list[float]:
    """Todos os números de um JSON (inclusive os escritos dentro de textos)."""
    if isinstance(objeto, bool) or objeto is None:
        return []
    if isinstance(objeto, (int, float)):
        return [float(objeto)]
    if isinstance(objeto, str):
        return numeros_citados(objeto)
    itens = objeto.values() if isinstance(objeto, dict) else objeto
    return [n for item in itens for n in numeros_do_json(item)]


def numeros_inventados(resposta: str, contexto: dict, pergunta: str = "") -> list[float]:
    """Números da resposta que não estão no JSON nem na pergunta (tolerância: 0,5 ou 1%)."""
    permitidos = numeros_do_json(contexto) + numeros_citados(pergunta) + sorted(NUMEROS_LIVRES)
    inventados = []
    for numero in numeros_citados(resposta):
        if numero <= 10 and numero == int(numero):
            continue  # "8 de cada 10", "3 frases", "4 cortes"...
        if not any(abs(numero - v) <= max(0.5, 0.01 * abs(v)) for v in permitidos):
            inventados.append(numero)
    return inventados


# ---------- cliente que mede tokens e tempo ----------

class ClienteMedido:
    """Repassa chat.completions.create ao cliente real e registra tokens, tempo e erros."""

    def __init__(self, cliente):
        self._cliente = cliente
        self.registros: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        inicio = time.perf_counter()
        registro = {"modelo": kwargs.get("model"), "entrada": 0, "saida": 0, "raciocinio": 0, "erro": None}
        try:
            resp = self._cliente.chat.completions.create(**kwargs)
        except Exception as erro:
            registro["erro"] = f"{type(erro).__name__}: {erro}"[:300]
            raise
        finally:
            registro["segundos"] = time.perf_counter() - inicio
            self.registros.append(registro)
        uso = getattr(resp, "usage", None)
        detalhes = getattr(uso, "completion_tokens_details", None)
        registro["entrada"] = getattr(uso, "prompt_tokens", 0) or 0
        registro["saida"] = getattr(uso, "completion_tokens", 0) or 0
        registro["raciocinio"] = getattr(detalhes, "reasoning_tokens", 0) or 0
        return resp

    def resumo(self, modelo: str) -> dict:
        regs = [r for r in self.registros if r["modelo"] == modelo]
        preco_entrada, preco_saida = PRECOS.get(modelo, (float("nan"), float("nan")))
        entrada, saida = sum(r["entrada"] for r in regs), sum(r["saida"] for r in regs)
        return {
            "chamadas": len(regs),
            "erros_api": sum(1 for r in regs if r["erro"]),
            "tokens_entrada": entrada,
            "tokens_saida": saida,
            "tokens_raciocinio": sum(r["raciocinio"] for r in regs),
            "custo_usd": round((entrada * preco_entrada + saida * preco_saida) / 1e6, 5),
            "tempo_medio_s": round(sum(r["segundos"] for r in regs) / len(regs), 2) if regs else None,
        }


# ---------- etapas ----------

def teste_fumaca(cliente, modelos) -> list[dict]:
    """Uma chamada mínima por modelo; ok=False mostra o erro da API (ex.: parâmetro recusado)."""
    resultados = []
    for modelo in modelos:
        try:
            resp = cliente.chat.completions.create(
                model=modelo, messages=[{"role": "user", "content": "Responda apenas: OK"}], **chat_params(modelo, 0.0, 20)
            )
            texto = (resp.choices[0].message.content or "").strip()
            resultados.append({"modelo": modelo, "ok": bool(texto), "detalhe": texto[:40] or "resposta vazia"})
        except Exception as erro:
            resultados.append({"modelo": modelo, "ok": False, "detalhe": f"{type(erro).__name__}: {erro}"[:300]})
    return resultados


def avaliar_modelo(cliente, modelo: str, df: pd.DataFrame, ns: dict, resultado_previsao: dict, gabarito: dict) -> dict:
    """Roda as perguntas de intenção, SQL e previsão com um modelo e devolve os acertos."""
    schema_text, schema_dict, perfil = ns["build_file_context"](df, "vendas_cliente.xlsx")
    contexto = contexto_previsao(resultado_previsao)
    tabela = tabela_previsao(resultado_previsao)
    detalhes = []

    acertos_intencao = 0
    for pergunta, esperado in PERGUNTAS_INTENCAO:
        intencao = ns["classify_intent"](cliente, modelo, pergunta, "", schema_text)
        acertou = esperado in intencao
        acertos_intencao += acertou
        detalhes.append(f"intenção {'OK ' if acertou else 'ERRO'} {pergunta!r} -> {intencao!r}")

    acertos_sql = 0
    sistema = ns["build_sql_system_prompt"](False, perfil)
    for pergunta, chave in PERGUNTAS_SQL:
        try:
            sql = ns["extract_sql"](ns["generate_sql"](cliente, modelo, sistema, schema_text, "", pergunta))
            valido, mensagem, sql = ns["validate_sql_tables"](sql, False, schema_dict)
            if not valido:
                raise ValueError(mensagem)
            acertou = resultado_contem(ps.sqldf(sql, {"df": df, "previsao": tabela}), gabarito[chave])
            motivo = "" if acertou else "resultado diferente do gabarito"
        except Exception as erro:
            acertou, motivo = False, f"{type(erro).__name__}: {str(erro)[:120]}"
        acertos_sql += acertou
        detalhes.append(f"SQL      {'OK ' if acertou else 'ERRO'} {pergunta!r} {motivo}")

    respostas_fieis = 0
    for pergunta in PERGUNTAS_PREVISAO:
        try:
            resposta = explain_forecast(cliente, modelo, pergunta, resultado_previsao)
            inventados = numeros_inventados(resposta, contexto, pergunta)
            fiel = bool(numeros_citados(resposta)) and not inventados
            motivo = f"números fora do JSON: {inventados}" if inventados else ""
        except Exception as erro:
            fiel, motivo = False, f"{type(erro).__name__}: {str(erro)[:120]}"
        respostas_fieis += fiel
        detalhes.append(f"previsão {'OK ' if fiel else 'ERRO'} {pergunta!r} {motivo}")

    linha = {
        "modelo": modelo,
        "intencao_acertos": f"{acertos_intencao}/{len(PERGUNTAS_INTENCAO)}",
        "sql_acertos": f"{acertos_sql}/{len(PERGUNTAS_SQL)}",
        "previsao_fieis": f"{respostas_fieis}/{len(PERGUNTAS_PREVISAO)}",
    }
    if isinstance(cliente, ClienteMedido):
        linha.update(cliente.resumo(modelo))
    return {"linha": linha, "detalhes": detalhes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compara os modelos da OpenAI do app.")
    parser.add_argument("--modelos", default=",".join(MODELOS), help="IDs separados por vírgula")
    parser.add_argument("--planilha", default=str(RAIZ / "data" / "vendas_cliente.xlsx"))
    parser.add_argument("--saida", default=str(RAIZ / "data" / "comparacao_modelos.csv"))
    parser.add_argument("--so-fumaca", action="store_true", help="só testa se cada modelo responde")
    args = parser.parse_args(argv)
    modelos = [m.strip() for m in args.modelos.split(",") if m.strip()]

    chave = os.environ.get("OPENAI_API_KEY", "").strip()
    if not chave:
        print('Defina a chave antes: $env:OPENAI_API_KEY = "sk-..." (PowerShell). Ela não é impressa.')
        return 2
    from openai import OpenAI

    cliente = ClienteMedido(OpenAI(api_key=chave))

    print("== Teste de fumaça ==")
    fumaca = teste_fumaca(cliente, modelos)
    for item in fumaca:
        print(f"  {'OK  ' if item['ok'] else 'FALHA'} {item['modelo']:<16} {item['detalhe']}")
    if not all(item["ok"] for item in fumaca):
        print("Algum modelo falhou: corrija ml/llm_utils.MODELOS antes de comparar a qualidade.")
        return 1
    if args.so_fumaca:
        return 0

    print("\n== Preparando dados e previsão ==")
    df = pd.read_excel(args.planilha, sheet_name="Worksheet")
    gabarito = calcular_gabarito(df)
    for chave_conf, valor in CONFERENCIA.items():
        if abs(gabarito[chave_conf] - valor) > 0.01:
            print(f"  aviso: {chave_conf} = {gabarito[chave_conf]} (CLAUDE.md diz {valor})")
    ns = carregar_funcoes_chat()
    resultado_previsao = run_predictive_pipeline(df, "comparacao", horizonte_previsao=90)

    linhas = []
    for modelo in modelos:
        print(f"\n== {modelo} ==")
        avaliacao = avaliar_modelo(cliente, modelo, df, ns, resultado_previsao, gabarito)
        print("\n".join("  " + d for d in avaliacao["detalhes"]))
        linhas.append(avaliacao["linha"])

    tabela = pd.DataFrame(linhas)
    print("\n== Resumo ==")
    print(tabela.to_string(index=False))
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(args.saida, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"\nSalvo em {args.saida}. Custo total estimado: US$ {tabela['custo_usd'].sum():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
