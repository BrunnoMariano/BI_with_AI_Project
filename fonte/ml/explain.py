"""Camada de IA da previsão (Fase 4): "o ML calcula, o LLM explica" (decisão 6 do CLAUDE.md).

- contexto_previsao: resume o resultado do pipeline num JSON pequeno (totais, faixas,
  modelo vencedor e métricas). É a ÚNICA coisa sobre a previsão que o LLM recebe.
- explain_forecast / gerar_resumo_executivo: pedem ao LLM que descreva esses números.
- tabela_previsao: previsão diária como tabela "previsao" para o pandasql do chat.

Nada aqui importa Streamlit nem OpenAI: o cliente da OpenAI é recebido como parâmetro,
então os testes usam um cliente falso e rodam offline.
"""

from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import pandas as pd

from ml.forecasting import agregar_previsao, total_com_intervalo
from ml.llm_utils import chat_params, usa_raciocinio

HORIZONTE_VALIDADO = 28
HORIZONTES_RESUMO = (7, 28, 90)
NOMES_MODELOS = {
    "holt_winters": "Holt-Winters",
    "lightgbm": "LightGBM",
    "baseline_sazonal_7d": "Baseline sazonal (7 dias)",
    "media_movel_28d": "Média móvel (28 dias)",
}
MESES = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
         "agosto", "setembro", "outubro", "novembro", "dezembro")
UNIDADES = {"receita": "R$", "pedidos": "pedidos pagos"}
# Modelos de raciocínio gastam tokens "pensando"; com limite baixo a resposta vem vazia.
MAX_TOKENS_RESPOSTA = 800
MAX_TOKENS_RACIOCINIO = 4000

# Perguntas sobre o futuro (usado só como plano B quando o classificador do LLM falha).
_PADROES_PREVISAO = (
    r"\bprevis", r"\bprever\b", r"\bpreve\b", r"\bprojec", r"\bprojet",
    r"\b(mes|semana|ano|trimestre) que vem\b", r"\bproxim[oa]s? (\d+ )?(mes|meses|semana|semanas|dias|trimestre)",
    r"\b(vou|vamos|vai|iremos|irei) (vender|faturar|receber)", r"\bvender(emos|ei|a)\b", r"\bfaturar(emos|ei|a)\b",
    r"\b(bater|atingir|alcancar|cumprir) a meta\b", r"\bmeta\b", r"\bfuturo\b",
)


# ---------- utilidades ----------

def _sem_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto or "").lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def parece_pergunta_de_previsao(pergunta: str) -> bool:
    """True se a pergunta fala do futuro (ex.: "quanto vou vender mês que vem?")."""
    texto = _sem_acentos(pergunta)
    return any(re.search(padrao, texto) for padrao in _PADROES_PREVISAO)


def resultado_tem_previsao(resultado) -> bool:
    """True se `resultado` é uma saída válida de run_predictive_pipeline (não erro/aviso)."""
    return isinstance(resultado, dict) and "previsoes" in resultado and "metadados" in resultado


def _data_br(valor) -> str:
    return pd.Timestamp(valor).strftime("%d/%m/%Y")


def _arredondar(valor, alvo: str):
    if valor is None or not np.isfinite(valor):
        return None
    return round(float(valor), 2) if alvo == "receita" else int(round(float(valor)))


def _pct(fracao):
    return None if fracao is None or not np.isfinite(fracao) else round(float(fracao) * 100, 1)


def _faixas(total: dict, alvo: str) -> dict:
    faixas = {
        f"faixa_{nivel}": {
            "minimo": _arredondar(total[f"lim_inf_{nivel}"], alvo),
            "maximo": _arredondar(total[f"lim_sup_{nivel}"], alvo),
        }
        for nivel in ("80", "95")
    }
    # As faixas usam os erros reais do backtest; se o modelo errou sempre para o mesmo
    # lado, a faixa fica deslocada e o valor previsto pode ficar fora dela.
    previsto, inferior, superior = total["previsto"], total["lim_inf_80"], total["lim_sup_80"]
    if np.isfinite(superior) and previsto > superior:
        faixas["observacao_faixa"] = (
            "o previsto ficou acima da faixa de 80% porque no backtest o modelo superestimou; "
            "o valor real tende a ficar abaixo do previsto, dentro da faixa"
        )
    elif np.isfinite(inferior) and previsto < inferior:
        faixas["observacao_faixa"] = (
            "o previsto ficou abaixo da faixa de 80% porque no backtest o modelo subestimou; "
            "o valor real tende a ficar acima do previsto, dentro da faixa"
        )
    return faixas


# ---------- JSON enviado ao LLM ----------

def _resumo_alvo(resultado: dict, alvo: str) -> dict:
    previsao = resultado["previsoes"][alvo]
    residuos = resultado["residuos"][alvo]
    meta = resultado["metadados"]["alvos"][alvo]
    tabela = resultado["modelos"][alvo]
    historico = resultado["historico"][alvo]
    serie = pd.Series(historico["real"].to_numpy(dtype=float), index=pd.DatetimeIndex(historico["data"]))
    vencedor = meta["modelo_vencedor"]
    linha = tabela.loc[tabela["modelo"] == vencedor].iloc[0]
    inicio = pd.Timestamp(previsao["data"].iloc[0])

    horizontes = {}
    for dias in HORIZONTES_RESUMO:
        if dias > len(previsao):
            continue
        total = total_com_intervalo(previsao, residuos, dias)
        bloco = {
            "de": _data_br(previsao["data"].iloc[0]),
            "ate": _data_br(previsao["data"].iloc[dias - 1]),
            "previsto": _arredondar(total["previsto"], alvo),
            **_faixas(total, alvo),
            "validado_no_backtest": dias <= HORIZONTE_VALIDADO,
        }
        if dias <= len(serie):
            anterior = float(serie.iloc[-dias:].sum())
            bloco[f"realizado_nos_{dias}_dias_anteriores"] = _arredondar(anterior, alvo)
            bloco["variacao_vs_periodo_anterior_pct"] = _pct((total["previsto"] - anterior) / anterior) if anterior else None
        horizontes[f"proximos_{dias}_dias"] = bloco

    limite_validado = inicio + pd.Timedelta(days=HORIZONTE_VALIDADO - 1)
    ultimo_dia = pd.Timestamp(previsao["data"].iloc[-1])
    meses = []
    for _, mes in agregar_previsao(previsao, residuos, "mes", serie).iterrows():
        periodo = pd.Timestamp(mes["periodo"])
        fim_mes = min(periodo + pd.offsets.MonthEnd(0), ultimo_dia)
        item = {
            "mes": f"{MESES[periodo.month - 1]}/{periodo.year}",
            "previsto": _arredondar(mes["previsto"], alvo),
            **_faixas(mes, alvo),
            "mes_completo": not bool(mes["parcial"]),
            "validado_no_backtest": bool(fim_mes <= limite_validado),
        }
        if mes["parcial"]:
            item["observacao"] = f"cobre só {int(mes['n_dias'])} dias do mês (previsão vai até {_data_br(ultimo_dia)})"
        if mes["realizado"] > 0:
            item["ja_realizado_no_mes"] = _arredondar(mes["realizado"], alvo)
            item["observacao_realizado"] = "o total do mês soma o já realizado com o previsto para os dias restantes"
        meses.append(item)

    return {
        "unidade": UNIDADES.get(alvo, alvo),
        "modelo_vencedor": NOMES_MODELOS.get(vencedor, vencedor),
        "metricas_backtest_do_vencedor": {
            "wape_pct": _pct(linha["wape"]),
            "mae_por_dia": _arredondar(linha["mae"], "receita"),
            "rmse_por_dia": _arredondar(linha["rmse"], "receita"),
            "erro_no_total_pct": _pct(linha["erro_total"]),
            "cobertura_faixa_80_pct": _pct(linha["cobertura_80"]),
        },
        "comparacao_modelos_wape_pct": {NOMES_MODELOS.get(m, m): _pct(w) for m, w in zip(tabela["modelo"], tabela["wape"])},
        **horizontes,
        "por_mes": meses,
    }


def contexto_previsao(resultado: dict) -> dict:
    """Resumo agregado da previsão (sem linhas da planilha) para enviar ao LLM.

    Contém, por alvo (receita e pedidos): totais previstos de 7/28/90 dias e por mês,
    faixas de 80% e 95%, modelo vencedor e métricas médias do backtest.
    """
    meta = resultado["metadados"]
    receita = meta["alvos"]["receita"]
    contexto = {
        "escopo": "Vendas B2C pagas (pedidos sem empresa compradora). Vendas corporativas (B2B) ficam fora da previsão.",
        "data_de_referencia": "data de criação do pedido",
        "treino": {"de": _data_br(receita["inicio_treino"]), "ate": _data_br(receita["fim_treino"]), "dias": receita["n_obs"]},
        "hoje_considerado": _data_br(receita["inicio_previsao"]),
        "explicacao_hoje": "Os dados vão até o dia anterior; a previsão começa nesta data. Use-a como 'hoje' para interpretar 'mês que vem', 'próxima semana' etc.",
        "validacao": (
            f"Backtest temporal com {meta['n_cortes']} cortes de {meta['horizonte']} dias: o modelo treina só com o passado "
            "de cada corte e é comparado com o real. Métricas = média dos cortes. WAPE = erro somado ÷ total real."
        ),
        "como_ler_as_faixas": (
            "faixa_80: em 8 de cada 10 cenários o valor real fica entre mínimo e máximo; faixa_95: em 95 de cada 100. "
            "Acima de 28 dias o modelo não foi validado e as faixas tendem a ser otimistas."
        ),
        "alvos": {alvo: _resumo_alvo(resultado, alvo) for alvo in resultado["previsoes"]},
    }
    b2b = resultado.get("b2b", {}).get("janela")
    if b2b:
        contexto["b2b_fora_da_previsao"] = {
            "media_mensal_receita_historica": round(float(b2b["media_mensal_receita"]), 2),
            "media_mensal_pedidos_historica": round(float(b2b["media_mensal_pedidos"]), 1),
            "observacao": "Só histórico (média desde o início do treino); não é previsão e não deve ser somado aos números previstos.",
        }
    return contexto


# ---------- chamadas ao LLM ----------

REGRAS_NUMEROS = (
    "REGRAS OBRIGATÓRIAS:\n"
    "1. Use SOMENTE os números do JSON. Nunca calcule, some, extrapole ou estime valores novos.\n"
    "2. Sempre que citar um valor previsto, cite também a faixa de 80% (mínimo a máximo) do mesmo período.\n"
    "3. Escreva valores em reais no formato brasileiro (ex.: R$ 12.345,67) e percentuais com vírgula (ex.: 51,2%).\n"
    "4. Deixe claro que é uma previsão de vendas B2C pagas; B2B fica fora.\n"
    "5. Se o período perguntado não estiver no JSON, diga que essa previsão não está disponível e indique a aba Previsões.\n"
    "6. Se o período não foi validado no backtest (validado_no_backtest = false), avise que a incerteza é maior.\n"
    "7. Se houver observacao_faixa, explique-a em linguagem simples (o valor mais provável é o da faixa).\n"
)


def _chamar_llm(client, model: str, system_prompt: str, user_content: str, temperatura: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        **chat_params(model, temperatura, MAX_TOKENS_RACIOCINIO if usa_raciocinio(model) else MAX_TOKENS_RESPOSTA),
    )
    texto = (resp.choices[0].message.content or "").strip()
    if not texto:
        raise RuntimeError("O modelo não devolveu texto. Tente de novo ou escolha outro modelo.")
    return texto


def _json_previsao(resultado: dict) -> str:
    return json.dumps(contexto_previsao(resultado), ensure_ascii=False, indent=1)


def explain_forecast(client, model: str, pergunta: str, resultado_pipeline: dict) -> str:
    """Responde uma pergunta sobre o futuro usando apenas o JSON de contexto_previsao."""
    system_prompt = (
        "Você explica previsões de vendas para gestores, em português do Brasil, de forma direta (até 5 frases "
        "ou uma pequena tabela markdown se a pergunta pedir vários períodos). Os números vêm de um modelo de "
        "Machine Learning e chegam prontos em JSON.\n\n"
        + REGRAS_NUMEROS
        + "8. Sobre metas: se o usuário não informar o valor da meta, peça o valor e mostre a previsão com a faixa. "
        "Se informar, apenas compare a meta com a faixa de 80% do período: abaixo do mínimo = provável bater; "
        "dentro da faixa = incerto; acima do máximo = improvável.\n"
        "9. Termine com uma frase curta citando o modelo vencedor e o WAPE do backtest.\n"
        "10. NUNCA inclua código, SQL ou o JSON na resposta."
    )
    user_content = f"Pergunta do usuário: {pergunta}\n\nPrevisão (JSON):\n{_json_previsao(resultado_pipeline)}"
    return _chamar_llm(client, model, system_prompt, user_content, temperatura=0.2)


def gerar_resumo_executivo(client, model: str, resultado_pipeline: dict) -> str:
    """Resumo executivo de exatamente 3 frases sobre os próximos 28 dias (números só do JSON)."""
    system_prompt = (
        "Escreva um resumo executivo da previsão de vendas em EXATAMENTE 3 frases, em português do Brasil, "
        "texto corrido, sem títulos, listas ou markdown.\n"
        "Frase 1: receita prevista nos próximos 28 dias, com a faixa de 80% e a variação em relação aos 28 dias anteriores.\n"
        "Frase 2: pedidos previstos nos próximos 28 dias, com a faixa de 80%.\n"
        "Frase 3: confiabilidade: modelo vencedor e WAPE médio do backtest da receita, lembrando que B2B fica fora.\n\n"
        + REGRAS_NUMEROS
    )
    user_content = f"Previsão (JSON):\n{_json_previsao(resultado_pipeline)}"
    return _chamar_llm(client, model, system_prompt, user_content, temperatura=0.2)


# ---------- tabela "previsao" para o pandasql ----------

COLUNAS_TABELA = ["data", "alvo", "previsto", "lim_inf_80", "lim_sup_80", "lim_inf_95", "lim_sup_95", "modelo"]


def tabela_previsao(resultado: dict) -> pd.DataFrame:
    """Previsão diária em formato longo (uma linha por dia e alvo), pronta para o ps.sqldf.

    A data vai como texto AAAA-MM-DD para que substr/BETWEEN funcionem no SQLite.
    """
    partes = []
    for alvo, previsao in resultado["previsoes"].items():
        parte = previsao[["data", "previsto", "lim_inf_80", "lim_sup_80", "lim_inf_95", "lim_sup_95"]].copy()
        parte["data"] = pd.to_datetime(parte["data"]).dt.strftime("%Y-%m-%d")
        parte["alvo"] = alvo
        parte["modelo"] = NOMES_MODELOS.get(resultado["metadados"]["alvos"][alvo]["modelo_vencedor"], "")
        partes.append(parte)
    tabela = pd.concat(partes, ignore_index=True)[COLUNAS_TABELA]
    colunas_valor = COLUNAS_TABELA[2:7]
    tabela[colunas_valor] = tabela[colunas_valor].astype(float).round(2)
    return tabela


def descrever_tabela_previsao(tabela: pd.DataFrame) -> str:
    """Esquema + regras da tabela "previsao", para acrescentar ao prompt de geração de SQL."""
    return (
        f"Tabela adicional 'previsao' ({len(tabela)} registros): previsão DIÁRIA das vendas B2C pagas gerada pelo "
        f"modelo de Machine Learning, de {tabela['data'].min()} a {tabela['data'].max()}.\n"
        "Colunas:\n"
        '- "data" (texto AAAA-MM-DD; para mês use substr("data", 1, 7))\n'
        "- \"alvo\" (texto: 'receita' em reais ou 'pedidos' em quantidade)\n"
        '- "previsto" (número: valor previsto do dia)\n'
        '- "lim_inf_80", "lim_sup_80", "lim_inf_95", "lim_sup_95" (número: faixas de 80% e 95% DO DIA)\n'
        '- "modelo" (texto: modelo vencedor)'
    )


INSTRUCOES_SQL_PREVISAO = (
    "\nTABELA DE PREVISÃO DISPONÍVEL (exceção à regra 1):\n"
    "- Além de 'df' (dados realizados), existe a tabela 'previsao' (previsão diária do modelo de ML).\n"
    "- Use 'previsao' SOMENTE quando a pergunta pedir valores futuros/previstos; para o passado use 'df'.\n"
    "- Sempre filtre \"alvo\" = 'receita' ou \"alvo\" = 'pedidos'.\n"
    "- Para totais de um período, some apenas \"previsto\". NUNCA some colunas lim_* (a faixa de um total não é a soma das faixas diárias).\n"
)
