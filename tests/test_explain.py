"""Testes da Fase 4: camada de IA da previsão (ml/explain.py e ml/llm_utils.py).

O LLM é substituído por um cliente falso que só guarda o que receberia: os testes
rodam offline e conferem que ele recebe APENAS o JSON agregado da previsão.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pandasql as ps
import pytest

from ml.explain import (
    contexto_previsao,
    descrever_tabela_previsao,
    explain_forecast,
    gerar_resumo_executivo,
    parece_pergunta_de_previsao,
    resultado_tem_previsao,
    tabela_previsao,
)
from ml.forecasting import total_com_intervalo
from ml.pipeline import run_predictive_pipeline
from tests.test_forecasting import _planilha_sintetica

NOME_SIGILOSO = "Fulana Sigilosa da Silva"


class ClienteFalso:
    """Imita client.chat.completions.create e guarda os argumentos de cada chamada."""

    def __init__(self, resposta="Resposta de teste."):
        self.chamadas = []
        self.resposta = resposta
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.chamadas.append(kwargs)
        mensagem = SimpleNamespace(content=self.resposta)
        return SimpleNamespace(choices=[SimpleNamespace(message=mensagem)])


@pytest.fixture(scope="module")
def resultado():
    df = _planilha_sintetica()
    df["Comprador"] = NOME_SIGILOSO  # dado pessoal que NUNCA pode ir para o LLM
    return run_predictive_pipeline(df, "teste", horizonte_previsao=90)


# ---------- detecção de pergunta sobre o futuro ----------

@pytest.mark.parametrize(
    "pergunta",
    ["quanto vou vender mês que vem?", "previsão de receita", "vamos bater a meta?",
     "Qual a PROJEÇÃO para os próximos 3 meses?", "quantos pedidos teremos na próxima semana"],
)
def test_parece_pergunta_de_previsao(pergunta):
    assert parece_pergunta_de_previsao(pergunta)


@pytest.mark.parametrize("pergunta", ["quanto vendeu em janeiro?", "qual o total de vendas?", "quais colunas tem?", "olá"])
def test_nao_parece_pergunta_de_previsao(pergunta):
    assert not parece_pergunta_de_previsao(pergunta)


# ---------- JSON da previsão ----------

def test_contexto_previsao(resultado):
    contexto = contexto_previsao(resultado)
    texto = json.dumps(contexto, ensure_ascii=False)
    assert NOME_SIGILOSO not in texto  # só números agregados, nenhuma linha da planilha
    assert set(contexto["alvos"]) == {"receita", "pedidos"}

    receita = contexto["alvos"]["receita"]
    total = total_com_intervalo(resultado["previsoes"]["receita"], resultado["residuos"]["receita"], 28)
    bloco = receita["proximos_28_dias"]
    assert bloco["previsto"] == pytest.approx(total["previsto"], abs=0.01)
    assert bloco["faixa_80"]["minimo"] == pytest.approx(total["lim_inf_80"], abs=0.01)
    assert bloco["faixa_95"]["minimo"] <= bloco["faixa_80"]["minimo"] <= bloco["faixa_80"]["maximo"] <= bloco["faixa_95"]["maximo"]
    # Faixa deslocada (viés do backtest) -> o JSON explica ao LLM; centrada -> sem observação.
    fora_da_faixa = not (bloco["faixa_80"]["minimo"] <= bloco["previsto"] <= bloco["faixa_80"]["maximo"])
    assert ("observacao_faixa" in bloco) == fora_da_faixa
    assert bloco["validado_no_backtest"] is True
    assert receita["proximos_90_dias"]["validado_no_backtest"] is False
    assert receita["metricas_backtest_do_vencedor"]["wape_pct"] == pytest.approx(
        resultado["metadados"]["alvos"]["receita"]["wape_vencedor"] * 100, abs=0.1
    )
    assert receita["por_mes"] and receita["por_mes"][0]["mes"].count("/") == 1
    assert isinstance(contexto["alvos"]["pedidos"]["proximos_7_dias"]["previsto"], int)


def test_resultado_tem_previsao(resultado):
    assert resultado_tem_previsao(resultado)
    assert not resultado_tem_previsao({"faltando": ["valor pago"]})
    assert not resultado_tem_previsao(None)


# ---------- chamadas ao LLM ----------

def test_explain_forecast_envia_so_o_json(resultado):
    cliente = ClienteFalso("Previsão de R$ 10 (faixa de 80%: R$ 8 a R$ 12).")
    resposta = explain_forecast(cliente, "gpt-6.1-terra", "quanto vou vender mês que vem?", resultado)
    assert resposta.startswith("Previsão")

    chamada = cliente.chamadas[0]
    sistema, usuario = chamada["messages"][0]["content"], chamada["messages"][1]["content"]
    assert "SOMENTE os números do JSON" in sistema and "faixa de 80%" in sistema
    assert "quanto vou vender mês que vem?" in usuario
    json_enviado = json.loads(usuario.split("Previsão (JSON):\n", 1)[1])
    assert json_enviado == json.loads(json.dumps(contexto_previsao(resultado)))
    assert NOME_SIGILOSO not in usuario
    assert chamada["temperature"] == 0.2
    assert chamada["reasoning_effort"] == "none"


def test_explain_forecast_modelo_de_raciocinio(resultado):
    cliente = ClienteFalso()
    explain_forecast(cliente, "gpt-6.1-sol", "previsão de receita", resultado)
    assert "temperature" not in cliente.chamadas[0]  # GPT-6.1 Sol não aceita temperature
    assert cliente.chamadas[0]["reasoning_effort"] == "low"
    assert cliente.chamadas[0]["max_completion_tokens"] >= 2000  # sobra espaço depois do "raciocínio"


def test_resumo_executivo(resultado):
    cliente = ClienteFalso("Frase um. Frase dois. Frase três.")
    assert gerar_resumo_executivo(cliente, "gpt-5.4", resultado) == "Frase um. Frase dois. Frase três."
    assert "EXATAMENTE 3 frases" in cliente.chamadas[0]["messages"][0]["content"]


def test_resposta_vazia_vira_erro(resultado):
    with pytest.raises(RuntimeError):
        explain_forecast(ClienteFalso(""), "gpt-5.4", "previsão", resultado)


# ---------- tabela "previsao" no pandasql ----------

def test_tabela_previsao_no_pandasql(resultado):
    tabela = tabela_previsao(resultado)
    assert list(tabela.columns) == ["data", "alvo", "previsto", "lim_inf_80", "lim_sup_80", "lim_inf_95", "lim_sup_95", "modelo"]
    assert len(tabela) == 2 * 90
    assert set(tabela["alvo"]) == {"receita", "pedidos"}

    inicio = resultado["metadados"]["alvos"]["receita"]["inicio_previsao"]
    fim = (pd.Timestamp(inicio) + pd.Timedelta(days=27)).date().isoformat()
    consulta = (
        "SELECT SUM(\"previsto\") AS total FROM previsao "
        f"WHERE \"alvo\" = 'receita' AND \"data\" BETWEEN '{inicio}' AND '{fim}'"
    )
    total_sql = ps.sqldf(consulta, {"df": _planilha_sintetica(), "previsao": tabela})["total"].iloc[0]
    esperado = resultado["previsoes"]["receita"]["previsto"].iloc[:28].sum()
    assert total_sql == pytest.approx(esperado, abs=0.5)

    descricao = descrever_tabela_previsao(tabela)
    assert "'previsao'" in descricao and inicio in descricao
