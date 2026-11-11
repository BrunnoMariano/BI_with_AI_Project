"""Testes offline de scripts/comparar_modelos.py (cliente falso no lugar da OpenAI)."""

import importlib.util
from types import SimpleNamespace

import pandas as pd
import pytest

from ml.explain import contexto_previsao
from ml.pipeline import run_predictive_pipeline
from tests.conftest import RAIZ_PROJETO
from tests.test_forecasting import _planilha_sintetica

_spec = importlib.util.spec_from_file_location("comparar_modelos", RAIZ_PROJETO / "scripts" / "comparar_modelos.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

VALOR = "CAST(REPLACE(REPLACE(REPLACE(REPLACE(COALESCE(\"Valor Pago\", '0'), 'R$', ''), ' ', ''), '.', ''), ',', '.') AS REAL)"
PAGO = "LOWER(TRIM(\"Status Venda\")) = 'pago'"
SQL_CERTO = {
    "total_pedidos": "SELECT COUNT(*) AS n FROM df",
    "pedidos_pagos": f"SELECT COUNT(*) AS n FROM df WHERE {PAGO}",
    "receita_paga": f"SELECT SUM({VALOR}) AS r FROM df WHERE {PAGO}",
    "receita_b2b": f"SELECT SUM({VALOR}) AS r FROM df WHERE {PAGO} AND \"Conta\" IS NOT NULL AND TRIM(\"Conta\") <> ''",
    "produto_mais_vendido": f"SELECT \"Produto\", COUNT(*) AS n FROM df WHERE {PAGO} GROUP BY \"Produto\" ORDER BY n DESC LIMIT 1",
}


def _brl(valor: float) -> str:
    return "R$ " + f"{valor:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")


class ClienteRoteiro:
    """Responde certo a cada pergunta do script, como um modelo perfeito faria."""

    def __init__(self, contexto: dict, inventar: bool = False):
        self.contexto, self.inventar = contexto, inventar
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        sistema, usuario = kwargs["messages"][0]["content"], kwargs["messages"][-1]["content"]
        pergunta = usuario.rsplit("\n", 1)[-1]
        if "Classifique a intenção" in sistema:
            texto = dict(cm.PERGUNTAS_INTENCAO)[pergunta]
        elif "Previsão (JSON)" in usuario:
            bloco = self.contexto["alvos"]["receita"]["proximos_28_dias"]
            texto = (
                f"A receita prevista é {_brl(bloco['previsto'])}, com faixa de 80% entre "
                f"{_brl(bloco['faixa_80']['minimo'])} e {_brl(bloco['faixa_80']['maximo'])}."
            )
            if self.inventar:
                texto += " Isso dá uns R$ 123.456,78 no trimestre."
        else:
            texto = SQL_CERTO[dict(cm.PERGUNTAS_SQL)[pergunta]]
        uso = SimpleNamespace(prompt_tokens=1000, completion_tokens=100, completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=texto))], usage=uso)


@pytest.fixture(scope="module")
def dados():
    df = _planilha_sintetica()
    return df, run_predictive_pipeline(df, "teste", horizonte_previsao=90)


def test_numeros_citados():
    assert cm.numeros_citados("R$ 23.935,27 em 07/04/2026, WAPE de 51,1% e uns 24 mil") == [23935.27, 51.1, 24000.0]


def test_numeros_inventados(dados):
    contexto = contexto_previsao(dados[1])
    previsto = contexto["alvos"]["receita"]["proximos_28_dias"]["previsto"]
    assert cm.numeros_inventados(f"Previsão de {_brl(previsto)} (faixa de 80%)", contexto) == []
    assert cm.numeros_inventados("Previsão de R$ 987.654,32", contexto) == [987654.32]
    assert cm.numeros_inventados("meta de R$ 50.000", contexto, "vamos bater a meta de R$ 50.000?") == []


def test_carregar_funcoes_chat():
    ns = cm.carregar_funcoes_chat()
    assert "forecast_query" in ns["INTENT_CATEGORIES"]
    assert ns["extract_sql"]("```sql\nSELECT 1\n```") == "SELECT 1"


def test_gabarito_e_resultado_contem(dados):
    df = dados[0]
    gabarito = cm.calcular_gabarito(df)
    pagos = df[df["Status Venda"] == "pago"]
    assert gabarito["total_pedidos"] == len(df)
    assert gabarito["pedidos_pagos"] == len(pagos)
    assert gabarito["receita_paga"] == pytest.approx(len(pagos) * 99.90, abs=0.01)
    assert gabarito["produto_mais_vendido"] == "Certificação A"
    assert cm.resultado_contem(pd.DataFrame({"x": ["a", 10.004]}), 10.0)
    assert not cm.resultado_contem(pd.DataFrame({"x": [11.0]}), 10.0)
    assert cm.resultado_contem(pd.DataFrame({"p": [" certificação a "]}), "Certificação A")


def test_avaliar_modelo_perfeito(dados):
    df, resultado = dados
    cliente = cm.ClienteMedido(ClienteRoteiro(contexto_previsao(resultado)))
    avaliacao = cm.avaliar_modelo(cliente, "gpt-6.1-terra", df, cm.carregar_funcoes_chat(), resultado, cm.calcular_gabarito(df))
    linha = avaliacao["linha"]
    assert (linha["intencao_acertos"], linha["sql_acertos"], linha["previsao_fieis"]) == ("4/4", "5/5", "2/2"), avaliacao["detalhes"]
    assert linha["chamadas"] == 11 and linha["erros_api"] == 0
    assert linha["custo_usd"] == pytest.approx(11 * (1000 * 1.00 + 100 * 5.00) / 1e6)


def test_avaliar_modelo_que_inventa_numero(dados):
    df, resultado = dados
    cliente = ClienteRoteiro(contexto_previsao(resultado), inventar=True)
    linha = cm.avaliar_modelo(cliente, "gpt-6-luna", df, cm.carregar_funcoes_chat(), resultado, cm.calcular_gabarito(df))["linha"]
    assert linha["previsao_fieis"] == "0/2"


def test_teste_fumaca_reporta_erro_da_api():
    class ClienteQueRecusa:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(ValueError("temperature não suportado"))))

    resultado = cm.teste_fumaca(ClienteQueRecusa(), ["gpt-6.1-sol"])
    assert resultado[0]["ok"] is False and "temperature" in resultado[0]["detalhe"]


def test_main_sem_chave(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cm.main([]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().out
