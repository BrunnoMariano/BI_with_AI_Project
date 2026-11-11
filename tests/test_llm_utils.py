"""Testes de ml/llm_utils.py: lista de modelos e parâmetros enviados à OpenAI."""

import pytest

from ml.llm_utils import MIN_TOKENS_COM_RACIOCINIO, MODELO_PADRAO, MODELOS, chat_params, usa_raciocinio


def test_lista_de_modelos():
    assert MODELO_PADRAO == "gpt-6.1-terra"
    assert list(MODELOS) == ["gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-luna", "gpt-5.4-mini"]
    assert "o4-mini" not in MODELOS and "o3" not in MODELOS  # desligados/desligando pela OpenAI
    assert "Recomendado" in MODELOS[MODELO_PADRAO]["rotulo"]


@pytest.mark.parametrize("modelo", ["gpt-6.1-terra", "gpt-6-luna"])
def test_gpt6_rapido_manda_none_e_temperature(modelo):
    assert chat_params(modelo, 0.0, 20) == {"max_completion_tokens": 20, "reasoning_effort": "none", "temperature": 0.0}
    assert not usa_raciocinio(modelo)


def test_gpt61_sol_sem_temperature_e_com_folga():
    # Sol não aceita reasoning_effort "none"; com raciocínio, a OpenAI manda remover temperature.
    params = chat_params("gpt-6.1-sol", 0.0, 20)
    assert params == {"max_completion_tokens": MIN_TOKENS_COM_RACIOCINIO, "reasoning_effort": "low"}
    assert chat_params("gpt-6.1-sol", 0.2, 4000)["max_completion_tokens"] == 4000
    assert usa_raciocinio("gpt-6.1-sol")


def test_gpt54_mini_igual_a_antes():
    assert chat_params("gpt-5.4-mini", 0.3, 500) == {"max_completion_tokens": 500, "temperature": 0.3}


@pytest.mark.parametrize("modelo", ["o3", "o4-mini", "o1-preview"])
def test_serie_o_antiga_sem_temperature(modelo):
    params = chat_params(modelo, 0.0, 20)
    assert "temperature" not in params and "reasoning_effort" not in params
    assert params["max_completion_tokens"] == MIN_TOKENS_COM_RACIOCINIO


def test_modelo_desconhecido_mantem_regra_antiga():
    # "gpt-4o" contém a letra "o", mas não é da série o (era o bug da regra antiga).
    assert chat_params("gpt-4o", 0.0, 50) == {"max_completion_tokens": 50, "temperature": 0.0}
