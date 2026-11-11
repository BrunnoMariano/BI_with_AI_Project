"""Modelos da OpenAI oferecidos no app e os parâmetros certos para cada um (sem Streamlit).

Regras da documentação oficial (developers.openai.com, consultada em 07/11/2026):
- Família GPT-6: "When reasoning effort is not `none`, remove `temperature`...".
  Terra e Luna aceitam `none`; o 6.1 Sol não (mínimo `low`), então nele nunca vai temperature.
- O padrão do GPT-6 é raciocínio `medium`: mandamos `none` explicitamente para manter as
  respostas rápidas como antes.
- `max_completion_tokens` inclui os tokens de raciocínio: quem raciocina precisa de folga.
"""

from __future__ import annotations

# reasoning_effort: "none" (rápido, aceita temperature), "low" (raciocina, sem temperature)
# ou None (não enviar o parâmetro; o modelo usa o padrão dele, que no 5.4 mini é "none").
MODELOS = {
    "gpt-6.1-terra": {"rotulo": "GPT-6.1 Terra (Recomendado)", "reasoning_effort": "none", "aceita_temperatura": True},
    "gpt-6.1-sol": {"rotulo": "GPT-6.1 Sol (Máximo desempenho)", "reasoning_effort": "low", "aceita_temperatura": False},
    "gpt-6-luna": {"rotulo": "GPT-6 Luna (Econômico)", "reasoning_effort": "none", "aceita_temperatura": True},
    "gpt-5.4-mini": {"rotulo": "GPT-5.4 mini (Anterior)", "reasoning_effort": None, "aceita_temperatura": True},
}
MODELO_PADRAO = "gpt-6.1-terra"

# Modelos antigos de raciocínio (o1/o3/o4): só aceitam a temperatura padrão.
PREFIXOS_SERIE_O = ("o1", "o3", "o4")
MIN_TOKENS_COM_RACIOCINIO = 2000


def _config(model_name: str) -> dict:
    nome = str(model_name or "").strip().lower()
    if nome in MODELOS:
        return MODELOS[nome]
    if nome.startswith(PREFIXOS_SERIE_O):
        return {"reasoning_effort": None, "aceita_temperatura": False, "raciocina": True}
    # Modelo fora da lista: comportamento antigo (temperature, sem reasoning_effort).
    return {"reasoning_effort": None, "aceita_temperatura": True}


def usa_raciocinio(model_name: str) -> bool:
    """True se o modelo gasta tokens "pensando" antes de responder (ex.: GPT-6.1 Sol, o3)."""
    config = _config(model_name)
    return config.get("raciocina", False) or config["reasoning_effort"] not in (None, "none")


def chat_params(model_name: str, temperatura: float, max_tokens: int) -> dict:
    """Parâmetros de client.chat.completions.create (sem model/messages) adequados ao modelo.

    Ex.: chat_params("gpt-6.1-terra", 0.0, 20) -> reasoning_effort="none", temperature=0.0, 20 tokens;
         chat_params("gpt-6.1-sol", 0.0, 20)   -> reasoning_effort="low", sem temperature, 2000 tokens.
    """
    config = _config(model_name)
    params = {"max_completion_tokens": int(max_tokens)}
    if config["reasoning_effort"] is not None:
        params["reasoning_effort"] = config["reasoning_effort"]
    if config["aceita_temperatura"]:
        params["temperature"] = temperatura
    if usa_raciocinio(model_name):
        params["max_completion_tokens"] = max(int(max_tokens), MIN_TOKENS_COM_RACIOCINIO)
    return params
