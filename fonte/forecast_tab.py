"""Aba "Previsões": exibe os resultados do pipeline de ML (fonte/ml/).

O ML calcula, esta tela só mostra. As funções de montagem (gráficos, tabela, textos)
são separadas do Streamlit para poderem ser testadas; render_forecast_tab desenha a aba.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

from ml.anomaly import detect_anomalies
from ml.conversion import score_pending, train_conversion
from ml.explain import gerar_resumo_executivo, resultado_tem_previsao
from ml.forecasting import agregar_historico, agregar_previsao, simulate, total_com_intervalo
from ml.llm_utils import MODELO_PADRAO
from ml.pipeline import run_predictive_pipeline, verificar_requisitos
from ml.preprocessing import clean_orders
from ui_styles import render_section_intro
from user_data import load_dataframe_from_storage

HORIZONTES = (7, 14, 28, 90)
HORIZONTE_KPI = 28
HORIZONTE_MAXIMO = max(HORIZONTES)
GRANULARIDADES = {"dia": "Dia", "semana": "Semana", "mes": "Mês"}
ALVOS = {"receita": "Receita", "pedidos": "Pedidos"}
NOMES_MODELOS = {
    "holt_winters": "Holt-Winters",
    "lightgbm": "LightGBM",
    "baseline_sazonal_7d": "Baseline sazonal (7 dias)",
    "media_movel_28d": "Média móvel (28 dias)",
    "regressao_logistica": "Regressão logística",
}
NOMES_VARIAVEIS = {
    "produto": "Produto",
    "cupom": "Cupom",
    "valor": "Valor",
    "hora": "Hora da criação",
    "dia_semana": "Dia da semana da criação",
    "is_b2b": "Venda corporativa (B2B)",
    "pedidos_anteriores": "Pedidos anteriores do comprador",
    "tentativas_no_dia": "Tentativas no mesmo dia",
}
COLUNAS_PENDENTES = {
    "pedido_id": "Pedido",
    "data_pedido": "Criado em",
    "idade_dias": "Em aberto há (dias)",
    "status": "Status",
    "gerou_link": "Link de pagamento",
    "produto": "Produto",
    "valor": "Valor",
    "probabilidade": "Probabilidade de pagar",
    "receita_esperada": "Receita esperada",
}
ROTULOS_DIRECAO = {"acima": "▲ Acima", "abaixo": "▼ Abaixo"}

# Paleta validada para fundo escuro (dataviz: azul = real, laranja = previsão).
COR_REAL = "#3987e5"
COR_PREVISTO = "#d95926"
FAIXA_80 = "rgba(217, 89, 38, 0.30)"
FAIXA_95 = "rgba(217, 89, 38, 0.14)"
COR_ALERTA = "#fab219"  # status "atenção": distinto das séries; a direção vem do formato do marcador
COR_ANEL_MARCADOR = "#0d1117"
COR_TEXTO = "#eef2f7"
COR_TEXTO_SECUNDARIO = "#b6c0cf"
COR_GRADE = "rgba(255,255,255,0.06)"
DIAS_VISIVEIS_HISTORICO = 90
DIAS_SEMANA = ("seg", "ter", "qua", "qui", "sex", "sáb", "dom")


# ---------- formatação (pt-BR) ----------

def formatar_numero(valor: float, casas: int = 0) -> str:
    """1234.5 -> "1.234,5" (separadores brasileiros)."""
    if valor is None or pd.isna(valor):
        return "–"
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def formatar_brl(valor: float, casas: int = 0) -> str:
    """Valor em reais: 1234.5 -> "R$ 1.235"."""
    return "–" if valor is None or pd.isna(valor) else f"R$ {formatar_numero(valor, casas)}"


def formatar_pct(valor: float, sinal: bool = False, casas: int = 1) -> str:
    """0.123 -> "12,3%" (ou "+12,3%" com sinal=True)."""
    if valor is None or pd.isna(valor):
        return "–"
    texto = formatar_numero(abs(valor) * 100, casas) + "%"
    if sinal:
        return ("+" if valor >= 0 else "−") + texto
    return ("−" if valor < 0 else "") + texto


def formatar_valor(valor: float, alvo: str, casas: int = 0) -> str:
    """Formata conforme o alvo: reais para receita, número para pedidos."""
    return formatar_brl(valor, casas) if alvo == "receita" else formatar_numero(valor, casas)


def formatar_data(valor) -> str:
    return pd.Timestamp(valor).strftime("%d/%m/%Y")


# ---------- cálculos de exibição ----------

def serie_historica(resultado: dict, alvo: str) -> pd.Series:
    """Histórico do alvo como pd.Series diária (índice = data)."""
    historico = resultado["historico"][alvo]
    return pd.Series(historico["real"].to_numpy(dtype=float), index=pd.DatetimeIndex(historico["data"]), name=alvo)


def calcular_kpis(resultado: dict, dias: int = HORIZONTE_KPI) -> dict:
    """Totais previstos dos próximos `dias` com faixa de 80% e variação vs os `dias` anteriores."""
    kpis = {}
    for alvo in ALVOS:
        previsao = resultado["previsoes"][alvo]
        total = total_com_intervalo(previsao, resultado["residuos"][alvo], dias)
        anterior = float(serie_historica(resultado, alvo).iloc[-dias:].sum())
        total["anterior"] = anterior
        total["variacao"] = (total["previsto"] - anterior) / anterior if anterior else float("nan")
        kpis[alvo] = total
    return kpis


def tabela_modelos(resultado: dict, alvo: str) -> pd.DataFrame:
    """Tabela de comparação dos modelos (média dos cortes do backtest), vencedor na primeira linha."""
    tabela = resultado["modelos"][alvo]
    vencedor = resultado["metadados"]["alvos"][alvo]["modelo_vencedor"]
    return pd.DataFrame(
        {
            "Modelo": [NOMES_MODELOS.get(m, m) + (" ★ vencedor" if m == vencedor else "") for m in tabela["modelo"]],
            "WAPE": tabela["wape"].to_numpy(),
            "MAE": tabela["mae"].to_numpy(),
            "RMSE": tabela["rmse"].to_numpy(),
            "Erro no total": tabela["erro_total"].to_numpy(),
            "Cobertura 80%": tabela["cobertura_80"].to_numpy(),
        }
    )


def tabela_csv(previsao_agregada: pd.DataFrame, alvo: str, granularidade: str) -> bytes:
    """CSV da previsão exibida (separador ";" e vírgula decimal, abre direto no Excel brasileiro)."""
    tabela = pd.DataFrame(
        {
            "periodo": previsao_agregada["periodo"].dt.strftime("%Y-%m-%d"),
            "granularidade": granularidade,
            "alvo": alvo,
            "previsto": previsao_agregada["previsto"].round(2),
            "lim_inf_80": previsao_agregada["lim_inf_80"].round(2),
            "lim_sup_80": previsao_agregada["lim_sup_80"].round(2),
            "lim_inf_95": previsao_agregada["lim_inf_95"].round(2),
            "lim_sup_95": previsao_agregada["lim_sup_95"].round(2),
            "dias_no_periodo": previsao_agregada["n_dias"],
            "ja_realizado": previsao_agregada["realizado"].round(2),
        }
    )
    return tabela.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def tabela_pendentes(fila: pd.DataFrame) -> pd.DataFrame:
    """Fila de pendentes com os nomes de coluna da tela, na ordem recebida (maior probabilidade primeiro)."""
    tabela = fila[[c for c in COLUNAS_PENDENTES if c in fila]].rename(columns=COLUNAS_PENDENTES)
    if "Link de pagamento" in tabela:
        tabela["Link de pagamento"] = tabela["Link de pagamento"].map({True: "Sim", False: "Não"})
    return tabela


def tabela_avaliacao(treino: dict) -> pd.DataFrame:
    """Comparação dos classificadores no corte temporal, com o vencedor marcado."""
    comparacao = treino["comparacao"]
    return pd.DataFrame(
        {
            "Modelo": [
                NOMES_MODELOS.get(m, m) + (" ★ vencedor" if m == treino["vencedor"] else "") for m in comparacao["modelo"]
            ],
            "AUC": comparacao["auc"].to_numpy(),
            "Precisão média": comparacao["precisao_media"].to_numpy(),
        }
    )


def tabela_importancia(treino: dict) -> pd.DataFrame:
    """Importância das variáveis com nomes legíveis, da maior para a menor."""
    importancia = treino["importancia"]
    return pd.DataFrame(
        {
            "Variável": importancia["variavel"].map(lambda v: NOMES_VARIAVEIS.get(v, v)).to_numpy(),
            "Queda no AUC ao embaralhar": importancia["importancia"].to_numpy(),
        }
    )


def tabela_alertas(anomalias: pd.DataFrame, alvo: str) -> pd.DataFrame:
    """Só os dias anômalos, do mais recente para o mais antigo, com textos prontos para a tela."""
    dias = anomalias[anomalias["anomalia"]].sort_values("data", ascending=False)
    return pd.DataFrame(
        {
            "Dia": _rotulo_periodo(dias["data"], "dia"),
            "Direção": dias["direcao"].map(ROTULOS_DIRECAO).to_numpy(),
            "Real": [formatar_valor(v, alvo) for v in dias["real"]],
            "Previsto": [formatar_valor(v, alvo) for v in dias["previsto"]],
            "Faixa de 95%": [
                f"{formatar_valor(a, alvo)} a {formatar_valor(b, alvo)}"
                for a, b in zip(dias["lim_inf_95"], dias["lim_sup_95"])
            ],
            "Desvio": [formatar_pct(v / 100, sinal=True, casas=0) for v in dias["desvio_pct"]],
        }
    )


# ---------- gráficos ----------

def _rotulo_periodo(periodo: pd.Series, granularidade: str, parcial=None) -> list[str]:
    if granularidade == "semana":
        rotulos = ["Semana de " + d.strftime("%d/%m/%Y") for d in periodo]
    elif granularidade == "mes":
        rotulos = [d.strftime("%m/%Y") for d in periodo]
    else:
        rotulos = [f"{d.strftime('%d/%m/%Y')} ({DIAS_SEMANA[d.dayofweek]})" for d in periodo]
    if parcial is not None:
        rotulos = [r + (" · período incompleto" if p else "") for r, p in zip(rotulos, parcial)]
    return rotulos


def _faixa(fig: go.Figure, x, inferior, superior, cor: str, nome: str, ordem: int) -> None:
    """Área entre dois limites (linha inferior invisível + superior preenchida até ela)."""
    fig.add_trace(go.Scatter(x=x, y=inferior, mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=x, y=superior, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=cor,
            name=nome, hoverinfo="skip", legendrank=ordem,
        )
    )


def _marcador_vertical(fig: go.Figure, x, texto: str) -> None:
    fig.add_shape(
        type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
        line={"color": COR_TEXTO_SECUNDARIO, "width": 1, "dash": "dot"},
    )
    fig.add_annotation(
        x=x, y=1, yref="paper", text=texto, showarrow=False, xanchor="left", yanchor="bottom",
        xshift=4, font={"color": COR_TEXTO_SECUNDARIO, "size": 12},
    )


def _layout(fig: go.Figure, alvo: str, granularidade: str = "dia") -> go.Figure:
    # Datas numéricas: o Plotly escreveria os meses em inglês ("Feb", "Apr").
    formato_data = "%m/%Y" if granularidade == "mes" else "%d/%m/%y"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.22)",
        font={"family": "Plus Jakarta Sans, Segoe UI, sans-serif", "color": COR_TEXTO},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        height=440,
        hovermode="x unified",
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.06, "x": 0, "traceorder": "normal",
            "font": {"color": COR_TEXTO_SECUNDARIO},
        },
        xaxis={"gridcolor": COR_GRADE, "zeroline": False, "tickformat": formato_data, "hoverformat": formato_data},
        yaxis={
            "gridcolor": COR_GRADE,
            "zeroline": False,
            "rangemode": "tozero",
            "title": "Receita (R$)" if alvo == "receita" else "Pedidos pagos",
            "tickprefix": "R$ " if alvo == "receita" else "",
            "tickformat": ",.0f",
        },
        separators=",.",
    )
    return fig


def _hover_valor(alvo: str) -> str:
    return "R$ %{y:,.0f}" if alvo == "receita" else "%{y:,.0f}"


def construir_grafico_previsao(
    historico: pd.Series, previsao: pd.DataFrame, residuos, alvo: str, granularidade: str = "dia"
) -> tuple[go.Figure, pd.DataFrame]:
    """Histórico (linha sólida), previsão (tracejada) e faixas de 80%/95%, com linha "hoje".

    Em semana/mês, o último período incompleto (ex.: 5 dias de julho) fica fora do
    gráfico para não parecer uma queda; ele continua na previsão agregada devolvida,
    que é a usada no download.
    """
    origem = previsao["data"].iloc[0]
    hist = agregar_historico(historico, granularidade, origem)
    agregada = agregar_previsao(previsao, residuos, granularidade, historico)
    prev = agregada
    if granularidade != "dia" and len(agregada) > 1 and agregada["parcial"].iloc[-1]:
        prev = agregada.iloc[:-1]

    fig = go.Figure()
    _faixa(fig, prev["periodo"], prev["lim_inf_95"], prev["lim_sup_95"], FAIXA_95, "Faixa de 95%", 4)
    _faixa(fig, prev["periodo"], prev["lim_inf_80"], prev["lim_sup_80"], FAIXA_80, "Faixa de 80%", 3)

    fig.add_trace(
        go.Scatter(
            x=hist["periodo"], y=hist["real"], mode="lines", name="Real", legendrank=1,
            line={"color": COR_REAL, "width": 2},
            customdata=_rotulo_periodo(hist["periodo"], granularidade, hist["parcial"]),
            hovertemplate="%{customdata}<br>Real: " + _hover_valor(alvo) + "<extra></extra>",
        )
    )

    # A linha prevista começa no último ponto real para não deixar buraco no gráfico.
    x_prev, y_prev = list(prev["periodo"]), list(prev["previsto"])
    if len(hist):
        x_prev, y_prev = [hist["periodo"].iloc[-1], *x_prev], [hist["real"].iloc[-1], *y_prev]
    detalhes = [
        f"{rotulo}<br>Faixa 80%: {formatar_valor(a, alvo)} a {formatar_valor(b, alvo)}"
        f"<br>Faixa 95%: {formatar_valor(c, alvo)} a {formatar_valor(d, alvo)}"
        + (f"<br>Inclui {formatar_valor(r, alvo)} já realizado" if r else "")
        for rotulo, a, b, c, d, r in zip(
            _rotulo_periodo(prev["periodo"], granularidade, prev["parcial"]),
            prev["lim_inf_80"], prev["lim_sup_80"], prev["lim_inf_95"], prev["lim_sup_95"], prev["realizado"],
        )
    ]
    if len(x_prev) > len(prev):
        detalhes = ["Último período real", *detalhes]
    fig.add_trace(
        go.Scatter(
            x=x_prev, y=y_prev, mode="lines", name="Previsto", legendrank=2,
            line={"color": COR_PREVISTO, "width": 2, "dash": "dash"},
            customdata=detalhes,
            hovertemplate="%{customdata}<br>Previsto: " + _hover_valor(alvo) + "<extra></extra>",
        )
    )
    _marcador_vertical(fig, origem, "hoje · início da previsão")
    _layout(fig, alvo, granularidade)
    if granularidade == "dia" and len(hist) > DIAS_VISIVEIS_HISTORICO:
        # Abre focado nos últimos 90 dias + previsão; o botão de autoescala mostra tudo.
        inicio = origem - pd.Timedelta(days=DIAS_VISIVEIS_HISTORICO)
        fig.update_xaxes(range=[inicio, prev["periodo"].iloc[-1] + pd.Timedelta(days=1)])
        fig.update_yaxes(range=[0, 1.05 * max(hist.loc[hist["periodo"] >= inicio, "real"].max(), prev["lim_sup_95"].max())])
    return fig, agregada


def construir_grafico_simulacao(historico: pd.Series, simulacao: dict, alvo: str, dias_antes: int = 84) -> go.Figure:
    """Modo simulação: real antes e depois do corte (sólido) x previsto feito no corte (tracejado)."""
    corte = simulacao["corte"]
    previsao = simulacao["previsao"]
    janela = historico.loc[corte - pd.Timedelta(days=dias_antes) : previsao["data"].iloc[-1]]

    fig = go.Figure()
    if previsao["lim_inf_80"].notna().all():
        _faixa(fig, previsao["data"], previsao["lim_inf_95"], previsao["lim_sup_95"], FAIXA_95, "Faixa de 95%", 4)
        _faixa(fig, previsao["data"], previsao["lim_inf_80"], previsao["lim_sup_80"], FAIXA_80, "Faixa de 80%", 3)
    fig.add_trace(
        go.Scatter(
            x=janela.index, y=janela.to_numpy(), mode="lines", name="Real", legendrank=1,
            line={"color": COR_REAL, "width": 2},
            customdata=_rotulo_periodo(pd.Series(janela.index), "dia"),
            hovertemplate="%{customdata}<br>Real: " + _hover_valor(alvo) + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=previsao["data"], y=previsao["previsto"], mode="lines", name="Previsto no corte", legendrank=2,
            line={"color": COR_PREVISTO, "width": 2, "dash": "dash"},
            hovertemplate="Previsto: " + _hover_valor(alvo) + "<extra></extra>",
        )
    )
    _marcador_vertical(fig, corte + pd.Timedelta(days=1), "corte · o modelo não viu daqui em diante")
    return _layout(fig, alvo)


def construir_grafico_anomalias(anomalias: pd.DataFrame, alvo: str) -> go.Figure:
    """Backtest: real, previsto e faixa de 95%, com triângulos nos dias fora da faixa."""
    fig = go.Figure()
    _faixa(fig, anomalias["data"], anomalias["lim_inf_95"], anomalias["lim_sup_95"], FAIXA_95, "Faixa de 95%", 5)
    fig.add_trace(
        go.Scatter(
            x=anomalias["data"], y=anomalias["previsto"], mode="lines", name="Previsto no backtest", legendrank=2,
            line={"color": COR_PREVISTO, "width": 2, "dash": "dash"},
            customdata=[
                f"{formatar_valor(a, alvo)} a {formatar_valor(b, alvo)}"
                for a, b in zip(anomalias["lim_inf_95"], anomalias["lim_sup_95"])
            ],
            hovertemplate="Previsto: " + _hover_valor(alvo) + "<br>Faixa 95%: %{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=anomalias["data"], y=anomalias["real"], mode="lines", name="Real", legendrank=1,
            line={"color": COR_REAL, "width": 2},
            customdata=_rotulo_periodo(anomalias["data"], "dia"),
            hovertemplate="%{customdata}<br>Real: " + _hover_valor(alvo) + "<extra></extra>",
        )
    )
    for direcao, simbolo, nome, ordem in (
        ("acima", "triangle-up", "Acima da faixa", 3),
        ("abaixo", "triangle-down", "Abaixo da faixa", 4),
    ):
        dias = anomalias[anomalias["direcao"] == direcao]
        if dias.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=dias["data"], y=dias["real"], mode="markers", name=nome, legendrank=ordem,
                marker={"symbol": simbolo, "size": 13, "color": COR_ALERTA, "line": {"color": COR_ANEL_MARCADOR, "width": 2}},
                customdata=[formatar_pct(v / 100, sinal=True, casas=0) for v in dias["desvio_pct"]],
                hovertemplate=nome + ": %{customdata} vs previsto<extra></extra>",
            )
        )
    return _layout(fig, alvo)


# ---------- cache (retreina só ao trocar de arquivo ou clicar em Retreinar) ----------

@st.cache_data(show_spinner=False, max_entries=8)
def _rodar_pipeline(file_id: str, versao: int, _registro_arquivo: dict, _df: pd.DataFrame | None) -> dict:
    """Roda o pipeline uma vez por (arquivo, versão). `versao` muda ao clicar em Retreinar."""
    df = _df if _df is not None else load_dataframe_from_storage(_registro_arquivo)
    faltando = verificar_requisitos(df)
    if faltando:
        return {"faltando": faltando, "colunas": [str(c) for c in df.columns[:20]]}
    try:
        return run_predictive_pipeline(df, file_id, horizonte_previsao=HORIZONTE_MAXIMO)
    except ValueError as erro:
        return {"erro": str(erro)}


def obter_resultado_previsao(file_id: str, registro_arquivo: dict, df: pd.DataFrame | None = None) -> dict | None:
    """Resultado do pipeline (do cache) para o arquivo, na mesma versão da aba Previsões.

    Usado pelo chat. Devolve None se o arquivo não for uma base de pedidos ou se a previsão falhar.
    """
    versao = st.session_state.get("forecast_versions", {}).get(file_id, 0)
    if df is None and st.session_state.get("loaded_file_id") == file_id:
        df = st.session_state.get("df_data")
    try:
        resultado = _rodar_pipeline(file_id, versao, registro_arquivo, df)
    except Exception:
        return None
    return resultado if resultado_tem_previsao(resultado) else None


@st.cache_data(show_spinner=False, max_entries=32)
def _rodar_pipeline_recorte(file_id: str, versao: int, chave_recorte: str, inicio: str, fim: str, _df: pd.DataFrame) -> dict:
    """Previsão de um recorte (ex.: um produto), com as mesmas datas de treino da base inteira.

    Erros também ficam no cache, para não retreinar a cada clique num recorte que não dá para prever.
    """
    try:
        return run_predictive_pipeline(_df, f"file:{file_id}|{chave_recorte}", inicio=inicio, fim=fim,
                                       horizonte_previsao=HORIZONTE_MAXIMO)
    except ValueError as erro:
        return {"erro": str(erro)}


def obter_previsao_recorte(file_id: str, registro_arquivo: dict, df_recorte: pd.DataFrame, chave_recorte: str = "") -> dict:
    """Resultado do pipeline para o Dashboard: base inteira (chave vazia) ou recorte filtrado.

    O recorte usa o início e o fim do treino da base inteira, então a previsão sempre parte
    do mesmo "hoje". Levanta ValueError com mensagem em português se não der para prever.
    """
    base = obter_resultado_previsao(file_id, registro_arquivo)
    if base is None:
        raise ValueError("Este arquivo não permite gerar previsão (precisa ser uma base de pedidos com data, status e valor).")
    if not chave_recorte:
        return base
    meta = base["metadados"]["alvos"]["receita"]
    versao = st.session_state.get("forecast_versions", {}).get(file_id, 0)
    resultado = _rodar_pipeline_recorte(file_id, versao, chave_recorte, meta["inicio_treino"], meta["fim_treino"], df_recorte)
    if "erro" in resultado:
        raise ValueError(f"Não foi possível prever para este recorte: {resultado['erro']}")
    return resultado


@st.cache_data(show_spinner=False, max_entries=8)
def _rodar_conversao(file_id: str, versao: int, _registro_arquivo: dict, _df: pd.DataFrame | None) -> dict:
    """Treina o classificador e pontua a fila (inteira, recentes e recuperáveis) uma vez por (arquivo, versão)."""
    df = _df if _df is not None else load_dataframe_from_storage(_registro_arquivo)
    try:
        pedidos = clean_orders(df)
        treino = train_conversion(pedidos)
        fila = score_pending(pedidos, treino)
    except ValueError as erro:
        return {"erro": str(erro)}
    return {"treino": {chave: valor for chave, valor in treino.items() if chave != "modelo"}, "fila": fila}


@st.cache_data(show_spinner=False, max_entries=64)
def _rodar_simulacao(file_id: str, versao: int, alvo: str, corte: str, modelo: str, _serie: pd.Series) -> dict:
    return simulate(_serie, corte, modelo)


# ---------- tela ----------

def _caption(texto: str) -> None:
    """st.caption escapando "$": no Markdown do Streamlit, "R$ 10 a R$ 20" viraria fórmula matemática."""
    st.caption(texto.replace("$", "\\$"))


def _aviso_fonte_generica(resultado: dict) -> None:
    itens = "\n".join(f"- {item}" for item in resultado["faltando"])
    colunas = ", ".join(resultado.get("colunas", [])) or "nenhuma"
    st.warning(
        "Este arquivo não parece ser uma base de pedidos, por isso a previsão não pode ser gerada.\n\n"
        f"Para prever, a base precisa ter:\n{itens}\n\n"
        f"Colunas encontradas no arquivo: {colunas}."
    )


def _render_resumo_executivo(resultado: dict, file_id: str, versao: int, openai_api_key: str, modelo: str) -> None:
    """Resumo de 3 frases escrito pela IA, gerado uma vez por arquivo (e a cada Retreinar)."""
    resumos = st.session_state.setdefault("forecast_summaries", {})
    chave = f"{file_id}:{versao}"
    with st.container(border=True):
        st.markdown("**Resumo executivo**")
        if chave not in resumos:
            if not openai_api_key:
                st.caption("Informe a chave da OpenAI na barra lateral para gerar o resumo automático.")
                return
            with st.spinner("Escrevendo o resumo executivo..."):
                try:
                    resumos[chave] = {"texto": gerar_resumo_executivo(OpenAI(api_key=openai_api_key), modelo, resultado)}
                except Exception as erro:
                    # Guarda o erro para não chamar a API de novo a cada clique na tela.
                    resumos[chave] = {"erro": str(erro)}
        resumo = resumos[chave]
        if "erro" in resumo:
            st.warning(f"Não foi possível gerar o resumo executivo: {resumo['erro']}")
        else:
            st.markdown(resumo["texto"].replace("$", "\\$"))
            _caption("Escrito pela IA usando apenas os números calculados pelo modelo (os mesmos dos indicadores abaixo).")
        if st.button("Gerar resumo de novo", key="prev_resumo_novo"):
            resumos.pop(chave, None)
            st.rerun()


def _render_kpis(resultado: dict) -> None:
    kpis = calcular_kpis(resultado)
    receita, pedidos = kpis["receita"], kpis["pedidos"]
    fim_treino = formatar_data(resultado["metadados"]["alvos"]["receita"]["fim_treino"])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Receita prevista · próximos 28 dias",
            formatar_brl(receita["previsto"]),
            help="Soma da receita B2C prevista. A faixa de 80% vem dos erros do modelo no backtest.",
        )
        _caption(f"Faixa de 80%: {formatar_brl(receita['lim_inf_80'])} a {formatar_brl(receita['lim_sup_80'])}")
    with col2:
        st.metric("Pedidos previstos · próximos 28 dias", formatar_numero(pedidos["previsto"]))
        _caption(f"Faixa de 80%: {formatar_numero(pedidos['lim_inf_80'])} a {formatar_numero(pedidos['lim_sup_80'])}")
    with col3:
        st.metric(
            "Receita vs 28 dias anteriores",
            formatar_pct(receita["variacao"], sinal=True),
            help=f"Compara a receita prevista com a realizada nos 28 dias até {fim_treino}.",
        )
        _caption(
            f"Base: {formatar_brl(receita['anterior'])} realizados · "
            f"pedidos: {formatar_pct(pedidos['variacao'], sinal=True)}"
        )


def _render_tabela_modelos(resultado: dict, alvo: str) -> None:
    tabela = tabela_modelos(resultado, alvo)
    unidade = "R$ " if alvo == "receita" else ""

    def destacar_vencedor(linha):
        estilo = "background-color: rgba(217, 89, 38, 0.18); font-weight: 600"
        return [estilo if "vencedor" in linha["Modelo"] else ""] * len(linha)

    estilizada = tabela.style.apply(destacar_vencedor, axis=1).format(
        {
            "WAPE": lambda v: formatar_pct(v),
            "MAE": lambda v: unidade + formatar_numero(v, 1),
            "RMSE": lambda v: unidade + formatar_numero(v, 1),
            "Erro no total": lambda v: formatar_pct(v, sinal=True),
            "Cobertura 80%": lambda v: formatar_pct(v, casas=0),
        }
    )
    st.dataframe(estilizada, hide_index=True, use_container_width=True)
    cortes = resultado["metadados"]["alvos"][alvo]["cortes_backtest"]
    _caption(
        f"Média de {len(cortes)} cortes de 28 dias ({formatar_data(cortes[0]['inicio_teste'])} a "
        f"{formatar_data(cortes[-1]['fim_teste'])}): o modelo treina só com o passado de cada corte e é "
        "comparado com o que de fato aconteceu. WAPE = erro somado ÷ total real (menor é melhor). "
        "Cobertura 80% = quantos dias reais caíram dentro da faixa de 80%."
    )


def _render_b2b(resultado: dict) -> None:
    b2b = resultado["b2b"]
    historico, janela = b2b["historico"], b2b["janela"]
    meta = resultado["metadados"]["alvos"]
    ticket_b2c = meta["receita"]["total_treino"] / meta["pedidos"]["total_treino"] if meta["pedidos"]["total_treino"] else 0
    with st.container(border=True):
        st.markdown("**Vendas corporativas (B2B)**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Histórico completo", formatar_brl(historico["receita"], 2))
        col1.caption(f"{historico['pedidos']} pedidos pagos")
        col2.metric(f"Janela de treino (desde {formatar_data(janela['inicio'])})", formatar_brl(janela["receita"], 2))
        col2.caption(f"{janela['pedidos']} pedidos pagos")
        col3.metric("Média mensal na janela", formatar_brl(janela["media_mensal_receita"], 2))
        col3.caption(f"{formatar_numero(janela['media_mensal_pedidos'], 1)} pedidos por mês")
        _caption(
            "Essas vendas não entram na previsão: são raras (cerca de "
            f"{formatar_numero(janela['media_mensal_pedidos'], 0)} por mês) e de alto valor (ticket médio de "
            f"{formatar_brl(janela['ticket_medio'])}, contra {formatar_brl(ticket_b2c)} no B2C). Poucos pedidos "
            "grandes e irregulares distorceriam o modelo, por isso aparecem só como histórico."
        )


def _render_previsao(resultado: dict, alvo: str, horizonte: int, granularidade: str) -> None:
    historico = serie_historica(resultado, alvo)
    previsao = resultado["previsoes"][alvo].iloc[:horizonte]
    fig, agregada = construir_grafico_previsao(historico, previsao, resultado["residuos"][alvo], alvo, granularidade)
    st.plotly_chart(fig, use_container_width=True)

    notas = []
    if horizonte > HORIZONTE_KPI:
        notas.append(
            "Acima de 28 dias o modelo não foi validado no backtest: as faixas usam os erros de 28 dias "
            "e tendem a ser otimistas."
        )
    if granularidade != "dia" and len(agregada) > 1 and agregada["parcial"].iloc[-1]:
        ultimo = agregada.iloc[-1]
        notas.append(
            f"O último período ({int(ultimo['n_dias'])} dias a partir de {formatar_data(ultimo['periodo'])}) "
            "está incompleto e ficou fora do gráfico, mas está no CSV."
        )
    if granularidade == "mes" and agregada["realizado"].iloc[0] > 0:
        notas.append("O primeiro mês da previsão soma o que já foi realizado com o que falta prever.")
    if notas:
        _caption(" ".join(notas))

    st.download_button(
        "Baixar previsão (CSV)",
        data=tabela_csv(agregada, alvo, granularidade),
        file_name=f"previsao_{alvo}_{horizonte}d_{granularidade}.csv",
        mime="text/csv",
        key="prev_download",
    )


def _render_simulacao(resultado: dict, file_id: str, versao: int, alvo: str) -> None:
    meta = resultado["metadados"]["alvos"][alvo]
    historico = serie_historica(resultado, alvo)
    minimo = (pd.Timestamp(meta["inicio_treino"]) + pd.DateOffset(months=2)).date()
    maximo = (pd.Timestamp(meta["fim_treino"]) - pd.Timedelta(days=HORIZONTE_KPI)).date()

    corte = st.date_input(
        "Data de corte (o modelo só enxerga os dados até este dia)",
        value=maximo, min_value=minimo, max_value=maximo, format="DD/MM/YYYY", key="prev_corte",
    )
    with st.spinner("Treinando com os dados até o corte..."):
        try:
            simulacao = _rodar_simulacao(file_id, versao, alvo, pd.Timestamp(corte).isoformat(), meta["modelo_vencedor"], historico)
        except ValueError as erro:
            st.warning(str(erro))
            return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("WAPE nos 28 dias", formatar_pct(simulacao["wape"]), help="Erro somado ÷ total real. Menor é melhor.")
    col2.metric("Erro no total do período", formatar_pct(simulacao["erro_total"], sinal=True),
                help="Positivo = previu mais do que aconteceu; negativo = previu menos.")
    col3.metric("Previsto no período", formatar_valor(simulacao["total_previsto"], alvo))
    col4.metric("Real no período", formatar_valor(simulacao["total_real"], alvo))

    st.plotly_chart(construir_grafico_simulacao(historico, simulacao, alvo), use_container_width=True)
    previsao = simulacao["previsao"]
    _caption(
        f"Modelo {NOMES_MODELOS.get(simulacao['modelo'], simulacao['modelo'])} treinado com "
        f"{simulacao['n_dias_treino']} dias (até {formatar_data(simulacao['corte'])}) e comparado com o real de "
        f"{formatar_data(previsao['data'].iloc[0])} a {formatar_data(previsao['data'].iloc[-1])}."
        + (
            f" Faixas medidas em {simulacao['n_cortes_intervalo']} corte(s) anteriores ao corte;"
            f" {formatar_pct(simulacao['cobertura_80'], casas=0)} dos dias reais ficaram na faixa de 80%."
            if simulacao["n_cortes_intervalo"]
            else ""
        )
    )
    for aviso in simulacao["avisos"]:
        st.info(aviso)


def _render_avaliacao_conversao(treino: dict) -> None:
    validacao = treino["validacao"]
    with st.expander("Como o modelo de pagamento foi avaliado"):
        col_modelos, col_variaveis = st.columns(2)
        with col_modelos:
            avaliacao = tabela_avaliacao(treino).style.format({"AUC": lambda v: formatar_numero(v, 3), "Precisão média": lambda v: formatar_numero(v, 3)})
            st.dataframe(avaliacao, hide_index=True, use_container_width=True)
        with col_variaveis:
            importancia = tabela_importancia(treino).style.format({"Queda no AUC ao embaralhar": lambda v: formatar_numero(v, 3)})
            st.dataframe(importancia, hide_index=True, use_container_width=True)
        _caption(
            f"Treinado com os {formatar_numero(validacao['n_treino'])} pedidos mais antigos e testado nos "
            f"{formatar_numero(validacao['n_teste'])} mais recentes (a partir de {formatar_data(validacao['inicio_teste'])}). "
            "AUC: 0,5 é acaso e 1,0 é perfeito. Precisão média: quanto os pedidos com nota alta de fato pagaram "
            f"(o acaso daria {formatar_numero(validacao['taxa_pago_teste'], 3)}). "
            "Usa só o que se sabe na criação do pedido; dados preenchidos no checkout ficam de fora."
        )


def _render_pendentes(file_id: str, versao: int, registro: dict, df: pd.DataFrame | None) -> None:
    with st.container(border=True):
        st.markdown("**Pedidos pendentes**")
        with st.spinner("Calculando a probabilidade de pagamento..."):
            try:
                conversao = _rodar_conversao(file_id, versao, registro, df)
            except Exception as erro:
                st.warning(f"Não foi possível calcular a probabilidade de pagamento: {erro}")
                return
        if "erro" in conversao:
            st.info(f"A probabilidade de pagamento não está disponível para este arquivo: {conversao['erro']}")
            return

        treino, fila = conversao["treino"], conversao["fila"]
        dias_resolucao = treino["validacao"]["dias_resolucao"]
        recentes, recuperaveis = fila["recentes"], fila["recuperaveis"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Receita esperada · abertos há até {dias_resolucao} dias", formatar_brl(recuperaveis["receita_esperada"]),
                      help="Soma de valor × probabilidade de pagamento dos pedidos recentes, sem as tentativas abandonadas.")
            _caption(f"{formatar_numero(recuperaveis['n_pedidos'])} pedidos · {formatar_brl(recuperaveis['valor_em_aberto'])} em aberto")
        with col2:
            st.metric("AUC do modelo", formatar_numero(treino["auc"], 2), help="Medido nos pedidos mais recentes, que o modelo não viu no treino.")
            _caption(f"{NOMES_MODELOS.get(treino['vencedor'], treino['vencedor'])} · precisão média {formatar_numero(treino['precisao_media'], 2)}")

        _caption(
            f"Ficaram fora do indicador {formatar_numero(recentes['n_abandonadas'])} dos {formatar_numero(recentes['n_pedidos'])} "
            "pedidos recentes: o comprador já tem outro pedido pago, então são tentativas abandonadas. "
            f"Fila inteira: {formatar_brl(fila['receita_esperada'])} esperados em {formatar_numero(fila['n_pedidos'])} pedidos "
            f"({formatar_brl(fila['valor_em_aberto'])} em aberto). Esse total está inflado: a probabilidade usa o que se sabia na "
            f"criação do pedido e não considera há quanto tempo ele está parado, e pedidos em aberto há mais de {dias_resolucao} dias "
            "raramente são pagos."
        )
        if fila["n_pedidos"]:
            tabela = tabela_pendentes(fila["pedidos"]).style.format(
                {
                    "Pedido": str,
                    "Criado em": formatar_data,
                    "Valor": lambda v: formatar_brl(v, 2),
                    "Probabilidade de pagar": lambda v: formatar_pct(v, casas=0),
                    "Receita esperada": lambda v: formatar_brl(v, 2),
                },
                na_rep="–",
            )
            st.dataframe(tabela, hide_index=True, use_container_width=True)
        else:
            st.info("Não há pedidos pendentes nem aguardando pagamento neste arquivo.")
        _render_avaliacao_conversao(treino)


def _render_alertas(resultado: dict, alvo: str) -> None:
    with st.container(border=True):
        st.markdown(f"**Alertas · {ALVOS[alvo].lower()}**")
        if "backtest" not in resultado:  # resultado em cache de antes da Fase 6
            st.info("Clique em Retreinar para calcular os alertas deste arquivo.")
            return
        anomalias = detect_anomalies(serie_historica(resultado, alvo), resultado["backtest"][alvo])
        n_anomalias = int(anomalias["anomalia"].sum())
        n_acima = int((anomalias["direcao"] == "acima").sum())
        total = len(anomalias)
        _caption(
            f"{n_anomalias} de {total} dias ficaram fora da faixa de 95% prevista no backtest "
            f"({n_acima} acima e {n_anomalias - n_acima} abaixo). Mesmo sem nada de errado, cerca de "
            f"{formatar_numero(total * 0.05)} dias ficariam fora por acaso. O desvio compara o real com o previsto."
        )
        st.plotly_chart(construir_grafico_anomalias(anomalias, alvo), use_container_width=True)
        if n_anomalias:
            st.dataframe(tabela_alertas(anomalias, alvo), hide_index=True, use_container_width=True)
        else:
            st.info("Nenhum dia ficou fora da faixa de 95% no período do backtest.")


def render_forecast_tab(fonte_dados: str, openai_api_key: str = "", modelo: str = MODELO_PADRAO) -> None:
    """Desenha a aba Previsões para o arquivo selecionado na barra lateral."""
    render_section_intro(
        "Previsões",
        "Previsão de pedidos e receita",
        "Projeção das vendas B2C pagas, com faixas de confiança e validação no passado.",
    )
    if fonte_dados == "Banco de Dados":
        st.info("A previsão funciona com arquivos de vendas. Em Conexão de Dados, escolha \"Arquivo CSV/Excel\".")
        return

    file_id = st.session_state.get("selected_file_id")
    registro = next((f for f in st.session_state.get("user_files", []) if f["file_id"] == file_id), None)
    if registro is None:
        st.info("Envie ou selecione um arquivo de vendas em Conexão de Dados para ver as previsões.")
        return

    versoes = st.session_state.setdefault("forecast_versions", {})
    versao = versoes.get(file_id, 0)
    # Reaproveita o arquivo já carregado pela aba Conversa, se for o mesmo.
    df = st.session_state.get("df_data") if st.session_state.get("loaded_file_id") == file_id else None

    with st.spinner("Treinando os modelos de previsão (leva alguns segundos)..."):
        try:
            resultado = _rodar_pipeline(file_id, versao, registro, df)
        except Exception as erro:
            st.error(f"Não foi possível gerar a previsão: {erro}")
            return
    if "faltando" in resultado:
        _aviso_fonte_generica(resultado)
        return
    if "erro" in resultado:
        st.warning(f"Não foi possível gerar a previsão com este arquivo: {resultado['erro']}")
        return

    meta = resultado["metadados"]["alvos"]["receita"]
    info, botao = st.columns([5, 1])
    info.caption(
        f"Arquivo **{registro.get('name', '')}** · treino de {formatar_data(meta['inicio_treino'])} a "
        f"{formatar_data(meta['fim_treino'])} ({meta['n_obs']} dias, só B2C) · previsão a partir de "
        f"{formatar_data(meta['inicio_previsao'])}"
    )
    if botao.button("Retreinar", use_container_width=True, key="prev_retreinar",
                    help="Roda o backtest e treina os modelos de novo para este arquivo."):
        versoes[file_id] = versao + 1
        st.rerun()

    _render_resumo_executivo(resultado, file_id, versao, openai_api_key, modelo)
    _render_kpis(resultado)

    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.4, 1.4])
    alvo = c1.segmented_control("Alvo", list(ALVOS), format_func=ALVOS.get, default="receita", key="prev_alvo") or "receita"
    modo = c2.segmented_control("Modo", ["previsao", "simulacao"], default="previsao", key="prev_modo",
                                format_func={"previsao": "Previsão", "simulacao": "Simulação"}.get) or "previsao"
    if modo == "previsao":
        horizonte = c3.segmented_control("Horizonte (dias)", list(HORIZONTES), default=HORIZONTE_KPI, key="prev_horizonte") or HORIZONTE_KPI
        granularidade = c4.segmented_control("Granularidade", list(GRANULARIDADES), format_func=GRANULARIDADES.get,
                                             default="dia", key="prev_granularidade") or "dia"
        _render_previsao(resultado, alvo, int(horizonte), granularidade)
    else:
        _render_simulacao(resultado, file_id, versao, alvo)

    vencedor = NOMES_MODELOS.get(resultado["metadados"]["alvos"][alvo]["modelo_vencedor"])
    st.markdown(f"**Comparação de modelos · {ALVOS[alvo].lower()}** (vencedor: {vencedor})")
    _render_tabela_modelos(resultado, alvo)
    _render_b2b(resultado)
    _render_pendentes(file_id, versao, registro, df)
    _render_alertas(resultado, alvo)
