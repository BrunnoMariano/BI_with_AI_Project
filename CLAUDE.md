# CLAUDE.md — Contexto do projeto (leia antes de qualquer alteração)

## 1. O que é este projeto

TCC de Engenharia de Computação: **"Desenvolvimento de uma plataforma de Business Intelligence (BI) integrada a Inteligência Artificial para análise preditiva de pedidos e vendas"**.

O sistema já existe e funciona como um BI conversacional em Streamlit: o usuário carrega um Excel/CSV (ou conecta um PostgreSQL), conversa em português com os dados (o GPT gera SQL e responde) e monta dashboards. **O objetivo agora é adicionar a camada de análise preditiva (Machine Learning)**, que ainda não existe: hoje todas as respostas descrevem o passado.

O autor está aprendendo programação. Ao final de cada tarefa, explique em linguagem simples o que foi feito, quais arquivos mudaram e como testar.

## 2. Como rodar

- Raiz do projeto: a pasta que contém `fonte/` e `requirements.txt`.
- Instalar: `pip install -r requirements.txt`
- Rodar: `streamlit run fonte/appLAB.py`
- O app exige `.streamlit/secrets.toml` com o bloco `[firebase]` (apiKey, authDomain, projectId, storageBucket, messagingSenderId, appId). Esse arquivo é secreto: **nunca leia em voz alta, imprima, copie ou faça commit dele**. A chave da OpenAI é digitada na barra lateral do app.
- O código de ML novo deve rodar e ser testável **sem Streamlit, sem Firebase e sem OpenAI** (funções puras recebendo DataFrame), para que os testes (`pytest`) funcionem offline.

## 3. Mapa do código existente

| Arquivo | Papel | Cuidado |
| --- | --- | --- |
| `fonte/appLAB.py` (~1.470 linhas) | App inteiro: login, sidebar, abas Conversa / Dashboard / Histórico, pipeline texto→SQL com retry, perfil do dataset (`detect_dataset_profile`, `build_dataset_profile`, `build_sql_system_prompt`) | Arquivo grande. Faça mudanças pontuais; lógica nova vai para `fonte/ml/` |
| `fonte/dashboard_service.py` | Visuais do dashboard (bar, line, area, pie, scatter, table, kpi), filtros globais, layouts, `build_echarts_spec`, `build_render_items` | O frontend repassa `option` direto ao ECharts |
| `fonte/dashboard_canvas_component/` | Componente React + ECharts (`bundle.js` já compilado) | **Não recompilar nem editar `bundle.js`** |
| `fonte/user_data.py` | Persistência do usuário: chat, arquivos, dashboards (Firestore `users/{uid}/...`) e leitura do arquivo (`load_dataframe_from_storage` usa `pd.read_excel`) | Reutilizar para salvar resultados de modelos |
| `fonte/firebase_service.py` | REST do Firebase Auth, Firestore e Storage (`set_document`, `get_document`) | Não alterar a autenticação |
| `fonte/auth.py`, `login_interface.py`, `config.py` | Login e configuração | Não mexer |
| `fonte/ui_styles.py` | Tema visual (`render_section_intro`, `render_app_hero`) | Usar nas telas novas para manter o padrão |
| `fonte/db_utils.py` | Introspecção de schema PostgreSQL | Não usado pelo ML |

Fluxo atual de um arquivo: upload na sidebar → Firebase Storage → `load_dataframe_from_storage` → `build_file_context` → perfil `client_sales_profile_v1` (base de vendas) ou `generic_file_v1`.

## 4. Os dados

Arquivo oficial e **único** (o cliente não enviará extrações novas): `vendas_cliente.xlsx`, aba `Worksheet`, 4.558 linhas, pedidos de certificações ANEC de 05/09/2024 a 07/04/2026. Guardar em `data/vendas_cliente.xlsx` (pasta fora do Git).

Colunas (nomes exatos, lidos por `pd.read_excel`):

| Coluna | Conteúdo | Observação |
| --- | --- | --- |
| `Número` | ID do pedido | Único |
| `Conta` | Empresa compradora | Preenchida = venda **B2B** (138 linhas) |
| `Comprador` | Nome do cliente | Chave de cliente (normalizar lower/trim) |
| `Produto` | 10 produtos | 1 nulo (pedido de teste R$ 0,00) |
| `Valor Pago` | Texto `R$1.234,56` | Converter para float |
| `Nome Pagamento` | Nome do pagador | Só existe se pago |
| `Forma de pagamento` | pix, cartão de crédito, boleto | Só existe se pago |
| `Link de pagamento` | `Link de pagamento` ou `---` | Etapa do funil |
| `Status Venda` | pago, pendente, expirado, cancelado, aguardando pagamento | Minúsculas |
| `Promotora` | Texto livre, 706 grafias | Normalizar (BEVI/Bevi/bevicred…, "Não se aplica" → vazio) |
| `Cupom` | 34 códigos | 52% preenchido |
| `Telefone/Celular`, `CEP`, `Logradouro`, `Número.1`, `Bairro`, `Cidade`, `Estado` | Contato e endereço | `Número.1` = número do endereço |
| `Data Venda` | Texto `dd/mm/aaaa hh:mm:ss` | Data do pagamento |
| `Criado em` | Texto `dd/mm/aaaa hh:mm:ss` | Data de criação do pedido |

Esta extração **já ativa** o perfil `client_sales_profile_v1` (colunas exigidas: `Produto`, `Valor Pago`, `Status Venda`, `Forma de pagamento`, `Data Venda`, `Criado em`). Não quebre essa detecção.

Problemas conhecidos: 14 pedidos com R$ 0,00; 22 pedidos com `Data Venda` anterior a `Criado em`; 4 pagos sem `Data Venda`; mesmo comprador cria vários pedidos até pagar; abril/2026 incompleto (até dia 7).

**Números de conferência (use em testes):** 4.558 linhas · 2.919 pagos · receita paga R$ 308.783,48 · 82 pedidos pagos B2B somando R$ 88.410,08.

**Dados pessoais (LGPD):** a planilha tem nomes, telefones e endereços reais. Nunca fazer commit do `.xlsx`, nunca imprimir linhas com dados pessoais em logs, e usar cópia anonimizada em exemplos e figuras.

## 5. Decisões já tomadas (não mudar sem perguntar)

1. **Venda = `Status Venda` = pago.**
2. **Data de referência das séries = `Criado em`** (não `Data Venda`).
3. **Previsão só com vendas B2C** (`Conta` vazia). B2B aparece apenas como bloco descritivo "Vendas corporativas" (histórico e média), sem previsão própria: são só 23 pedidos na janela de treino.
4. **Janela de treino a partir de 01/08/2025** (antes disso o volume é baixo e dominado por B2B). Na implementação, detectar automaticamente o primeiro mês com ≥ 100 pedidos pagos.
5. **Último dia incompleto é descartado**; a série termina em 06/04/2026 e a previsão parte de 07/04/2026.
6. **O ML calcula, o LLM explica.** Números de previsão vêm sempre do pipeline Python; o GPT só recebe esses resultados em JSON e os descreve. Nunca pedir ao LLM que calcule ou estime valores futuros.
7. **Validação sempre temporal**, nunca aleatória. Previsão: backtest com origem deslizante, 4 cortes de 28 dias, métrica principal WAPE (também MAE, RMSE, erro no total, cobertura do intervalo). Reportar a **média dos cortes**.
8. **Vazamento proibido no classificador de pagamento:** não usar `Promotora`, endereço (`CEP`…`Estado`), `Forma de pagamento`, `Nome Pagamento`, `Data Venda`. Esses campos são preenchidos no checkout e revelam o resultado.
9. Bibliotecas de ML: `scikit-learn`, `statsmodels`, `lightgbm`, `joblib` (e `holidays` para feriados). Sem Prophet.

Resultados de referência do protótipo (série diária B2C, 4 cortes de 28 dias, 16/12/2025–06/04/2026): Holt-Winters (tendência amortecida, sazonalidade 7 dias) WAPE médio 51% na receita; baseline sazonal 7 dias 62%; média móvel 28 dias 73%. Classificador de pagamento sem vazamento: AUC ≈ 0,73. Se seus resultados divergirem muito disso, investigue antes de seguir.

## 6. Arquitetura alvo

Toda lógica nova em `fonte/ml/`; `appLAB.py` só chama funções e exibe.

```
fonte/ml/
  schema.py         # COLUMN_ALIASES, detect_schema, to_canonical
  preprocessing.py  # parse_brl, parse_datetime_br, normalize_promotora, clean_orders
  features.py       # build_daily_series, build_order_features, calendar_features
  forecasting.py    # baselines, holt_winters, lightgbm, backtest, fit_best, forecast
  conversion.py     # train_conversion, score_pending          (Fase 6)
  anomaly.py        # detect_anomalies                         (Fase 6)
  metrics.py        # mae, rmse, wape, cobertura, auc
  registry.py       # salvar/carregar resultados (joblib + Firestore users/{uid}/models/{source_key})
  pipeline.py       # run_predictive_pipeline(df_raw, source_key) -> dict
tests/              # pytest, um arquivo por módulo
notebooks/01_eda.ipynb
data/               # planilha (fora do Git)
```

## 7. Fases e estado atual

Marque `[x]` ao concluir cada fase.

- [x] **Fase 0** — Preparação: `fonte/ml/`, `tests/`, `notebooks/`, `data/`; adicionar `data/`, `*.xlsx` e `*.joblib` ao `.gitignore`; dependências no `requirements.txt`.
- [ ] **Fase 1** — Esquema e limpeza (`schema.py`, `preprocessing.py`), perfil do dataset usando `detect_schema`.
- [ ] **Fase 2** — Previsão de vendas (`features.py`, `forecasting.py`, `metrics.py`, `registry.py`, `pipeline.py`) com backtest e intervalos de 80%/95%.
- [ ] **Fase 3** — Aba "Previsões" em `appLAB.py`: KPIs de 28 dias, gráfico real + previsto + faixa, seletores, tabela de modelos, modo simulação (data de corte escolhida → previsto × real), botão Retreinar, download CSV.
- [ ] **Fase 4** — IA: intenção `forecast_query` em `classify_intent`, `explain_forecast`, resumo executivo automático, tabela `previsao` no pandasql.
- [ ] **Fase 5** — Visual "Previsão" no Dashboard (`dashboard_service.py`), sem SQL do LLM.
- [ ] **Fase 6** — Probabilidade de pagamento dos pendentes e alertas de anomalia.

Implemente uma fase por vez. Não adiante fases futuras sem pedido.

## 8. Regras de trabalho

- Antes de editar, leia os arquivos que serão alterados. Mudanças pequenas e focadas; não reformate nem reorganize código que não faz parte da tarefa.
- Interface e mensagens em português do Brasil.
- Toda função nova de ML com docstring curta e pelo menos um teste. Rode `pytest` ao terminar cada fase e mostre o resultado.
- Ao final de cada fase, confirme que `streamlit run fonte/appLAB.py` ainda abre sem erro e que o chat e o dashboard continuam funcionando.
- Use `st.cache_data` para não retreinar a cada interação; retreino só ao trocar de arquivo ou clicar em "Retreinar".
- Não altere login, Firebase Auth, `bundle.js` nem a conexão PostgreSQL.
- Bug conhecido a corrigir na Fase 4: em `appLAB.py`, `temperature=1.0 if 'o' in model_name` deveria checar o prefixo do modelo (`model_name.startswith(("o1", "o3", "o4"))`).
- Se algo no pedido contradizer este arquivo, pare e pergunte.
