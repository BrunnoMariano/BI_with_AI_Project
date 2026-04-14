import copy
import hashlib
import json
import math
from typing import Any
from uuid import uuid4

import pandas as pd
import pandasql as ps
from sqlalchemy import create_engine, text

from user_data import get_dashboard_record, save_dashboard_record


DASHBOARD_BREAKPOINT_COLS = {"lg": 24, "md": 12, "sm": 4}
DASHBOARD_LAYOUT_DEFAULT = {"w": 8, "h": 6, "minW": 4, "minH": 4}


def build_source_descriptor(
    db_mode: bool,
    db_creds: dict[str, Any] | None = None,
    selected_file_id: str | None = None,
    selected_file_name: str = "",
) -> dict[str, str] | None:
    if db_mode:
        creds = db_creds or {}
        host = str(creds.get("host", "")).strip()
        port = str(creds.get("port", "")).strip()
        dbname = str(creds.get("dbname", "")).strip()
        if not (host and port and dbname):
            return None
        raw_key = f"database:{host}:{port}:{dbname}".lower()
        return {
            "source_type": "database",
            "source_key": raw_key,
            "source_label": f"{dbname} @ {host}:{port}",
        }

    if not selected_file_id:
        return None
    return {
        "source_type": "file",
        "source_key": f"file:{selected_file_id}",
        "source_label": selected_file_name or "Arquivo selecionado",
    }


def build_dashboard_id(source_key: str) -> str:
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:24]


def default_dashboard_record(source_type: str, source_key: str, source_label: str) -> dict[str, Any]:
    return {
        "dashboard_id": build_dashboard_id(source_key),
        "source_type": source_type,
        "source_key": source_key,
        "title": f"Dashboard - {source_label}",
        "items": [],
        "layouts": {breakpoint: [] for breakpoint in DASHBOARD_BREAKPOINT_COLS},
    }


def load_dashboard(user_id: str, source_key: str, source_type: str, source_label: str) -> dict[str, Any]:
    dashboard_id = build_dashboard_id(source_key)
    record = get_dashboard_record(dashboard_id, user_id=user_id)
    if not record:
        return default_dashboard_record(source_type, source_key, source_label)

    record.setdefault("dashboard_id", dashboard_id)
    record.setdefault("source_type", source_type)
    record.setdefault("source_key", source_key)
    record.setdefault("title", f"Dashboard - {source_label}")
    record.setdefault("items", [])
    record["layouts"] = normalize_layouts(record.get("items", []), record.get("layouts"))
    record["items"] = _sync_item_layouts(record.get("items", []), record["layouts"])
    return record


def save_dashboard(user_id: str, source_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    dashboard_id = build_dashboard_id(source_key)
    data = dict(payload)
    data["dashboard_id"] = dashboard_id
    data["source_key"] = source_key
    data["layouts"] = normalize_layouts(data.get("items", []), data.get("layouts"))
    data["items"] = _sync_item_layouts(data.get("items", []), data["layouts"])
    return save_dashboard_record(dashboard_id, data, user_id=user_id)


def _default_layout_for_index(index: int) -> dict[str, int]:
    base_w = DASHBOARD_LAYOUT_DEFAULT["w"]
    base_h = DASHBOARD_LAYOUT_DEFAULT["h"]
    cols = DASHBOARD_BREAKPOINT_COLS["lg"]
    per_row = max(1, cols // base_w)
    x = (index % per_row) * base_w
    y = (index // per_row) * base_h
    return {
        "x": x,
        "y": y,
        "w": base_w,
        "h": base_h,
        "minW": DASHBOARD_LAYOUT_DEFAULT["minW"],
        "minH": DASHBOARD_LAYOUT_DEFAULT["minH"],
    }


def normalize_layouts(items: list[dict[str, Any]], layouts: dict[str, list[dict[str, Any]]] | None) -> dict[str, list[dict[str, Any]]]:
    layouts = copy.deepcopy(layouts or {})
    item_index = {item["chart_id"]: idx for idx, item in enumerate(items)}
    normalized: dict[str, list[dict[str, Any]]] = {}

    for breakpoint, cols in DASHBOARD_BREAKPOINT_COLS.items():
        existing_map = {entry.get("i"): dict(entry) for entry in layouts.get(breakpoint, []) if entry.get("i")}
        entries: list[dict[str, Any]] = []
        for item in items:
            chart_id = item["chart_id"]
            existing = existing_map.get(chart_id, {})
            default_layout = item.get("layout") or _default_layout_for_index(item_index[chart_id])

            if breakpoint == "lg":
                width = int(existing.get("w", default_layout.get("w", DASHBOARD_LAYOUT_DEFAULT["w"])))
                min_width = int(existing.get("minW", default_layout.get("minW", DASHBOARD_LAYOUT_DEFAULT["minW"])))
            elif breakpoint == "md":
                derived_width = min(cols, max(4, math.ceil(int(default_layout.get("w", DASHBOARD_LAYOUT_DEFAULT["w"])) / 2)))
                derived_min_width = min(derived_width, max(2, math.ceil(int(default_layout.get("minW", DASHBOARD_LAYOUT_DEFAULT["minW"])) / 2)))
                width = int(existing.get("w", derived_width))
                min_width = int(existing.get("minW", derived_min_width))
            else:
                width = int(existing.get("w", cols))
                min_width = int(existing.get("minW", cols))

            height = int(existing.get("h", default_layout.get("h", DASHBOARD_LAYOUT_DEFAULT["h"])))
            min_height = int(existing.get("minH", default_layout.get("minH", DASHBOARD_LAYOUT_DEFAULT["minH"])))
            x = int(existing.get("x", default_layout.get("x", 0)))
            y = int(existing.get("y", default_layout.get("y", item_index[chart_id] * DASHBOARD_LAYOUT_DEFAULT["h"])))

            width = max(1, min(width, cols))
            min_width = max(1, min(min_width, width))
            x = min(max(0, x), max(0, cols - width))

            entries.append(
                {
                    "i": chart_id,
                    "x": x,
                    "y": max(0, y),
                    "w": width,
                    "h": max(2, height),
                    "minW": min_width,
                    "minH": max(2, min_height),
                }
            )
        normalized[breakpoint] = entries
    return normalized


def build_default_layouts(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return normalize_layouts(items, None)


def _sync_item_layouts(items: list[dict[str, Any]], layouts: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    lg_layout_map = {entry.get("i"): entry for entry in layouts.get("lg", []) if entry.get("i")}
    synced_items = []
    for item in items:
        item_copy = copy.deepcopy(item)
        if item_copy.get("chart_id") in lg_layout_map:
            layout_entry = lg_layout_map[item_copy["chart_id"]]
            item_copy["layout"] = {
                "x": layout_entry.get("x", 0),
                "y": layout_entry.get("y", 0),
                "w": layout_entry.get("w", DASHBOARD_LAYOUT_DEFAULT["w"]),
                "h": layout_entry.get("h", DASHBOARD_LAYOUT_DEFAULT["h"]),
                "minW": layout_entry.get("minW", DASHBOARD_LAYOUT_DEFAULT["minW"]),
                "minH": layout_entry.get("minH", DASHBOARD_LAYOUT_DEFAULT["minH"]),
            }
        synced_items.append(item_copy)
    return synced_items


def build_visual_system_prompt(db_mode: bool, profile: dict[str, Any] | None = None) -> str:
    if db_mode:
        dialect_instructions = (
            "4. A consulta deve ser compatível com PostgreSQL.\n"
            "5. Para agregações por mês, use to_char em colunas de data.\n"
        )
    else:
        dialect_instructions = (
            "4. A consulta deve ser compatível com SQLite.\n"
            "5. A tabela para consulta é sempre df.\n"
            "6. Para agregações por mês, use substr/strftime compatíveis com SQLite.\n"
        )
        if profile and profile.get("profile_id") == "client_sales_profile_v1":
            dialect_instructions += (
                "7. Se o pedido envolver faturamento/receita, converta Valor Pago de texto monetário para número.\n"
                "8. Em pedidos genéricos sobre vendas, considere status pago como padrão.\n"
                "9. Em análises temporais, prefira Data Venda.\n"
            )

    prompt = (
        "Você é uma especialista sênior em analytics. Sua tarefa é traduzir o pedido do usuário "
        "em uma especificação de visual para dashboard.\n\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. Respeite o pedido do usuário se ele escolher explicitamente o tipo do visual.\n"
        "2. Se o pedido for vago, escolha a visualização mais útil para dashboard.\n"
        "3. Use apenas tabelas e colunas existentes no esquema fornecido.\n"
        f"{dialect_instructions}"
        "10. Retorne exatamente no formato abaixo, sem markdown e sem explicações.\n"
        "11. Tipos aceitos: bar, line, area, pie, scatter, table, kpi.\n"
        "12. Dimension é a coluna categórica ou temporal principal.\n"
        "13. Metric é a medida principal.\n"
        "14. Series é opcional e pode ficar vazio.\n\n"
        "FORMATO:\n"
        "Query: <sql>\n"
        "VisualType: <tipo>\n"
        "Dimension: <coluna ou vazio>\n"
        "Metric: <coluna ou alias ou vazio>\n"
        "Series: <coluna ou vazio>\n"
        "Title: <titulo>\n"
    )
    return prompt


def generate_visual_instructions(
    client,
    model_name: str,
    schema_text: str,
    question: str,
    db_mode: bool,
    profile: dict[str, Any] | None = None,
    df_preview: pd.DataFrame | None = None,
) -> str:
    preview_text = ""
    if df_preview is not None and not df_preview.empty:
        preview_text = df_preview.head(5).to_markdown(index=False)
    messages = [
        {"role": "system", "content": build_visual_system_prompt(db_mode, profile)},
        {
            "role": "user",
            "content": f"Esquema:\n{schema_text}\n\nPreview:\n{preview_text}\n\nPedido:\n{question}",
        },
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=1.0 if "o" in model_name else 0.0,
        max_completion_tokens=900,
    )
    return response.choices[0].message.content.strip()


def parse_visual_instructions(text: str) -> dict[str, str | None]:
    result = {
        "query_text": "",
        "visual_type": "bar",
        "dimension": "",
        "metric": "",
        "series": "",
        "title": "Novo visual",
    }
    current_key = None
    query_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("query:"):
            current_key = "query_text"
            value = raw_line.split(":", 1)[1].strip()
            if value:
                query_lines.append(value)
        elif lower.startswith("visualtype:"):
            current_key = None
            result["visual_type"] = raw_line.split(":", 1)[1].strip().lower() or "bar"
        elif lower.startswith("dimension:"):
            current_key = None
            result["dimension"] = raw_line.split(":", 1)[1].strip()
        elif lower.startswith("metric:"):
            current_key = None
            result["metric"] = raw_line.split(":", 1)[1].strip()
        elif lower.startswith("series:"):
            current_key = None
            result["series"] = raw_line.split(":", 1)[1].strip()
        elif lower.startswith("title:"):
            current_key = None
            result["title"] = raw_line.split(":", 1)[1].strip() or "Novo visual"
        elif current_key == "query_text":
            query_lines.append(raw_line.rstrip())

    result["query_text"] = "\n".join(query_lines).strip().strip("`")
    for key in ("dimension", "metric", "series", "title"):
        if isinstance(result[key], str):
            result[key] = result[key].strip()
    return result


def run_visual_query(
    query_text: str,
    db_mode: bool,
    uri: str | None = None,
    df_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if db_mode:
        if not uri:
            raise ValueError("Conexao de banco indisponivel para executar o visual.")
        engine = create_engine(uri)
        try:
            return pd.read_sql(text(query_text), engine)
        finally:
            engine.dispose()

    if df_data is None:
        raise ValueError("DataFrame indisponivel para executar o visual.")
    return ps.sqldf(query_text, {"df": df_data})


def _strip_quotes(value: str | None) -> str | None:
    if not isinstance(value, str):
        return value
    return value.strip().strip('"').strip("'")


def _first_non_empty(values: list[str | None], df: pd.DataFrame) -> str | None:
    for value in values:
        candidate = _strip_quotes(value)
        if candidate and candidate in df.columns:
            return candidate
    return None


def _coerce_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def build_echarts_spec(df_result: pd.DataFrame, visual_spec: dict[str, Any]) -> dict[str, Any]:
    df = df_result.copy()
    if df.empty:
        return {"render_mode": "empty", "message": "Sem dados para exibir."}

    visual_type = str(visual_spec.get("visual_type", "bar")).lower().strip()
    if visual_type not in {"bar", "line", "area", "pie", "scatter", "table", "kpi"}:
        visual_type = "bar"

    title = visual_spec.get("title") or "Novo visual"
    dimension = _first_non_empty([visual_spec.get("dimension")], df)
    metric = _first_non_empty([visual_spec.get("metric")], df)
    series = _first_non_empty([visual_spec.get("series")], df)

    if visual_type == "kpi":
        metric = metric or _first_non_empty(list(df.columns), df)
        value = _coerce_scalar(df.iloc[0][metric]) if metric else _coerce_scalar(df.iloc[0, 0])
        return {
            "render_mode": "kpi",
            "title": title,
            "value": value,
            "label": metric or "Indicador",
        }

    if visual_type == "table":
        return {
            "render_mode": "table",
            "title": title,
            "columns": list(df.columns),
            "rows": _json_safe(df.head(100)),
        }

    if visual_type == "pie":
        dimension = dimension or df.columns[0]
        metric = metric or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
        data = [
            {"name": _coerce_scalar(row[dimension]), "value": _coerce_scalar(row[metric])}
            for _, row in df.iterrows()
        ]
        return {
            "render_mode": "echarts",
            "title": title,
            "option": {
                "backgroundColor": "transparent",
                "tooltip": {"trigger": "item"},
                "legend": {"bottom": 0, "textStyle": {"color": "#c7d1df"}},
                "series": [
                    {
                        "type": "pie",
                        "radius": ["38%", "70%"],
                        "avoidLabelOverlap": True,
                        "label": {"color": "#e8edf6"},
                        "data": data,
                    }
                ],
            },
        }

    dimension = dimension or df.columns[0]
    metric = metric or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    categories = [_coerce_scalar(value) for value in df[dimension].tolist()] if dimension in df.columns else []
    values = [_coerce_scalar(value) for value in df[metric].tolist()] if metric in df.columns else []

    if series and series in df.columns:
        grouped_data: dict[str, list[dict[str, Any]]] = {}
        for _, row in df.iterrows():
            group_name = str(_coerce_scalar(row[series]) or "Série")
            grouped_data.setdefault(group_name, []).append(
                {
                    "name": _coerce_scalar(row[dimension]) if dimension in df.columns else "",
                    "value": _coerce_scalar(row[metric]) if metric in df.columns else None,
                    "x": _coerce_scalar(row[dimension]) if dimension in df.columns else None,
                    "y": _coerce_scalar(row[metric]) if metric in df.columns else None,
                }
            )
        series_config = []
        for group_name, points in grouped_data.items():
            if visual_type == "scatter":
                series_config.append({"name": group_name, "type": "scatter", "data": [[point["x"], point["y"]] for point in points]})
            else:
                series_entry = {
                    "name": group_name,
                    "type": "line" if visual_type in {"line", "area"} else "bar",
                    "data": [point["value"] for point in points],
                }
                if visual_type == "area":
                    series_entry["areaStyle"] = {}
                    series_entry["smooth"] = True
                if visual_type == "line":
                    series_entry["smooth"] = True
                series_config.append(series_entry)
        option = {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis"},
            "legend": {"top": 0, "textStyle": {"color": "#c7d1df"}},
            "grid": {"left": 16, "right": 16, "top": 54, "bottom": 24, "containLabel": True},
            "xAxis": {
                "type": "category" if visual_type != "scatter" else "value",
                "data": categories if visual_type != "scatter" else None,
                "axisLabel": {"color": "#aab7c9"},
                "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.14)"}},
            },
            "yAxis": {
                "type": "value",
                "axisLabel": {"color": "#aab7c9"},
                "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
            },
            "series": series_config,
        }
        return {"render_mode": "echarts", "title": title, "option": option}

    if visual_type == "scatter":
        scatter_data = [[_coerce_scalar(row[dimension]), _coerce_scalar(row[metric])] for _, row in df.iterrows()]
        option = {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "item"},
            "grid": {"left": 16, "right": 16, "top": 46, "bottom": 24, "containLabel": True},
            "xAxis": {
                "type": "value",
                "axisLabel": {"color": "#aab7c9"},
                "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
            },
            "yAxis": {
                "type": "value",
                "axisLabel": {"color": "#aab7c9"},
                "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
            },
            "series": [{"type": "scatter", "symbolSize": 12, "data": scatter_data}],
        }
        return {"render_mode": "echarts", "title": title, "option": option}

    series_entry = {
        "type": "line" if visual_type in {"line", "area"} else "bar",
        "data": values,
        "itemStyle": {"borderRadius": [8, 8, 0, 0]},
    }
    if visual_type == "line":
        series_entry["smooth"] = True
    if visual_type == "area":
        series_entry["areaStyle"] = {}
        series_entry["smooth"] = True

    option = {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 16, "right": 16, "top": 46, "bottom": 24, "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": categories,
            "axisLabel": {"color": "#aab7c9"},
            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.14)"}},
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"color": "#aab7c9"},
            "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
        },
        "series": [series_entry],
    }
    return {"render_mode": "echarts", "title": title, "option": option}


def build_dashboard_item(
    prompt: str,
    visual_spec: dict[str, Any],
    echarts_spec: dict[str, Any],
    chart_id: str | None = None,
    title_override: str = "",
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    title = title_override.strip() if title_override else str(visual_spec.get("title") or "Novo visual")
    item = {
        "chart_id": chart_id or uuid4().hex,
        "title": title,
        "prompt": prompt,
        "visual_type": str(visual_spec.get("visual_type", "bar")).lower(),
        "query_text": visual_spec.get("query_text", ""),
        "data_binding": {
            "dimension": visual_spec.get("dimension", ""),
            "metric": visual_spec.get("metric", ""),
            "series": visual_spec.get("series", ""),
        },
        "echarts_spec": _json_safe(echarts_spec),
        "layout": layout or {},
    }
    return item


def upsert_dashboard_item(record: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    new_record = copy.deepcopy(record)
    items = new_record.get("items", [])
    replaced = False
    for idx, existing in enumerate(items):
        if existing.get("chart_id") == item["chart_id"]:
            items[idx] = item
            replaced = True
            break
    if not replaced:
        default_layout = _default_layout_for_index(len(items))
        item = dict(item)
        item["layout"] = item.get("layout") or default_layout
        items.append(item)

    new_record["items"] = items
    new_record["layouts"] = normalize_layouts(items, new_record.get("layouts"))
    return new_record


def delete_dashboard_item(record: dict[str, Any], chart_id: str) -> dict[str, Any]:
    new_record = copy.deepcopy(record)
    new_items = [item for item in new_record.get("items", []) if item.get("chart_id") != chart_id]
    new_record["items"] = new_items
    new_record["layouts"] = normalize_layouts(new_items, new_record.get("layouts"))
    return new_record


def duplicate_dashboard_item(record: dict[str, Any], chart_id: str) -> dict[str, Any]:
    source_item = next((item for item in record.get("items", []) if item.get("chart_id") == chart_id), None)
    if not source_item:
        return record

    duplicated = copy.deepcopy(source_item)
    duplicated["chart_id"] = uuid4().hex
    duplicated["title"] = f"{source_item.get('title', 'Visual')} (cópia)"
    layout = dict(source_item.get("layout") or {})
    layout["x"] = int(layout.get("x", 0)) + 1
    layout["y"] = int(layout.get("y", 0)) + 1
    duplicated["layout"] = layout
    return upsert_dashboard_item(record, duplicated)


def update_dashboard_layouts(record: dict[str, Any], layouts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    new_record = copy.deepcopy(record)
    new_record["layouts"] = normalize_layouts(new_record.get("items", []), layouts)
    new_record["items"] = _sync_item_layouts(new_record.get("items", []), new_record["layouts"])
    return new_record


def reset_dashboard_layouts(record: dict[str, Any]) -> dict[str, Any]:
    new_record = copy.deepcopy(record)
    new_record["layouts"] = build_default_layouts(new_record.get("items", []))
    return new_record


def build_render_items(
    dashboard_record: dict[str, Any],
    db_mode: bool,
    uri: str | None = None,
    df_data: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    items_to_render = []
    for item in dashboard_record.get("items", []):
        payload = {
            "chart_id": item.get("chart_id"),
            "title": item.get("title", "Visual"),
            "visual_type": item.get("visual_type", "bar"),
            "prompt": item.get("prompt", ""),
        }
        try:
            stored_spec = item.get("echarts_spec")
            if isinstance(stored_spec, dict) and stored_spec.get("render_mode"):
                payload["render_spec"] = _json_safe(stored_spec)
            else:
                df_result = run_visual_query(item.get("query_text", ""), db_mode=db_mode, uri=uri, df_data=df_data)
                spec = build_echarts_spec(
                    df_result,
                    {
                        "visual_type": item.get("visual_type"),
                        "dimension": item.get("data_binding", {}).get("dimension"),
                        "metric": item.get("data_binding", {}).get("metric"),
                        "series": item.get("data_binding", {}).get("series"),
                        "title": item.get("title"),
                    },
                )
                payload["render_spec"] = _json_safe(spec)
            payload["status"] = "ready"
        except Exception as exc:
            payload["render_spec"] = {
                "render_mode": "error",
                "message": str(exc),
            }
            payload["status"] = "error"
        items_to_render.append(payload)
    return items_to_render


def dump_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False)
