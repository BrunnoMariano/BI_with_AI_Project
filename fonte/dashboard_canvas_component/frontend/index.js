import React from "react";
import { createRoot } from "react-dom/client";
import { Responsive, WidthProvider } from "react-grid-layout";
import * as echarts from "echarts";
import { Streamlit } from "streamlit-component-lib";

const e = React.createElement;
const ResponsiveGridLayout = WidthProvider(Responsive);
const root = createRoot(document.getElementById("root"));

function emitValue(payload) {
  Streamlit.setComponentValue({
    event_id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    ...payload,
  });
}

function ChartSurface({ renderSpec }) {
  const containerRef = React.useRef(null);
  const chartRef = React.useRef(null);
  const resizeFrameRef = React.useRef(null);
  const heightFrameRef = React.useRef(null);

  React.useEffect(() => {
    const container = containerRef.current;
    if (!container || !renderSpec || renderSpec.render_mode !== "echarts") {
      if (chartRef.current) {
        chartRef.current.dispose();
        chartRef.current = null;
      }
      return undefined;
    }

    if (!chartRef.current) {
      chartRef.current = echarts.init(container, null, { renderer: "canvas" });
    }

    const observer = new ResizeObserver(() => {
      if (resizeFrameRef.current) {
        cancelAnimationFrame(resizeFrameRef.current);
      }
      resizeFrameRef.current = requestAnimationFrame(() => {
        if (chartRef.current) {
          chartRef.current.resize({
            animation: {
              duration: 0,
            },
          });
        }
      });
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      if (resizeFrameRef.current) {
        cancelAnimationFrame(resizeFrameRef.current);
        resizeFrameRef.current = null;
      }
      if (heightFrameRef.current) {
        cancelAnimationFrame(heightFrameRef.current);
        heightFrameRef.current = null;
      }
      if (chartRef.current) {
        chartRef.current.dispose();
        chartRef.current = null;
      }
    };
  }, [renderSpec?.render_mode]);

  React.useEffect(() => {
    if (!renderSpec || renderSpec.render_mode !== "echarts" || !chartRef.current) {
      return undefined;
    }

    chartRef.current.setOption(
      {
        animation: false,
        ...renderSpec.option,
      },
      true,
      true,
    );

    if (heightFrameRef.current) {
      cancelAnimationFrame(heightFrameRef.current);
    }
    heightFrameRef.current = requestAnimationFrame(() => {
      Streamlit.setFrameHeight();
    });

    return () => {
      if (heightFrameRef.current) {
        cancelAnimationFrame(heightFrameRef.current);
        heightFrameRef.current = null;
      }
    };
  }, [renderSpec]);

  if (!renderSpec) {
    return e("div", { className: "message-slot" }, "Sem configuração visual.");
  }

  if (renderSpec.render_mode === "table") {
    const columns = renderSpec.columns || [];
    const rows = renderSpec.rows || [];
    return e(
      "div",
      { className: "table-slot" },
      e(
        "table",
        { className: "dashboard-table" },
        e(
          "thead",
          null,
          e(
            "tr",
            null,
            columns.map((column) => e("th", { key: column }, column)),
          ),
        ),
        e(
          "tbody",
          null,
          rows.map((row, rowIndex) =>
            e(
              "tr",
              { key: `${rowIndex}` },
              columns.map((column) =>
                e("td", { key: `${rowIndex}-${column}` }, row[column] == null ? "-" : String(row[column])),
              ),
            ),
          ),
        ),
      ),
    );
  }

  if (renderSpec.render_mode === "kpi") {
    return e(
      "div",
      { className: "kpi-slot" },
      e("span", { className: "kpi-label" }, renderSpec.label || "Indicador"),
      e("div", { className: "kpi-value" }, renderSpec.value == null ? "-" : String(renderSpec.value)),
    );
  }

  if (renderSpec.render_mode === "error") {
    return e("div", { className: "message-slot" }, renderSpec.message || "Erro ao renderizar o visual.");
  }

  if (renderSpec.render_mode === "empty") {
    return e("div", { className: "message-slot" }, renderSpec.message || "Sem dados para exibir.");
  }

  return e("div", { ref: containerRef, className: "chart-slot" });
}

function CanvasCard({ item, selected, onSelectLocal, onSelectCommit, onAction }) {
  const visualType = (item.visual_type || "visual").toUpperCase();
  const handleAction = (actionType, event) => {
    event.preventDefault();
    event.stopPropagation();
    onAction(actionType, item.chart_id);
  };
  return e(
    "div",
    {
      className: `canvas-card${selected ? " selected" : ""}`,
      onMouseDownCapture: (event) => {
        if (event.target.closest(".card-actions")) {
          return;
        }
        onSelectLocal(item.chart_id);
      },
      onClick: (event) => {
        if (event.target.closest(".card-actions")) {
          return;
        }
        onSelectCommit(item.chart_id);
      },
    },
    e(
      "div",
      { className: "card-header card-drag-handle" },
      e(
        "div",
        { className: "card-title-wrap" },
        e("span", { className: "card-kicker" }, visualType),
        e(
          "div",
          { className: "card-title-row" },
          selected ? e("span", { className: "card-selected-dot", "aria-hidden": "true" }) : null,
          e("div", { className: "card-title" }, item.title || "Visual"),
        ),
      ),
      e(
        "div",
        { className: "card-drag-grip", "aria-hidden": "true" },
        e("span", null),
        e("span", null),
        e("span", null),
      ),
      e(
        "div",
        { className: "card-actions" },
        e(
          "button",
          {
            className: "card-action",
            onClick: (event) => handleAction("edit", event),
            type: "button",
            title: "Editar visual",
          },
          "Edit",
        ),
        e(
          "button",
          {
            className: "card-action",
            onClick: (event) => handleAction("duplicate", event),
            type: "button",
            title: "Duplicar visual",
          },
          "Copy",
        ),
        e(
          "button",
          {
            className: "card-action danger",
            onClick: (event) => handleAction("remove", event),
            type: "button",
            title: "Remover visual",
          },
          "Del",
        ),
      ),
    ),
    e(
      "div",
      { className: "card-body" },
      e(ChartSurface, { renderSpec: item.render_spec }),
    ),
  );
}

function DashboardCanvas(props) {
  const { args } = props;
  const items = args.items || [];
  const layouts = args.layouts || {};
  const editable = args.editable !== false;
  const selectedItemId = args.selected_item_id || null;
  const lastLayoutsRef = React.useRef(JSON.stringify(layouts || {}));
  const [currentBreakpoint, setCurrentBreakpoint] = React.useState("lg");
  const [localSelectedId, setLocalSelectedId] = React.useState(selectedItemId);
  const [localLayouts, setLocalLayouts] = React.useState(layouts || {});

  React.useEffect(() => {
    lastLayoutsRef.current = JSON.stringify(layouts || {});
    setLocalSelectedId(selectedItemId);
    setLocalLayouts(layouts || {});
    Streamlit.setFrameHeight();
  }, [items.length, JSON.stringify(layouts), selectedItemId]);

  const emitLayoutsIfChanged = React.useCallback(
    (nextLayouts, nextSelectedId = localSelectedId) => {
      const serialized = JSON.stringify(nextLayouts || {});
      if (serialized === lastLayoutsRef.current) {
        return;
      }
      lastLayoutsRef.current = serialized;
      emitValue({
        event_type: "layout_change",
        selected_item_id: nextSelectedId,
        layouts: nextLayouts,
        item_action: {},
      });
    },
    [localSelectedId],
  );

  if (!items.length) {
    return e(
      "div",
      {
        className: "canvas-shell",
        onMouseDown: () => {
          setLocalSelectedId(null);
          emitValue({
            event_type: "select",
            selected_item_id: null,
            layouts: localLayouts,
            item_action: {},
          });
        },
      },
      e(
        "div",
        { className: "canvas-empty" },
        e("strong", null, "Seu dashboard ainda está vazio"),
        e("div", null, "Crie o primeiro visual no painel lateral e ele aparecerá aqui."),
      ),
    );
  }

  return e(
    "div",
    {
      className: "canvas-shell",
      onMouseDown: (event) => {
        if (event.target.closest(".canvas-card") || event.target.closest(".react-resizable-handle")) {
          return;
        }
        setLocalSelectedId(null);
        emitValue({
          event_type: "select",
          selected_item_id: null,
          layouts: localLayouts,
          item_action: {},
        });
      },
    },
    e(
      ResponsiveGridLayout,
      {
        className: "layout",
        layouts: localLayouts,
        breakpoints: { lg: 1200, md: 768, sm: 0 },
        cols: { lg: 24, md: 12, sm: 4 },
        rowHeight: 64,
        margin: [14, 14],
        isDraggable: editable,
        isResizable: editable,
        resizeHandles: ["se"],
        compactType: "vertical",
        useCSSTransforms: true,
        preventCollision: false,
        draggableHandle: ".card-drag-handle",
        draggableCancel: ".card-actions,.card-action,button,.card-body,.chart-slot,.table-slot,.message-slot,.kpi-slot,.react-resizable-handle",
        onDragStart: (_layout, oldItem) => {
          const chartId = oldItem?.i || null;
          if (!chartId) {
            return;
          }
          setLocalSelectedId(chartId);
        },
        onResizeStart: (_layout, oldItem) => {
          const chartId = oldItem?.i || null;
          if (!chartId) {
            return;
          }
          setLocalSelectedId(chartId);
        },
        onBreakpointChange: (breakpoint) => setCurrentBreakpoint(breakpoint || "lg"),
        onDragStop: (_layout, _oldItem, _newItem, _placeholder, event, element) => {
          void event;
          void element;
          const nextLayouts = { ...localLayouts, [currentBreakpoint]: _layout };
          setLocalLayouts(nextLayouts);
          emitLayoutsIfChanged(nextLayouts, _newItem?.i || localSelectedId);
        },
        onResizeStop: (_layout, _oldItem, _newItem) => {
          const nextLayouts = { ...localLayouts, [currentBreakpoint]: _layout };
          setLocalLayouts(nextLayouts);
          emitLayoutsIfChanged(nextLayouts, _newItem?.i || localSelectedId);
        },
      },
      items.map((item) =>
        e(
          "div",
          {
            key: item.chart_id,
          },
          e(CanvasCard, {
            item,
            selected: localSelectedId === item.chart_id,
            onSelectLocal: (chartId) => {
              setLocalSelectedId(chartId);
            },
            onSelectCommit: (chartId) => {
              setLocalSelectedId(chartId);
              emitValue({
                event_type: "select",
                selected_item_id: chartId,
                layouts: localLayouts,
                item_action: {},
              });
            },
            onAction: (actionType, chartId) =>
              emitValue({
                event_type: "item_action",
                selected_item_id: chartId,
                layouts: localLayouts,
                item_action: { type: actionType, chart_id: chartId },
              }),
          }),
        ),
      ),
    ),
  );
}

function onRender(event) {
  const args = event.detail.args || {};
  root.render(e(DashboardCanvas, { args }));
  window.setTimeout(() => Streamlit.setFrameHeight(), 0);
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
Streamlit.setFrameHeight(640);
