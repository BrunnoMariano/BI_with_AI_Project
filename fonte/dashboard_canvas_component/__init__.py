import os

import streamlit.components.v1 as components


_COMPONENT_PATH = os.path.join(os.path.dirname(__file__), "frontend")
_dashboard_canvas = components.declare_component("dashboard_canvas", path=_COMPONENT_PATH)


def dashboard_canvas_component(
    items,
    layouts,
    selected_item_id=None,
    editable=True,
    key="dashboard_canvas",
):
    return _dashboard_canvas(
        items=items,
        layouts=layouts,
        selected_item_id=selected_item_id,
        editable=editable,
        default={
            "event_id": "",
            "event_type": "noop",
            "selected_item_id": selected_item_id,
            "layouts": layouts,
            "item_action": {},
        },
        key=key,
    )
