from typing import Any

import streamlit as st


def render_json(data: Any) -> None:
    st.json(data)


def render_error(error: Exception) -> None:
    st.error(str(error))


def render_result_path(data: dict) -> None:
    saved_result_path = data.get("saved_result_path")
    saved_csv_path = data.get("saved_csv_path")
    if saved_result_path:
        st.caption(f"JSON saved: {saved_result_path}")
    if saved_csv_path:
        st.caption(f"CSV saved: {saved_csv_path}")


def text_area_json(label: str, value: str, height: int = 180) -> str:
    return st.text_area(label, value=value, height=height)


def render_metric_row(metrics: dict[str, Any]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, value)
