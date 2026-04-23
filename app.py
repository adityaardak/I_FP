from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit.errors import StreamlitAPIException

try:
    import altair as alt

    ALTAIR_AVAILABLE = True
except Exception:
    alt = None
    ALTAIR_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    IsolationForest = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

from utils.barcode_utils import (
    BarcodeScanResult,
    LiveBarcodeProcessor,
    build_barcode_log_record,
    get_barcode_decoder_status,
    guess_lookup_columns,
    load_barcode_logs,
    load_order_dataframe,
    lookup_orders_detailed,
    normalize_code,
    retry_decode_crop,
    save_barcode_log,
    scan_barcode_image,
)
from utils.powerbi_utils import (
    PowerBIPage,
    build_page_embed_url,
    build_powerbi_embed_html,
    get_report_pages,
    load_powerbi_settings,
)

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


APP_DIR = Path(__file__).resolve().parent
PBIX_PATH = next(APP_DIR.glob("*.pbix"), APP_DIR / "dashboard project AICW- final (1)- Ishita Singhal.pbix")
BARCODE_LOG_DIR = APP_DIR / ".barcode_logs"


st.set_page_config(
    page_title="Power BI Risk Console",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_app_style() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(15, 118, 110, 0.08), transparent 28%),
                    linear-gradient(180deg, #f8fafc 0%, #eef6f6 100%);
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
            }
            .app-hero {
                background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%);
                color: white;
                border-radius: 24px;
                padding: 1.35rem 1.5rem;
                margin-bottom: 1rem;
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
            }
            .info-card, .metric-card, .why-box {
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #dbe7e8;
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            }
            .metric-card .card-title {
                font-size: 0.84rem;
                color: #4b5563;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.45rem;
                font-weight: 700;
            }
            .metric-card .card-value {
                font-size: 1.05rem;
                color: #0f172a;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }
            .metric-card .card-note {
                color: #475569;
                line-height: 1.45;
                font-size: 0.92rem;
            }
            .status-chip {
                display: inline-block;
                padding: 0.2rem 0.55rem;
                border-radius: 999px;
                background: #ccfbf1;
                color: #115e59;
                font-size: 0.83rem;
                font-weight: 600;
            }
            .warn-chip {
                background: #ffedd5;
                color: #9a3412;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_secret_overrides() -> dict[str, str]:
    overrides: dict[str, str] = {}
    try:
        secrets_obj = dict(st.secrets)
    except Exception:
        return overrides

    for key in (
        "POWERBI_IFRAME_URL",
        "POWERBI_REPORT_ID",
        "POWERBI_TENANT_ID",
        "POWERBI_GROUP_ID",
        "POWERBI_CLIENT_ID",
        "POWERBI_CLIENT_SECRET",
        "POWERBI_EMBED_URL",
        "POWERBI_EMBED_TOKEN",
        "POWERBI_ACCESS_TOKEN",
        "POWERBI_TOKEN_TYPE",
        "POWERBI_PAGES_JSON",
    ):
        if key in secrets_obj:
            overrides[key] = str(secrets_obj[key])

    nested = secrets_obj.get("powerbi")
    if isinstance(nested, dict):
        for key, value in nested.items():
            overrides[f"POWERBI_{str(key).upper()}"] = str(value)
    return overrides


def ensure_session_defaults(pages: list[PowerBIPage]) -> None:
    default_page = next((page for page in pages if page.is_active), pages[0] if pages else None)
    st.session_state.setdefault("selected_page_name", default_page.name if default_page else "")
    st.session_state.setdefault("decoded_code", "")
    st.session_state.setdefault("last_scan_result", None)
    st.session_state.setdefault("barcode_retry_result", None)
    st.session_state.setdefault("barcode_history", [])
    st.session_state.setdefault("barcode_lookup_source", "manual")
    st.session_state.setdefault("order_dataframe", None)
    st.session_state.setdefault("order_file_name", "")
    st.session_state.setdefault("last_lookup_context", {})
    st.session_state.setdefault("anomaly_identifier_column", "")
    st.session_state.setdefault("anomaly_code_columns", [])
    st.session_state.setdefault("anomaly_numeric_columns", [])
    st.session_state.setdefault("anomaly_categorical_columns", [])
    st.session_state.setdefault("anomaly_date_column", "")
    st.session_state.setdefault("anomaly_contamination", 0.08)
    st.session_state.setdefault("anomaly_row_limit", 15)
    st.session_state.setdefault("anomaly_filter_text", "")
    st.session_state.setdefault("anomaly_date_range", ())


def clean_text(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
    except Exception:
        if value is None:
            return ""
    return " ".join(str(value).strip().split())


def column_key(value: str) -> str:
    return normalize_code(value).lower()


def format_float(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value) or np.isinf(value):
            return "n/a"
    except Exception:
        pass
    return f"{float(value):.2f}"


def format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def render_card(title: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
        <div class="card-note">{note}</div>
    </div>
    """


def get_current_page(pages: list[PowerBIPage]) -> PowerBIPage | None:
    selected_name = st.session_state.get("selected_page_name")
    current = next((page for page in pages if page.name == selected_name), None)
    if current:
        return current
    if pages:
        st.session_state.selected_page_name = pages[0].name
        return pages[0]
    return None


def get_loaded_order_dataframe() -> pd.DataFrame | None:
    dataframe = st.session_state.get("order_dataframe")
    if isinstance(dataframe, pd.DataFrame):
        return dataframe
    return None


def _rank_column(columns: list[str], keyword_groups: list[list[str]], exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    best_column = None
    best_score = -1
    for column in columns:
        if column in exclude:
            continue
        normalized = column_key(column)
        score = 0
        for group_index, group in enumerate(keyword_groups):
            group_score = max(5, 100 - group_index * 8)
            for keyword in group:
                keyword_key = column_key(keyword)
                if normalized == keyword_key:
                    score = max(score, group_score + 12)
                elif normalized.startswith(keyword_key):
                    score = max(score, group_score + 6)
                elif keyword_key in normalized:
                    score = max(score, group_score)
        if score > best_score:
            best_score = score
            best_column = column
    return best_column


def _rank_columns(
    dataframe: pd.DataFrame,
    keyword_groups: list[list[str]],
    limit: int = 5,
    exclude: set[str] | None = None,
) -> list[str]:
    exclude = exclude or set()
    ranked: list[tuple[int, str]] = []
    for column in dataframe.columns:
        if column in exclude:
            continue
        normalized = column_key(column)
        score = 0
        for group_index, group in enumerate(keyword_groups):
            group_score = max(5, 110 - group_index * 9)
            for keyword in group:
                keyword_key = column_key(keyword)
                if normalized == keyword_key:
                    score = max(score, group_score + 12)
                elif normalized.startswith(keyword_key):
                    score = max(score, group_score + 6)
                elif keyword_key in normalized:
                    score = max(score, group_score)
        if score > 0:
            ranked.append((score, column))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [column for _, column in ranked[:limit]]


def detect_transaction_column(dataframe: pd.DataFrame) -> str | None:
    keyword_groups = [
        ["order_id", "orderid"],
        ["order_number", "order_no", "orderno", "ordernumber"],
        ["invoice_id", "invoiceid", "invoice"],
        ["bill_no", "billno", "bill_number"],
        ["transaction_id", "transactionid", "transaction", "txn"],
        ["receipt", "basket"],
    ]
    return _rank_column(list(dataframe.columns), keyword_groups)


def detect_item_column(dataframe: pd.DataFrame, exclude: set[str] | None = None) -> str | None:
    keyword_groups = [
        ["product_name", "productname", "item_name", "itemname"],
        ["product", "item", "description", "product_description", "item_description"],
        ["sku"],
        ["product_code", "productcode", "item_code", "itemcode"],
        ["barcode", "bar_code", "ean", "upc"],
    ]
    return _rank_column(list(dataframe.columns), keyword_groups, exclude=exclude)


def detect_code_columns(dataframe: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    keyword_groups = [
        ["barcode", "bar_code", "ean", "upc"],
        ["sku"],
        ["product_code", "productcode", "item_code", "itemcode"],
        ["model", "part_number", "code"],
    ]
    return _rank_columns(dataframe, keyword_groups, limit=4, exclude=exclude)


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def detect_numeric_columns(dataframe: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    keyword_groups = [
        ["price", "unit_price", "mrp", "cost", "amount"],
        ["quantity", "qty", "stock", "volume"],
        ["weight", "size", "discount", "total"],
    ]
    ranked = _rank_columns(dataframe, keyword_groups, limit=5, exclude=exclude)
    detected = list(ranked)
    for column in dataframe.columns:
        if column in exclude or column in detected:
            continue
        converted = _coerce_numeric_series(dataframe[column])
        if float(converted.notna().mean()) >= 0.7:
            detected.append(column)
    return detected[:5]


def detect_categorical_columns(dataframe: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    keyword_groups = [
        ["category", "subcategory", "segment", "type", "family"],
        ["brand", "manufacturer", "vendor", "supplier"],
        ["region", "zone", "customer_type", "channel"],
    ]
    ranked = _rank_columns(dataframe, keyword_groups, limit=5, exclude=exclude)
    detected = list(ranked)
    for column in dataframe.columns:
        if column in exclude or column in detected:
            continue
        if dataframe[column].dtype != object:
            continue
        cardinality = int(dataframe[column].nunique(dropna=True))
        if 1 < cardinality <= max(15, int(len(dataframe) * 0.35)):
            detected.append(column)
    return detected[:5]


def detect_date_column(dataframe: pd.DataFrame, exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    keyword_groups = [
        ["order_date", "transaction_date", "invoice_date", "created_at", "created_date"],
        ["date", "bill_date", "posting_date"],
    ]
    ranked = _rank_columns(dataframe, keyword_groups, limit=2, exclude=exclude)
    if ranked:
        return ranked[0]
    for column in dataframe.columns:
        if column in exclude:
            continue
        parsed = pd.to_datetime(dataframe[column], errors="coerce")
        if float(parsed.notna().mean()) >= 0.7:
            return column
    return None


def _safe_session_state_update(key: str, value: Any) -> None:
    try:
        st.session_state[key] = value
    except StreamlitAPIException:
        pass


def get_anomaly_field_state(dataframe: pd.DataFrame) -> dict[str, Any]:
    code_defaults = detect_code_columns(dataframe)
    auto_identifier = detect_item_column(dataframe) or detect_transaction_column(dataframe) or (code_defaults[0] if code_defaults else dataframe.columns[0])
    auto_numeric = detect_numeric_columns(dataframe, exclude={auto_identifier} if auto_identifier else None)
    auto_code = detect_code_columns(dataframe, exclude={auto_identifier} if auto_identifier else None)
    auto_categorical = detect_categorical_columns(
        dataframe,
        exclude=set([auto_identifier] + auto_code + auto_numeric) if auto_identifier else set(auto_code + auto_numeric),
    )
    auto_date = detect_date_column(
        dataframe,
        exclude=set([auto_identifier] + auto_code + auto_numeric + auto_categorical) if auto_identifier else set(auto_code + auto_numeric + auto_categorical),
    )

    def _valid_columns(columns: list[str]) -> list[str]:
        return [column for column in columns if column in dataframe.columns]

    identifier_column = st.session_state.get("anomaly_identifier_column", "")
    if identifier_column not in dataframe.columns:
        identifier_column = auto_identifier or dataframe.columns[0]

    code_columns = _valid_columns(st.session_state.get("anomaly_code_columns", []))
    if not code_columns:
        code_columns = auto_code[:3]

    numeric_columns = _valid_columns(st.session_state.get("anomaly_numeric_columns", []))
    if not numeric_columns:
        numeric_columns = auto_numeric[:4]

    categorical_columns = _valid_columns(st.session_state.get("anomaly_categorical_columns", []))
    if not categorical_columns:
        categorical_columns = auto_categorical[:4]

    date_column = clean_text(st.session_state.get("anomaly_date_column", ""))
    if date_column in {"(None)", "None"} or date_column not in dataframe.columns:
        date_column = auto_date or ""

    contamination = min(max(float(st.session_state.get("anomaly_contamination", 0.08)), 0.01), 0.35)
    row_limit = min(max(int(st.session_state.get("anomaly_row_limit", 15)), 5), 100)

    return {
        "anomaly_identifier_column": identifier_column,
        "anomaly_code_columns": code_columns,
        "anomaly_numeric_columns": numeric_columns,
        "anomaly_categorical_columns": categorical_columns,
        "anomaly_date_column": date_column,
        "anomaly_contamination": contamination,
        "anomaly_row_limit": row_limit,
        "anomaly_filter_text": clean_text(st.session_state.get("anomaly_filter_text", "")),
        "anomaly_date_range": st.session_state.get("anomaly_date_range", ()),
    }


def sync_default_anomaly_fields(dataframe: pd.DataFrame) -> None:
    for key, value in get_anomaly_field_state(dataframe).items():
        _safe_session_state_update(key, value)


def update_order_dataframe(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return get_loaded_order_dataframe()

    dataframe = load_order_dataframe(uploaded_file)
    st.session_state.order_dataframe = dataframe
    st.session_state.order_file_name = uploaded_file.name
    sync_default_anomaly_fields(dataframe)
    return dataframe


def render_anomaly_sidebar_controls(dataframe: pd.DataFrame) -> None:
    sync_default_anomaly_fields(dataframe)
    state = get_anomaly_field_state(dataframe)
    with st.sidebar.expander("Risk Detector Controls", expanded=False):
        st.selectbox(
            "Identifier column",
            options=list(dataframe.columns),
            index=list(dataframe.columns).index(state["anomaly_identifier_column"]),
            key="anomaly_identifier_column",
        )
        st.multiselect(
            "Code columns",
            options=list(dataframe.columns),
            default=state["anomaly_code_columns"],
            key="anomaly_code_columns",
        )
        st.multiselect(
            "Numeric columns",
            options=list(dataframe.columns),
            default=state["anomaly_numeric_columns"],
            key="anomaly_numeric_columns",
        )
        st.multiselect(
            "Categorical columns",
            options=list(dataframe.columns),
            default=state["anomaly_categorical_columns"],
            key="anomaly_categorical_columns",
        )
        date_options = ["(None)"] + list(dataframe.columns)
        selected_date = state["anomaly_date_column"] if state["anomaly_date_column"] in dataframe.columns else "(None)"
        st.selectbox(
            "Date column",
            options=date_options,
            index=date_options.index(selected_date),
            key="anomaly_date_column",
        )
        st.slider(
            "Anomaly sensitivity",
            min_value=0.01,
            max_value=0.25,
            value=state["anomaly_contamination"],
            step=0.01,
            key="anomaly_contamination",
        )
        st.number_input(
            "Rows to show",
            min_value=5,
            max_value=100,
            value=state["anomaly_row_limit"],
            step=5,
            key="anomaly_row_limit",
        )
        st.text_input(
            "Barcode / item filter",
            value=state["anomaly_filter_text"],
            key="anomaly_filter_text",
        )

        date_column = clean_text(st.session_state.get("anomaly_date_column", ""))
        if date_column and date_column in dataframe.columns:
            parsed_dates = pd.to_datetime(dataframe[date_column], errors="coerce")
            valid_dates = parsed_dates.dropna()
            if not valid_dates.empty:
                default_range = (valid_dates.min().date(), valid_dates.max().date())
                current_range = st.session_state.get("anomaly_date_range", ())
                if not current_range or len(current_range) != 2:
                    _safe_session_state_update("anomaly_date_range", default_range)
                st.date_input("Date range", key="anomaly_date_range")


def parse_numeric_value(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace(",", "").replace("₹", "").replace("$", "")
    try:
        return float(text)
    except Exception:
        return None


def normalize_series(values: pd.Series | np.ndarray) -> pd.Series:
    series = pd.Series(values, copy=False).astype(float)
    if series.empty:
        return series
    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(minimum, maximum):
        return pd.Series(np.clip(series.to_numpy(), 0.0, 1.0), index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum)


def _make_feature_name(prefix: str, column: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in column.lower())
    return f"{prefix}{safe}"


def _infer_special_numeric_columns(numeric_columns: list[str]) -> dict[str, str]:
    def _pick(keywords: list[str]) -> str:
        for column in numeric_columns:
            normalized = column_key(column)
            if any(keyword in normalized for keyword in keywords):
                return column
        return ""

    return {
        "quantity": _pick(["quantity", "qty", "units", "pieces", "count"]),
        "price": _pick(["unitprice", "price", "mrp", "cost", "rate"]),
        "amount": _pick(["total", "amount", "net", "gross", "value"]),
    }


def prepare_anomaly_features(
    dataframe: pd.DataFrame,
    identifier_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    code_columns: list[str],
    date_column: str,
    filter_text: str,
    date_range: tuple[date, date] | tuple[Any, ...],
) -> dict[str, Any]:
    working = dataframe.copy().reset_index().rename(columns={"index": "source_index"})
    working["identifier_value"] = working[identifier_column].map(clean_text)
    working["row_label"] = [
        f"{identifier or 'Row'} [{int(source_index)}]"
        for identifier, source_index in zip(working["identifier_value"], working["source_index"])
    ]

    searchable_columns = [column for column in [identifier_column] + code_columns + categorical_columns if column in working.columns]
    if filter_text and searchable_columns:
        filter_key = filter_text.casefold()
        search_blob = working[searchable_columns].fillna("").astype(str).agg(" ".join, axis=1).str.casefold()
        working = working[search_blob.str.contains(filter_key, na=False)].copy()

    parsed_dates = pd.Series(pd.NaT, index=working.index, dtype="datetime64[ns]")
    if date_column and date_column in working.columns:
        parsed_dates = pd.to_datetime(working[date_column], errors="coerce")
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date = pd.Timestamp(date_range[0])
            end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask = parsed_dates.between(start_date, end_date, inclusive="both")
            working = working[mask.fillna(False)].copy()
            parsed_dates = parsed_dates.loc[working.index]
    working["parsed_date"] = parsed_dates

    numeric_feature_map: dict[str, str] = {}
    rarity_feature_map: dict[str, str] = {}
    feature_parts: list[pd.DataFrame] = []

    for column in numeric_columns:
        if column not in working.columns:
            continue
        feature_name = _make_feature_name("num__", column)
        numeric_feature_map[column] = feature_name
        working[feature_name] = _coerce_numeric_series(working[column])
        fill_value = float(working[feature_name].median()) if working[feature_name].notna().any() else 0.0
        feature_parts.append(pd.DataFrame({feature_name: working[feature_name].fillna(fill_value)}, index=working.index))

    encoded_source_columns = [column for column in categorical_columns + code_columns if column in working.columns]
    for column in encoded_source_columns:
        cleaned = working[column].map(clean_text).replace("", "__missing__")
        counts = cleaned.value_counts(normalize=True)
        rarity_name = _make_feature_name("rarity__", column)
        rarity_feature_map[column] = rarity_name
        working[rarity_name] = cleaned.map(lambda value: 1.0 - float(counts.get(value, 0.0)))
        feature_parts.append(pd.DataFrame({rarity_name: working[rarity_name]}, index=working.index))

        top_values = cleaned.value_counts().head(12).index.tolist()
        compact = cleaned.where(cleaned.isin(top_values), "__other__")
        feature_parts.append(pd.get_dummies(compact, prefix=_make_feature_name("cat__", column)).astype(float))

    special_numeric = _infer_special_numeric_columns(numeric_columns)
    quantity_column = special_numeric["quantity"]
    price_column = special_numeric["price"]
    amount_column = special_numeric["amount"]

    if quantity_column and price_column and amount_column and quantity_column in numeric_feature_map and price_column in numeric_feature_map and amount_column in numeric_feature_map:
        quantity_values = working[numeric_feature_map[quantity_column]]
        price_values = working[numeric_feature_map[price_column]]
        amount_values = working[numeric_feature_map[amount_column]]
        expected_amount = quantity_values * price_values
        scale = np.maximum(np.maximum(amount_values.abs(), expected_amount.abs()), 1.0)
        working["amount_mismatch_score"] = ((amount_values - expected_amount).abs() / scale).fillna(0.0)
        feature_parts.append(pd.DataFrame({"amount_mismatch_score": working["amount_mismatch_score"]}, index=working.index))
    else:
        working["amount_mismatch_score"] = 0.0

    baseline_group_column = next((column for column in code_columns if column in working.columns), identifier_column)
    group_keys = working[baseline_group_column].map(clean_text).replace("", "__missing__")
    working["group_key"] = group_keys

    if price_column and price_column in numeric_feature_map:
        price_feature = numeric_feature_map[price_column]
        price_medians = working.groupby("group_key")[price_feature].transform("median")
        scale = np.maximum(np.maximum(working[price_feature].abs(), price_medians.abs()), 1.0)
        working["price_group_deviation"] = ((working[price_feature] - price_medians).abs() / scale).fillna(0.0)
        feature_parts.append(pd.DataFrame({"price_group_deviation": working["price_group_deviation"]}, index=working.index))
    else:
        working["price_group_deviation"] = 0.0

    if quantity_column and quantity_column in numeric_feature_map:
        quantity_feature = numeric_feature_map[quantity_column]
        quantity_medians = working.groupby("group_key")[quantity_feature].transform("median")
        scale = np.maximum(np.maximum(working[quantity_feature].abs(), quantity_medians.abs()), 1.0)
        working["quantity_group_deviation"] = ((working[quantity_feature] - quantity_medians).abs() / scale).fillna(0.0)
        feature_parts.append(pd.DataFrame({"quantity_group_deviation": working["quantity_group_deviation"]}, index=working.index))
    else:
        working["quantity_group_deviation"] = 0.0

    if working["parsed_date"].notna().any():
        feature_parts.append(
            pd.DataFrame(
                {
                    "date_dayofweek": working["parsed_date"].dt.dayofweek.fillna(-1).astype(float),
                    "date_month": working["parsed_date"].dt.month.fillna(0).astype(float),
                    "date_day": working["parsed_date"].dt.day.fillna(0).astype(float),
                },
                index=working.index,
            )
        )

    feature_frame = pd.concat(feature_parts, axis=1) if feature_parts else pd.DataFrame(index=working.index)
    feature_frame = feature_frame.fillna(0.0).astype(float)
    if feature_frame.empty:
        feature_frame["baseline_signal"] = 0.0

    return {
        "working_df": working.reset_index(drop=True),
        "feature_df": feature_frame.reset_index(drop=True),
        "numeric_feature_map": numeric_feature_map,
        "rarity_feature_map": rarity_feature_map,
        "identifier_column": identifier_column,
    }


def run_isolation_forest(feature_df: pd.DataFrame, contamination: float) -> dict[str, pd.Series]:
    row_count = len(feature_df)
    baseline = pd.Series(np.zeros(row_count, dtype=float))
    if feature_df.empty or row_count < 8 or not SKLEARN_AVAILABLE:
        return {"score": baseline, "flag": pd.Series(np.zeros(row_count, dtype=int))}

    contamination = min(max(contamination, 0.01), 0.35)
    if contamination * row_count < 1:
        contamination = min(0.35, (1.0 / max(row_count, 1)) + 0.01)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df.fillna(0.0))
    model = IsolationForest(n_estimators=250, contamination=contamination, random_state=42)
    prediction = model.fit_predict(scaled)
    score = pd.Series(-model.score_samples(scaled)).astype(float)
    return {"score": normalize_series(score), "flag": pd.Series((prediction == -1).astype(int))}


def compute_statistical_outliers(working_df: pd.DataFrame, numeric_feature_map: dict[str, str]) -> dict[str, Any]:
    zscore_frame = pd.DataFrame(index=working_df.index)
    iqr_frame = pd.DataFrame(index=working_df.index)
    stats_reference: dict[str, dict[str, float]] = {}

    for column, feature_name in numeric_feature_map.items():
        values = working_df[feature_name]
        if not values.notna().any():
            continue

        mean_value = float(values.mean())
        median_value = float(values.median())
        std_value = float(values.std(ddof=0))
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
        else:
            lower_bound = mean_value - 3 * std_value
            upper_bound = mean_value + 3 * std_value

        if std_value > 0:
            z_values = ((values - mean_value).abs() / std_value).fillna(0.0)
        else:
            z_values = pd.Series(np.zeros(len(values)), index=values.index, dtype=float)

        zscore_frame[column] = np.clip(z_values / 4.0, 0.0, 1.0)
        iqr_frame[column] = ((values < lower_bound) | (values > upper_bound)).astype(float).fillna(0.0)
        stats_reference[column] = {
            "median": median_value,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    if zscore_frame.empty:
        zscore_frame["baseline"] = 0.0
        iqr_frame["baseline"] = 0.0

    return {"zscore_frame": zscore_frame, "iqr_frame": iqr_frame, "stats_reference": stats_reference}


def explain_row_anomaly(row: pd.Series, analysis_context: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    comparison_rows: list[dict[str, Any]] = []

    for column, feature_name in analysis_context["numeric_feature_map"].items():
        value = row.get(feature_name)
        if pd.isna(value):
            continue
        stats = analysis_context["stats_reference"].get(column)
        if not stats:
            continue
        comparison_rows.append(
            {
                "field": column,
                "value": format_float(value),
                "median": format_float(stats["median"]),
                "expected_range": f"{format_float(stats['lower_bound'])} to {format_float(stats['upper_bound'])}",
            }
        )
        if value > stats["upper_bound"]:
            reasons.append(f"{column} is unusually high for this dataset.")
        elif value < stats["lower_bound"]:
            reasons.append(f"{column} is unusually low for this dataset.")

    for column, feature_name in analysis_context["rarity_feature_map"].items():
        rarity_score = float(row.get(feature_name, 0.0))
        field_value = clean_text(row.get(column))
        if field_value and rarity_score >= 0.85:
            reasons.append(f"Rare {column}: {field_value}.")

    if float(row.get("amount_mismatch_score", 0.0)) >= 0.15:
        reasons.append("Amount is inconsistent with quantity multiplied by unit price.")
    if float(row.get("price_group_deviation", 0.0)) >= 0.30:
        reasons.append("Price is unusual compared with similar items or codes.")
    if float(row.get("quantity_group_deviation", 0.0)) >= 0.30:
        reasons.append("Quantity is unusual compared with similar items or codes.")

    status = str(row["risk_status"])
    if not reasons:
        reasons.append("This row looks consistent with the overall dataset." if status == "Normal" else "The overall row pattern is unusual compared with the rest of the dataset.")

    recommendation = "Likely normal transaction"
    if status == "High Risk":
        recommendation = "Needs manual review"
    elif status == "Unusual":
        recommendation = "Review recommended"

    return {
        "major_reason": reasons[0],
        "all_reasons": reasons[:5],
        "comparison_rows": comparison_rows,
        "recommendation": recommendation,
    }


def combine_anomaly_signals(prepared: dict[str, Any], contamination: float) -> pd.DataFrame:
    working_df = prepared["working_df"].copy()
    feature_df = prepared["feature_df"]
    numeric_feature_map = prepared["numeric_feature_map"]
    rarity_feature_map = prepared["rarity_feature_map"]

    isolation_result = run_isolation_forest(feature_df, contamination)
    statistical_result = compute_statistical_outliers(working_df, numeric_feature_map)

    working_df["isolation_score"] = isolation_result["score"].to_numpy()
    working_df["isolation_flag"] = isolation_result["flag"].to_numpy()
    working_df["zscore_signal"] = statistical_result["zscore_frame"].mean(axis=1).to_numpy()
    working_df["iqr_signal"] = statistical_result["iqr_frame"].mean(axis=1).to_numpy()
    rarity_columns = list(rarity_feature_map.values())
    working_df["rarity_signal"] = working_df[rarity_columns].mean(axis=1).fillna(0.0) if rarity_columns else 0.0
    working_df["rule_signal"] = np.clip(
        (working_df["amount_mismatch_score"] + working_df["price_group_deviation"] + working_df["quantity_group_deviation"]) / 3.0,
        0.0,
        1.0,
    )

    components: list[tuple[str, float]] = []
    if float(working_df["isolation_score"].max()) > 0:
        components.append(("isolation_score", 0.45))
    if float(working_df["zscore_signal"].max()) > 0:
        components.append(("zscore_signal", 0.20))
    if float(working_df["iqr_signal"].max()) > 0:
        components.append(("iqr_signal", 0.10))
    if float(working_df["rarity_signal"].max()) > 0:
        components.append(("rarity_signal", 0.15))
    if float(working_df["rule_signal"].max()) > 0:
        components.append(("rule_signal", 0.10))
    if not components:
        components = [("zscore_signal", 1.0)]

    total_weight = sum(weight for _, weight in components)
    weighted_sum = sum(working_df[column].fillna(0.0) * weight for column, weight in components)
    working_df["anomaly_score"] = normalize_series(weighted_sum / max(total_weight, 1e-9))

    if len(working_df) >= 10:
        high_threshold = float(working_df["anomaly_score"].quantile(0.90))
        unusual_threshold = float(working_df["anomaly_score"].quantile(0.70))
    else:
        high_threshold = 0.75
        unusual_threshold = 0.45

    def _label(score: float) -> str:
        if score >= max(high_threshold, 0.70):
            return "High Risk"
        if score >= max(unusual_threshold, 0.35):
            return "Unusual"
        return "Normal"

    working_df["risk_status"] = working_df["anomaly_score"].map(_label)
    explanations = working_df.apply(
        lambda row: explain_row_anomaly(
            row,
            {
                "numeric_feature_map": numeric_feature_map,
                "rarity_feature_map": rarity_feature_map,
                "stats_reference": statistical_result["stats_reference"],
            },
        ),
        axis=1,
    )
    working_df["major_reason"] = explanations.map(lambda item: item["major_reason"])
    working_df["reason_list"] = explanations.map(lambda item: item["all_reasons"])
    working_df["comparison_rows"] = explanations.map(lambda item: item["comparison_rows"])
    working_df["review_recommendation"] = explanations.map(lambda item: item["recommendation"])
    return working_df.sort_values(["anomaly_score", "isolation_flag"], ascending=[False, False]).reset_index(drop=True)


def render_risk_summary_cards(results: pd.DataFrame, identifier_column: str) -> None:
    st.markdown("#### Risk Summary Cards")
    total_rows = int(len(results))
    high_risk_rows = int((results["risk_status"] == "High Risk").sum())
    unusual_rows = int((results["risk_status"] == "Unusual").sum())
    normal_rows = int((results["risk_status"] == "Normal").sum())
    most_abnormal = clean_text(results.iloc[0].get(identifier_column) if not results.empty else "") or "n/a"

    card_columns = st.columns(5)
    card_columns[0].markdown(render_card("Rows Analyzed", str(total_rows), "Filtered rows included in anomaly scoring."), unsafe_allow_html=True)
    card_columns[1].markdown(render_card("Anomalies Found", str(high_risk_rows + unusual_rows), "Rows labelled unusual or high risk."), unsafe_allow_html=True)
    card_columns[2].markdown(render_card("High Risk Rows", str(high_risk_rows), "Rows needing the fastest manual review."), unsafe_allow_html=True)
    card_columns[3].markdown(render_card("Normal Rows", str(normal_rows), "Rows that fit the current dataset pattern."), unsafe_allow_html=True)
    card_columns[4].markdown(render_card("Most Abnormal", most_abnormal, "Top row by combined anomaly score."), unsafe_allow_html=True)


def render_matched_barcode_risk(results: pd.DataFrame, identifier_column: str) -> None:
    st.markdown("#### Matched Barcode Risk Card")
    decoded_code = clean_text(st.session_state.get("decoded_code", ""))
    lookup_context = st.session_state.get("last_lookup_context", {})
    matched_indices = lookup_context.get("matched_row_indices", []) if isinstance(lookup_context, dict) else []

    if not decoded_code:
        st.info("Decode a barcode to see the risk posture of the matched order row.")
        return

    matched_rows = results[results["source_index"].isin(matched_indices)] if matched_indices else pd.DataFrame()
    if matched_rows.empty:
        st.info("A barcode value is available, but no matching order row was found in the loaded sheet for row-level risk scoring.")
        return

    row = matched_rows.sort_values("anomaly_score", ascending=False).iloc[0]
    risk_columns = st.columns(4)
    risk_columns[0].markdown(render_card("Matched Item", clean_text(row.get(identifier_column)) or row["row_label"], "Highest-risk row among the current barcode matches."), unsafe_allow_html=True)
    risk_columns[1].markdown(render_card("Anomaly Score", format_score(float(row["anomaly_score"])), "Combined Isolation Forest and statistical outlier score."), unsafe_allow_html=True)
    risk_columns[2].markdown(render_card("Anomaly Status", str(row["risk_status"]), str(row["major_reason"])), unsafe_allow_html=True)
    risk_columns[3].markdown(render_card("Review Recommendation", str(row["review_recommendation"]), "Use this to prioritize manual review."), unsafe_allow_html=True)
    st.markdown(
        f"""<div class="why-box"><strong>Why it was flagged:</strong><br>{"<br>".join(str(reason) for reason in row["reason_list"])}</div>""",
        unsafe_allow_html=True,
    )


def render_top_anomaly_table(results: pd.DataFrame, identifier_column: str, numeric_columns: list[str], row_limit: int) -> None:
    st.markdown("#### Top Anomalies Table")
    if results.empty:
        st.info("No rows are available after the current filters.")
        return

    preview_columns = [identifier_column, "anomaly_score", "risk_status", "major_reason"]
    preview_columns.extend(column for column in numeric_columns[:3] if column in results.columns and column not in preview_columns)
    anomaly_table = results.head(row_limit).copy()
    anomaly_table["anomaly_score"] = anomaly_table["anomaly_score"].map(format_score)
    st.dataframe(
        anomaly_table[preview_columns].rename(columns={identifier_column: "identifier", "anomaly_score": "score", "risk_status": "status", "major_reason": "reason"}),
        use_container_width=True,
        hide_index=True,
    )


def render_reason_breakdown(results: pd.DataFrame, identifier_column: str) -> None:
    st.markdown("#### Reason Breakdown")
    if results.empty:
        st.info("Reason details will appear once rows are analyzed.")
        return

    lookup_context = st.session_state.get("last_lookup_context", {})
    matched_indices = lookup_context.get("matched_row_indices", []) if isinstance(lookup_context, dict) else []
    default_row = results.iloc[0]["row_label"]
    if matched_indices:
        matched_rows = results[results["source_index"].isin(matched_indices)]
        if not matched_rows.empty:
            default_row = matched_rows.sort_values("anomaly_score", ascending=False).iloc[0]["row_label"]

    options = results["row_label"].tolist()
    default_index = options.index(default_row) if default_row in options else 0
    selected_row_label = st.selectbox("Select a row for review", options=options, index=default_index)
    row = results[results["row_label"] == selected_row_label].iloc[0]

    summary_columns = st.columns(3)
    summary_columns[0].metric("Identifier", clean_text(row.get(identifier_column)) or row["row_label"])
    summary_columns[1].metric("Status", str(row["risk_status"]))
    summary_columns[2].metric("Anomaly Score", format_score(float(row["anomaly_score"])))

    if row["comparison_rows"]:
        st.dataframe(pd.DataFrame(row["comparison_rows"]), use_container_width=True, hide_index=True)
    else:
        st.info("No numeric comparison ranges were available for this row.")

    st.markdown(
        f"""<div class="why-box"><strong>Review recommendation:</strong> {row["review_recommendation"]}<br><br>{"<br>".join(str(reason) for reason in row["reason_list"])}</div>""",
        unsafe_allow_html=True,
    )


def render_anomaly_visual(results: pd.DataFrame, numeric_columns: list[str]) -> None:
    st.markdown("#### Risk Visual")
    if results.empty:
        st.info("A risk chart will appear after anomaly scoring is complete.")
        return
    if not ALTAIR_AVAILABLE:
        st.info("Altair is not available in this environment, so the anomaly chart cannot be displayed.")
        return

    primary_numeric = next((column for column in numeric_columns if column in results.columns), "")
    if primary_numeric:
        chart = (
            alt.Chart(results.head(200))
            .mark_circle(size=90, opacity=0.8)
            .encode(
                x=alt.X(f"{primary_numeric}:Q", title=primary_numeric),
                y=alt.Y("anomaly_score:Q", title="Anomaly Score"),
                color=alt.Color("risk_status:N", scale=alt.Scale(domain=["Normal", "Unusual", "High Risk"], range=["#0f766e", "#d97706", "#b91c1c"])),
                tooltip=[
                    alt.Tooltip("row_label:N", title="Row"),
                    alt.Tooltip("risk_status:N", title="Status"),
                    alt.Tooltip("anomaly_score:Q", title="Score", format=".3f"),
                    alt.Tooltip(f"{primary_numeric}:Q", title=primary_numeric, format=".2f"),
                    alt.Tooltip("major_reason:N", title="Reason"),
                ],
            )
            .properties(height=360)
        )
    else:
        chart = (
            alt.Chart(results.head(20))
            .mark_bar(color="#0f766e")
            .encode(
                x=alt.X("anomaly_score:Q", title="Anomaly Score"),
                y=alt.Y("row_label:N", title="Row", sort="-x"),
                tooltip=[
                    alt.Tooltip("row_label:N", title="Row"),
                    alt.Tooltip("risk_status:N", title="Status"),
                    alt.Tooltip("anomaly_score:Q", title="Score", format=".3f"),
                    alt.Tooltip("major_reason:N", title="Reason"),
                ],
            )
            .properties(height=360)
        )
    st.altair_chart(chart, use_container_width=True)


def render_review_recommendation_box(results: pd.DataFrame) -> None:
    st.markdown("#### Review Recommendation Box")
    if results.empty:
        st.info("Review guidance will appear once rows are analyzed.")
        return

    high_risk = int((results["risk_status"] == "High Risk").sum())
    unusual = int((results["risk_status"] == "Unusual").sum())
    if high_risk > 0:
        message = f"Possible review needed: {high_risk} high-risk row(s) stand out and should be checked first."
    elif unusual > 0:
        message = f"Likely normal overall, but {unusual} row(s) look unusual and deserve a quick review."
    else:
        message = "Likely normal transaction set: no rows crossed the current unusual-risk thresholds."

    st.markdown(
        f"""<div class="why-box">{message}<br><br>Highest scoring row: <strong>{results.iloc[0]["row_label"]}</strong><br>Primary reason: {results.iloc[0]["major_reason"]}</div>""",
        unsafe_allow_html=True,
    )


def render_sidebar(settings, pages: list[PowerBIPage], current_page: PowerBIPage | None, dataframe: pd.DataFrame | None) -> None:
    st.sidebar.markdown("## Report Navigation")
    st.sidebar.caption("Streamlit buttons control the embedded Power BI report pages.")

    for page in pages:
        if st.sidebar.button(
            page.button_label,
            key=f"page-button-{page.name}",
            type="primary" if current_page and page.name == current_page.name else "secondary",
            use_container_width=True,
        ):
            st.session_state.selected_page_name = page.name

    st.sidebar.markdown("---")
    if current_page:
        st.sidebar.markdown(f"**Active page:** {current_page.button_label}")
        st.sidebar.caption(f"Internal Power BI page name: `{current_page.name}`")
        st.sidebar.link_button(
            "Open Current Page in Power BI",
            build_page_embed_url(settings, current_page.name),
            use_container_width=True,
        )

    embed_mode_label = "JavaScript SDK" if settings.can_use_js_sdk else "Iframe fallback"
    data_status = "Order data loaded" if dataframe is not None else "No order data yet"
    data_class = "status-chip" if dataframe is not None else "status-chip warn-chip"
    st.sidebar.markdown(
        f"""
        <div class="info-card">
            <div><span class="status-chip">{embed_mode_label}</span></div>
            <div style="margin-top:0.6rem;"><span class="{data_class}">{data_status}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    order_file_name = st.session_state.get("order_file_name", "")
    if order_file_name:
        st.sidebar.caption(f"Current order file: `{order_file_name}`")

    if dataframe is not None:
        render_anomaly_sidebar_controls(dataframe)


def render_powerbi_panel(settings, current_page: PowerBIPage | None, height: int) -> None:
    if current_page is None:
        st.error("No Power BI report pages were discovered. Keep the PBIX file beside the app or provide a page catalog.")
        return

    if not settings.can_use_js_sdk:
        st.warning(
            "Power BI JavaScript SDK mode is not fully configured, so the app is using iframe mode. "
            "The dashboard still remains embedded and page buttons still work."
        )

    components.html(
        build_powerbi_embed_html(
            settings=settings,
            page_name=current_page.name,
            height=height,
            bridge_url=None,
            enable_visual_sync=False,
            page_width=current_page.width,
            page_height=current_page.height,
        ),
        height=height + 75,
    )


def record_barcode_history(decoded_code: str, source: str) -> None:
    if not decoded_code:
        return
    history = st.session_state.barcode_history
    last_code = history[0]["decoded_code"] if history else None
    if decoded_code == last_code:
        return
    history.insert(
        0,
        {
            "decoded_code": decoded_code,
            "source": source,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    st.session_state.barcode_history = history[:25]
    st.session_state.barcode_lookup_source = source


def get_active_barcode_scan_result(decoded_code: str | None = None) -> BarcodeScanResult | None:
    retry_state = st.session_state.get("barcode_retry_result")
    retry_result = retry_state.get("result") if isinstance(retry_state, dict) else None
    candidates = [retry_result, st.session_state.get("last_scan_result")]
    normalized_decoded = normalize_code(decoded_code or "")

    for candidate in candidates:
        if not candidate:
            continue
        primary = candidate.primary_decoded
        if not normalized_decoded:
            return candidate
        if primary and normalize_code(primary.value) == normalized_decoded:
            return candidate
    return None


def render_barcode_decoder_status() -> None:
    status = get_barcode_decoder_status()
    rows = [{"Decoder": name, "Status": "Ready" if available else "Unavailable"} for name, available in status.items()]
    with st.expander("Decoder Runtime", expanded=False):
        st.caption("Decode order: zxing-cpp -> OpenCV barcode -> pyzbar -> QR fallback.")
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_barcode_result_panel(
    scan_result: BarcodeScanResult | None,
    matched_rows: pd.DataFrame | None = None,
    matched_columns: list[str] | None = None,
    heading: str = "Decode Result",
) -> None:
    if scan_result is None:
        st.info("Run a scan first to see the decoder details.")
        return

    primary = scan_result.primary_decoded
    st.markdown(f"#### {heading}")
    st.code(primary.value if primary else "Not decoded")

    detector_confidence = "n/a"
    if primary and primary.confidence is not None:
        detector_confidence = f"{primary.confidence:.2f}"
    elif scan_result.regions:
        detector_confidence = f"{scan_result.regions[0].confidence:.2f}"

    match_status = "Pending"
    if matched_rows is not None:
        match_status = "Yes" if not matched_rows.empty else "No"

    metric_columns = st.columns(6)
    metric_columns[0].metric("Format", primary.symbology if primary else "Unavailable")
    metric_columns[1].metric("Decoder", primary.decoder if primary and primary.decoder else "Unavailable")
    metric_columns[2].metric("Variant", primary.preprocessing if primary and primary.preprocessing else "Unavailable")
    metric_columns[3].metric("Rotation", f"{primary.rotation} deg" if primary else "-")
    metric_columns[4].metric("Detector Conf.", detector_confidence)
    metric_columns[5].metric("Order Match", match_status)

    detail_lines: list[str] = []
    if primary and primary.source_region_id:
        detail_lines.append(f"Crop used: `{primary.source_region_id}`")
    elif scan_result.regions:
        detail_lines.append(f"Detected crop: `{scan_result.regions[0].crop_id}`")
    if scan_result.source_name:
        detail_lines.append(f"Source image: `{scan_result.source_name}`")
    if matched_rows is not None and matched_columns:
        detail_lines.append(f"Matched column(s): {', '.join(matched_columns)}")
    if detail_lines:
        st.caption(" | ".join(detail_lines))

    if primary:
        st.success("The barcode value is backed by the multi-decoder pipeline.")
    else:
        st.warning(scan_result.failure_message or "No barcode value was decoded from the selected image.")
        if scan_result.suggestions:
            st.caption("Suggestions: " + " | ".join(scan_result.suggestions))

    with st.expander("Decode Attempt Details", expanded=False):
        if scan_result.decoder_trace:
            st.code("\n".join(scan_result.decoder_trace[:36]))
        else:
            st.caption("No decoder trace is available for this scan.")


def render_dashboard_tab(settings, current_page: PowerBIPage | None) -> None:
    st.subheader("Dashboard Viewer")
    st.caption("The Power BI report stays embedded here while Streamlit handles navigation.")
    render_powerbi_panel(settings=settings, current_page=current_page, height=690)


def render_uploaded_barcode_flow() -> None:
    uploaded_image = st.file_uploader(
        "Upload a screenshot or photo containing a barcode or QR code",
        type=["png", "jpg", "jpeg", "webp"],
        key="barcode-upload",
    )
    if not uploaded_image:
        return

    source_name = uploaded_image.name
    image = Image.open(uploaded_image).convert("RGB")
    preview_columns = st.columns(2)
    with preview_columns[0]:
        st.markdown("#### Uploaded Image")
        st.image(image, use_container_width=True)

    if st.button("Scan Uploaded Image", key="scan-uploaded-image", type="primary"):
        with st.spinner("Detecting barcode regions and decoding values..."):
            scan_result = scan_barcode_image(image, source_name=source_name)
            st.session_state.last_scan_result = scan_result
            st.session_state.barcode_retry_result = None
            primary = scan_result.primary_decoded
            st.session_state.decoded_code = primary.value if primary else ""
            if primary:
                record_barcode_history(primary.value, "uploaded image")

    scan_result = st.session_state.get("last_scan_result")
    if scan_result is None or scan_result.source_name != source_name:
        return

    with preview_columns[1]:
        st.markdown("#### Detection Overlay")
        st.image(scan_result.annotated_image, use_container_width=True)

    render_barcode_result_panel(scan_result)

    if scan_result.regions:
        crop_labels = [f"Crop {index + 1} - detector {region.confidence:.2f}" for index, region in enumerate(scan_result.regions)]
        selected_label = st.selectbox("Detected crop to inspect", crop_labels, key="barcode-crop-picker")
        selected_index = crop_labels.index(selected_label)
        selected_region = scan_result.regions[selected_index]
        crop_columns = st.columns([1, 1.2])
        with crop_columns[0]:
            st.image(selected_region.crop, caption="Padded crop", use_container_width=True)
        with crop_columns[1]:
            st.caption("Retry runs the full preprocessing and decoder cascade on the padded crop.")
            if st.button("Retry Decode on Selected Crop", key="retry-decode-crop"):
                with st.spinner("Retrying decode on the selected crop..."):
                    retry_result = retry_decode_crop(
                        selected_region.crop,
                        confidence=selected_region.confidence,
                        source_region_id=selected_region.crop_id,
                        source_name=f"{source_name}:{selected_region.crop_id}",
                    )
                    st.session_state.barcode_retry_result = {"crop_id": selected_region.crop_id, "result": retry_result}
                    st.session_state.barcode_lookup_source = "crop retry"
                    primary = retry_result.primary_decoded
                    if primary:
                        st.session_state.decoded_code = primary.value
                        record_barcode_history(primary.value, "crop retry")

            retry_state = st.session_state.get("barcode_retry_result")
            if isinstance(retry_state, dict) and retry_state.get("crop_id") == selected_region.crop_id:
                retry_result = retry_state.get("result")
                if retry_result:
                    st.image(retry_result.annotated_image, caption="Retry crop overlay", use_container_width=True)
                    render_barcode_result_panel(retry_result, heading="Retry Decode Result")


def render_live_camera_flow() -> None:
    st.caption("The live preview is mirrored horizontally so it feels natural like a selfie camera.")
    if WEBRTC_AVAILABLE:
        webrtc_context = webrtc_streamer(
            key="barcode-live-stream",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=LiveBarcodeProcessor,
            async_processing=True,
        )

        if webrtc_context.video_processor:
            latest_result, latest_code = webrtc_context.video_processor.snapshot()
            if latest_result:
                st.session_state.last_scan_result = latest_result
                st.session_state.barcode_retry_result = None
            if latest_code:
                st.session_state.decoded_code = latest_code
                record_barcode_history(latest_code, "live camera")
            else:
                st.info("Point the barcode at the camera and hold it steady for a moment.")

            if latest_result and latest_result.regions:
                st.image(latest_result.annotated_image, caption="Mirrored live detection preview", use_container_width=True)
                render_barcode_result_panel(latest_result)
        return

    st.info("`streamlit-webrtc` is not available in this environment, so the app is falling back to `st.camera_input`.")
    camera_image = st.camera_input("Take a camera snapshot for barcode scanning", key="barcode-camera-fallback")
    if camera_image:
        source_name = getattr(camera_image, "name", "camera-snapshot")
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Camera snapshot", use_container_width=True)
        if st.button("Scan Camera Snapshot", key="scan-camera-snapshot", type="primary"):
            scan_result = scan_barcode_image(image, source_name=source_name)
            st.session_state.last_scan_result = scan_result
            st.session_state.barcode_retry_result = None
            primary = scan_result.primary_decoded
            st.session_state.decoded_code = primary.value if primary else ""
            if primary:
                record_barcode_history(primary.value, "camera snapshot")

        scan_result = st.session_state.get("last_scan_result")
        if scan_result and scan_result.source_name == source_name:
            st.image(scan_result.annotated_image, caption="Detection overlay", use_container_width=True)
            render_barcode_result_panel(scan_result)


def render_lookup_results() -> None:
    st.markdown("#### Order Lookup")
    order_file = st.file_uploader(
        "Upload an Excel or CSV file containing order data",
        type=["xlsx", "csv"],
        key="order-lookup-file",
    )

    dataframe = update_order_dataframe(order_file)
    decoded_code = st.text_input(
        "Decoded barcode or order code",
        value=st.session_state.get("decoded_code", ""),
        key="decoded-code-input",
    )
    st.session_state.decoded_code = decoded_code

    if dataframe is None:
        st.info("Upload an order file to match the decoded value against likely columns such as barcode, SKU, product code, item code, or order ID.")
        return

    st.caption(f"Loaded `{st.session_state.order_file_name}` with {len(dataframe)} rows and {len(dataframe.columns)} columns.")
    suggested_columns = guess_lookup_columns(dataframe)
    selected_columns = st.multiselect(
        "Columns to search",
        options=list(dataframe.columns),
        default=suggested_columns[: min(5, len(suggested_columns))],
        key="lookup-columns",
    )
    item_column = detect_item_column(dataframe, exclude={detect_transaction_column(dataframe)} if detect_transaction_column(dataframe) else None)

    if not decoded_code:
        st.info("Scan or type a code above to run the order lookup.")
        return

    matched_rows, searched_columns, matched_columns = lookup_orders_detailed(dataframe, decoded_code, selected_columns)
    normalized_target = normalize_code(decoded_code)
    exact_match = False
    for column in searched_columns:
        if column not in matched_rows.columns:
            continue
        exact_match = bool(matched_rows[column].map(normalize_code).eq(normalized_target).any()) or exact_match

    matched_items: list[str] = []
    matched_identifier = ""
    if item_column and item_column in matched_rows.columns and not matched_rows.empty:
        matched_items = [item for item in matched_rows[item_column].map(clean_text).drop_duplicates().tolist() if item]
        matched_identifier = matched_items[0] if matched_items else ""
    elif not matched_rows.empty:
        first_column = matched_columns[0] if matched_columns else searched_columns[0]
        matched_identifier = clean_text(matched_rows.iloc[0][first_column]) if first_column in matched_rows.columns else ""

    st.session_state.last_lookup_context = {
        "decoded_code": decoded_code,
        "matched_columns": matched_columns,
        "matched_row_indices": [int(index) for index in matched_rows.index.tolist()],
        "matched_items": matched_items,
        "matched_identifier": matched_identifier,
        "item_column": item_column,
        "exact_match": exact_match,
    }

    active_scan_result = get_active_barcode_scan_result(decoded_code)
    if active_scan_result:
        render_barcode_result_panel(active_scan_result, matched_rows=matched_rows, matched_columns=matched_columns, heading="Decode + Match Summary")
    else:
        st.markdown("#### Decode + Match Summary")
        st.code(decoded_code)
        st.info("This lookup is using the current code value, but no decoder metadata is available for it in this session.")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Decoded Code", decoded_code)
    metric_columns[1].metric("Records Found", int(len(matched_rows)))
    metric_columns[2].metric("Matched Columns", int(len(matched_columns)))
    metric_columns[3].metric("Exact Match", "Yes" if exact_match else "No")
    if matched_columns:
        st.caption(f"Matched column(s): {', '.join(matched_columns)}")

    if matched_items:
        st.success("Matched item(s): " + ", ".join(matched_items[:5]))
    elif matched_identifier:
        st.success(f"Matched identifier: {matched_identifier}")

    log_record = build_barcode_log_record(
        decoded_code=decoded_code,
        matched_rows=matched_rows,
        searched_columns=searched_columns,
        source=st.session_state.get("barcode_lookup_source", "manual"),
        scan_result=active_scan_result,
        matched_columns=matched_columns,
    )

    save_columns = st.columns([1, 3])
    if save_columns[0].button("Save Lookup Log", key="save-barcode-log", use_container_width=True):
        saved_path = save_barcode_log(log_record, BARCODE_LOG_DIR)
        save_columns[1].success(f"Saved lookup log: {saved_path.name}")

    if matched_rows.empty:
        st.warning("No matching order records were found in the selected columns.")
    else:
        st.dataframe(matched_rows, use_container_width=True, hide_index=True)

    st.markdown("#### Decoded Value History")
    if st.session_state.barcode_history:
        st.dataframe(st.session_state.barcode_history, use_container_width=True, hide_index=True)
    else:
        st.info("Decoded values will appear here as you scan.")

    st.markdown("#### Saved Lookup Logs")
    saved_logs = load_barcode_logs(BARCODE_LOG_DIR)
    if saved_logs:
        st.dataframe(saved_logs[:20], use_container_width=True, hide_index=True)
    else:
        st.info("No barcode lookup logs have been saved yet.")


def render_barcode_tab() -> None:
    st.subheader("Barcode / Order Lookup")
    st.caption("Scan from an uploaded screenshot or a mirrored live camera preview, then match the decoded value against an order sheet.")
    render_barcode_decoder_status()

    mode = st.radio("Scanning mode", options=["Upload image", "Live camera"], horizontal=True)
    if mode == "Upload image":
        render_uploaded_barcode_flow()
    else:
        render_live_camera_flow()

    st.markdown("---")
    render_lookup_results()


def render_anomaly_tab() -> None:
    st.subheader("Order Risk & Anomaly Detector")
    st.caption("Analyze the uploaded order sheet for unusual prices, quantities, amounts, rare category patterns, and rows that deserve review.")

    anomaly_upload = st.file_uploader(
        "Upload an Excel or CSV file for risk analysis",
        type=["xlsx", "csv"],
        key="anomaly-order-file",
    )
    dataframe = update_order_dataframe(anomaly_upload)
    if dataframe is None:
        dataframe = get_loaded_order_dataframe()

    if dataframe is None:
        st.info("Upload an order file in this tab or in the Barcode / Order Lookup tab to unlock anomaly detection.")
        return

    state = get_anomaly_field_state(dataframe)
    identifier_column = state["anomaly_identifier_column"]
    code_columns = [column for column in state["anomaly_code_columns"] if column in dataframe.columns]
    numeric_columns = [column for column in state["anomaly_numeric_columns"] if column in dataframe.columns]
    categorical_columns = [column for column in state["anomaly_categorical_columns"] if column in dataframe.columns]
    date_column = state["anomaly_date_column"] if state["anomaly_date_column"] in dataframe.columns else ""

    prepared = prepare_anomaly_features(
        dataframe=dataframe,
        identifier_column=identifier_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        code_columns=code_columns,
        date_column=date_column,
        filter_text=state["anomaly_filter_text"],
        date_range=state["anomaly_date_range"],
    )

    working_df = prepared["working_df"]
    if working_df.empty:
        st.warning("No rows remain after the current date or barcode/item filters.")
        return

    results = combine_anomaly_signals(prepared, state["anomaly_contamination"])

    render_risk_summary_cards(results, identifier_column)
    render_matched_barcode_risk(results, identifier_column)

    top_columns = st.columns([1.2, 0.8])
    with top_columns[0]:
        render_top_anomaly_table(results, identifier_column, numeric_columns, state["anomaly_row_limit"])
    with top_columns[1]:
        render_review_recommendation_box(results)

    bottom_columns = st.columns([1.1, 0.9])
    with bottom_columns[0]:
        render_anomaly_visual(results, numeric_columns)
    with bottom_columns[1]:
        render_reason_breakdown(results, identifier_column)


apply_app_style()
st.markdown(
    """
    <div class="app-hero">
        <div style="font-size:1.7rem; font-weight:700;">Power BI Risk Console</div>
        <div style="margin-top:0.35rem; font-size:1rem; opacity:0.92;">
            A clean Streamlit shell around the embedded Power BI dashboard, focused on barcode-driven order lookup
            and practical order risk review.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

secret_overrides = load_secret_overrides()
settings = load_powerbi_settings(secret_overrides)
pages = get_report_pages(
    settings=settings,
    pbix_path=PBIX_PATH,
    workdir=APP_DIR,
    catalog_json=secret_overrides.get("POWERBI_PAGES_JSON"),
)
ensure_session_defaults(pages)
order_dataframe = get_loaded_order_dataframe()
current_page = get_current_page(pages)
render_sidebar(settings=settings, pages=pages, current_page=current_page, dataframe=order_dataframe)
current_page = get_current_page(pages)
order_dataframe = get_loaded_order_dataframe()

overview_columns = st.columns(4)
overview_columns[0].markdown(
    f'<div class="info-card"><strong>Report Pages</strong><br>{len(pages) if pages else 0}</div>',
    unsafe_allow_html=True,
)
overview_columns[1].markdown(
    f'<div class="info-card"><strong>Current Page</strong><br>{current_page.button_label if current_page else "Unavailable"}</div>',
    unsafe_allow_html=True,
)
overview_columns[2].markdown(
    f'<div class="info-card"><strong>Embed Source</strong><br>{"SDK" if settings.can_use_js_sdk else "Iframe"}</div>',
    unsafe_allow_html=True,
)
overview_columns[3].markdown(
    f'<div class="info-card"><strong>Order Data</strong><br>{st.session_state.order_file_name if order_dataframe is not None else "Not loaded"}</div>',
    unsafe_allow_html=True,
)

tab_dashboard, tab_barcode, tab_anomaly = st.tabs(
    [
        "Dashboard Viewer",
        "Barcode / Order Lookup",
        "Order Risk & Anomaly Detector",
    ]
)

with tab_dashboard:
    render_dashboard_tab(settings=settings, current_page=current_page)

with tab_barcode:
    render_barcode_tab()

with tab_anomaly:
    render_anomaly_tab()
