from __future__ import annotations

import io
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from utils.powerbi_utils import PageVisualSnapshot, PowerBIVisual, sanitize_filename


@dataclass(slots=True)
class VisualAnalysis:
    visual_name: str
    title: str
    visual_type: str
    export_status: str
    data_backed: bool
    source_note: str
    query_fields: list[str] = field(default_factory=list)
    kpi_label: str | None = None
    kpi_value: str | None = None
    top_rows: list[dict[str, Any]] = field(default_factory=list)
    bottom_rows: list[dict[str, Any]] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    trend: str | None = None
    action_hint: str | None = None
    preview_rows: list[dict[str, Any]] = field(default_factory=list)
    data_path: str | None = None
    crop_path: str | None = None
    ocr_text: str | None = None


@dataclass(slots=True)
class PageDataSummary:
    page_name: str
    page_display_name: str
    visual_analyses: list[VisualAnalysis]
    kpis: list[dict[str, str]] = field(default_factory=list)
    top_categories: list[dict[str, str]] = field(default_factory=list)
    low_categories: list[dict[str, str]] = field(default_factory=list)
    trend_highlights: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    action_points: list[str] = field(default_factory=list)
    exported_visuals: list[str] = field(default_factory=list)
    failed_visuals: list[str] = field(default_factory=list)
    visual_only_visuals: list[str] = field(default_factory=list)
    transparency_lines: list[str] = field(default_factory=list)


def _clean_numeric_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("₹", "")
    text = text.replace("€", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    text = text.replace("%", "")
    return text


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).map(_clean_numeric_text)
    return pd.to_numeric(cleaned, errors="coerce")


def _format_number(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    absolute_value = abs(float(value))
    prefix = "-" if float(value) < 0 else ""
    if absolute_value >= 1_000_000_000:
        return f"{prefix}{absolute_value / 1_000_000_000:.2f}B"
    if absolute_value >= 1_000_000:
        return f"{prefix}{absolute_value / 1_000_000:.2f}M"
    if absolute_value >= 1_000:
        return f"{prefix}{absolute_value / 1_000:.2f}K"
    if absolute_value.is_integer():
        return f"{int(float(value))}"
    return f"{float(value):.2f}"


def _normalize_display_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _load_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()
    except Exception:
        return None


def ocr_image_text(image_path: Path) -> str | None:
    if not image_path.exists():
        return None
    engine = _load_ocr_engine()
    if engine is None:
        return None

    try:
        result, _ = engine(str(image_path))
    except Exception:
        return None

    if not result:
        return None
    text = " ".join(item[1] for item in result if len(item) > 1 and item[1])
    return text.strip() or None


def attach_ocr_fallback(snapshot: PageVisualSnapshot) -> PageVisualSnapshot:
    for visual in snapshot.visuals:
        if visual.is_exported or not visual.crop_path:
            continue
        visual.ocr_text = ocr_image_text(visual.crop_path)
    return snapshot


def _read_visual_dataframe(visual: PowerBIVisual) -> pd.DataFrame | None:
    raw_csv = visual.raw_csv
    if raw_csv:
        try:
            return pd.read_csv(io.StringIO(raw_csv)).fillna("")
        except Exception:
            pass

    if visual.data_path and visual.data_path.exists():
        try:
            return pd.read_csv(visual.data_path).fillna("")
        except Exception:
            return None
    return None


def _detect_datetime_column(dataframe: pd.DataFrame) -> str | None:
    for column in dataframe.columns:
        parsed = pd.to_datetime(dataframe[column], errors="coerce")
        if parsed.notna().mean() >= 0.7:
            return column
    return None


def _preview_rows(dataframe: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    preview = dataframe.head(limit).copy()
    return [{column: _normalize_display_value(value) for column, value in row.items()} for row in preview.to_dict(orient="records")]


def analyze_visual(visual: PowerBIVisual) -> VisualAnalysis:
    dataframe = _read_visual_dataframe(visual)
    analysis = VisualAnalysis(
        visual_name=visual.name,
        title=visual.display_title,
        visual_type=visual.visual_type,
        export_status=visual.export_status,
        data_backed=bool(dataframe is not None and not dataframe.empty),
        source_note="Power BI visual export" if visual.is_exported else "Screenshot context only",
        query_fields=list(visual.query_fields),
        data_path=str(visual.data_path) if visual.data_path else None,
        crop_path=str(visual.crop_path) if visual.crop_path else None,
        ocr_text=visual.ocr_text,
    )

    if dataframe is None or dataframe.empty:
        if visual.export_error:
            analysis.highlights.append("Export was unavailable for this visual.")
        if visual.ocr_text:
            analysis.highlights.append("OCR fallback text was used for context only.")
            analysis.source_note = "OCR fallback from visual crop"
        return analysis

    analysis.preview_rows = _preview_rows(dataframe)
    numeric_columns: list[str] = []
    numeric_cache: dict[str, pd.Series] = {}
    for column in dataframe.columns:
        numeric_series = _to_numeric(dataframe[column])
        if numeric_series.notna().mean() >= 0.6 or (len(dataframe) <= 2 and numeric_series.notna().any()):
            numeric_columns.append(column)
            numeric_cache[column] = numeric_series

    datetime_column = _detect_datetime_column(dataframe)
    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns and column != datetime_column
    ]

    primary_numeric = numeric_columns[0] if numeric_columns else None
    primary_category = datetime_column or (categorical_columns[0] if categorical_columns else None)

    if primary_numeric and len(dataframe) == 1:
        raw_value = numeric_cache[primary_numeric].iloc[0]
        analysis.kpi_label = primary_numeric
        analysis.kpi_value = _format_number(raw_value)
        analysis.highlights.append(f"{primary_numeric}: {analysis.kpi_value}")

    if primary_numeric and primary_category:
        working = dataframe[[primary_category, primary_numeric]].copy()
        working["_metric"] = numeric_cache[primary_numeric]
        working = working.dropna(subset=["_metric"])

        if not working.empty:
            if datetime_column == primary_category:
                working["_dt"] = pd.to_datetime(working[primary_category], errors="coerce")
                working = working.sort_values("_dt")
                first_value = float(working["_metric"].iloc[0])
                last_value = float(working["_metric"].iloc[-1])
                diff = last_value - first_value
                direction = "increased" if diff > 0 else "decreased" if diff < 0 else "stayed flat"
                analysis.trend = (
                    f"{primary_numeric} {direction} from {_format_number(first_value)} to {_format_number(last_value)} "
                    f"across {primary_category}."
                )
                analysis.highlights.append(analysis.trend)
                if abs(diff) > max(abs(first_value), 1) * 0.3:
                    analysis.anomalies.append(
                        f"Large movement in {primary_numeric}: {_format_number(first_value)} to {_format_number(last_value)}."
                    )
            else:
                sorted_values = working.sort_values("_metric", ascending=False)
                top_row = sorted_values.iloc[0]
                bottom_row = sorted_values.iloc[-1]
                total_value = float(sorted_values["_metric"].sum()) if len(sorted_values) else 0.0
                share = float(top_row["_metric"] / total_value) if total_value else 0.0
                analysis.top_rows = [
                    {
                        primary_category: _normalize_display_value(row[primary_category]),
                        primary_numeric: _format_number(float(row["_metric"])),
                    }
                    for _, row in sorted_values.head(3).iterrows()
                ]
                analysis.bottom_rows = [
                    {
                        primary_category: _normalize_display_value(row[primary_category]),
                        primary_numeric: _format_number(float(row["_metric"])),
                    }
                    for _, row in sorted_values.tail(3).iterrows()
                ]
                analysis.highlights.append(
                    f"Top {primary_category}: {_normalize_display_value(top_row[primary_category])} at {_format_number(float(top_row['_metric']))}."
                )
                analysis.highlights.append(
                    f"Lowest {primary_category}: {_normalize_display_value(bottom_row[primary_category])} at {_format_number(float(bottom_row['_metric']))}."
                )
                if share >= 0.6:
                    analysis.anomalies.append(
                        f"{_normalize_display_value(top_row[primary_category])} contributes about {share:.0%} of this visual's total."
                    )

    if primary_numeric and len(dataframe) > 1 and not analysis.action_hint:
        metric_series = numeric_cache[primary_numeric].dropna()
        if not metric_series.empty:
            average_value = float(metric_series.mean())
            max_value = float(metric_series.max())
            min_value = float(metric_series.min())
            if max_value >= average_value * 1.5:
                analysis.anomalies.append(
                    f"{primary_numeric} has a strong spike up to {_format_number(max_value)} compared with an average of {_format_number(average_value)}."
                )
            if min_value <= average_value * 0.5 and average_value > 0:
                analysis.anomalies.append(
                    f"{primary_numeric} drops as low as {_format_number(min_value)} compared with an average of {_format_number(average_value)}."
                )

    if analysis.data_backed:
        analysis.source_note = "Power BI visual export"
    if analysis.trend:
        analysis.action_hint = "Review the recent trend and compare it with the strongest and weakest periods."
    elif analysis.top_rows:
        analysis.action_hint = "Check whether the top-performing category is overly concentrated and whether the bottom segment needs intervention."
    elif analysis.kpi_value:
        analysis.action_hint = "Track this KPI against targets and use neighboring visuals to explain the drivers."
    return analysis


def build_page_data_summary(snapshot: PageVisualSnapshot) -> PageDataSummary:
    visual_analyses = [analyze_visual(visual) for visual in snapshot.visuals]
    summary = PageDataSummary(
        page_name=snapshot.page_name,
        page_display_name=snapshot.page_display_name,
        visual_analyses=visual_analyses,
    )

    for analysis in visual_analyses:
        if analysis.data_backed:
            summary.exported_visuals.append(analysis.title)
        elif analysis.export_status == "failed":
            summary.failed_visuals.append(analysis.title)
        else:
            summary.visual_only_visuals.append(analysis.title)

        if analysis.kpi_value:
            summary.kpis.append(
                {
                    "label": analysis.title,
                    "metric": analysis.kpi_label or "Value",
                    "value": analysis.kpi_value,
                }
            )

        if analysis.top_rows:
            top_row = analysis.top_rows[0]
            label_key, value_key = list(top_row.keys())
            summary.top_categories.append(
                {
                    "visual": analysis.title,
                    "label": str(top_row[label_key]),
                    "value": str(top_row[value_key]),
                }
            )
        if analysis.bottom_rows:
            bottom_row = analysis.bottom_rows[0]
            label_key, value_key = list(bottom_row.keys())
            summary.low_categories.append(
                {
                    "visual": analysis.title,
                    "label": str(bottom_row[label_key]),
                    "value": str(bottom_row[value_key]),
                }
            )

        if analysis.trend:
            summary.trend_highlights.append(analysis.trend)

        summary.anomalies.extend(analysis.anomalies)
        if analysis.action_hint:
            summary.action_points.append(analysis.action_hint)

    if not summary.action_points:
        summary.action_points = [
            "Use the highest and lowest values to prioritize where the page needs attention.",
            "Refresh visual exports after filters change so the exact numbers stay aligned with the screenshot.",
        ]

    if summary.exported_visuals:
        summary.transparency_lines.append("KPI values and exact numbers are backed by Power BI visual export data.")
    else:
        summary.transparency_lines.append("Exact visual exports are not available for this page, so the summary relies on screenshot context and optional OCR fallback.")

    if summary.failed_visuals:
        summary.transparency_lines.append(
            f"Some visuals could not be exported: {', '.join(summary.failed_visuals[:5])}."
        )

    if summary.visual_only_visuals:
        summary.transparency_lines.append(
            f"Visual-only context was used for: {', '.join(summary.visual_only_visuals[:5])}."
        )

    if snapshot.bridge_message:
        summary.transparency_lines.append(snapshot.bridge_message)

    return summary


def build_visual_cards(summary: PageDataSummary) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for analysis in summary.visual_analyses:
        cards.append(
            {
                "title": analysis.title,
                "visual_type": analysis.visual_type,
                "source_note": analysis.source_note,
                "highlights": analysis.highlights or ["No exact export data was available for this visual."],
                "anomalies": analysis.anomalies,
                "trend": analysis.trend,
                "top_rows": analysis.top_rows,
                "bottom_rows": analysis.bottom_rows,
                "preview_rows": analysis.preview_rows,
                "crop_path": analysis.crop_path,
                "data_path": analysis.data_path,
                "query_fields": analysis.query_fields,
                "ocr_text": analysis.ocr_text,
            }
        )
    return cards


def _rows_to_text(rows: list[dict[str, Any]], prefix: str) -> list[str]:
    lines: list[str] = []
    for row in rows[:3]:
        parts = [f"{key}={value}" for key, value in row.items()]
        lines.append(f"{prefix}: " + ", ".join(parts))
    return lines


def build_data_context(summary: PageDataSummary) -> str:
    lines = [
        f"Page: {summary.page_display_name}",
        "Use Power BI exported visual data as the source of truth for all exact numbers.",
        "Use screenshot context only for layout, chart type, and visual grouping.",
    ]

    if summary.kpis:
        lines.append("KPI values:")
        for item in summary.kpis[:8]:
            lines.append(f"- {item['label']}: {item['metric']} = {item['value']}")

    if summary.top_categories:
        lines.append("Top categories:")
        for item in summary.top_categories[:8]:
            lines.append(f"- {item['visual']}: {item['label']} -> {item['value']}")

    if summary.low_categories:
        lines.append("Lowest categories:")
        for item in summary.low_categories[:8]:
            lines.append(f"- {item['visual']}: {item['label']} -> {item['value']}")

    if summary.trend_highlights:
        lines.append("Trend highlights:")
        for line in summary.trend_highlights[:8]:
            lines.append(f"- {line}")

    if summary.anomalies:
        lines.append("Anomalies:")
        for line in summary.anomalies[:8]:
            lines.append(f"- {line}")

    lines.append("Visual export coverage:")
    lines.append(f"- Exported visuals: {', '.join(summary.exported_visuals[:12]) or 'None'}")
    lines.append(f"- Failed visuals: {', '.join(summary.failed_visuals[:12]) or 'None'}")
    lines.append(f"- Visual-only visuals: {', '.join(summary.visual_only_visuals[:12]) or 'None'}")

    lines.append("Visual details:")
    for analysis in summary.visual_analyses:
        lines.append(
            f"- {analysis.title} [{analysis.visual_type}] source={analysis.source_note}; fields={', '.join(analysis.query_fields) or 'n/a'}"
        )
        lines.extend(f"  {line}" for line in analysis.highlights[:3])
        lines.extend(f"  {line}" for line in _rows_to_text(analysis.top_rows, "Top row"))
        lines.extend(f"  {line}" for line in _rows_to_text(analysis.bottom_rows, "Bottom row"))
        if analysis.ocr_text:
            lines.append(f"  OCR fallback text: {analysis.ocr_text[:180]}")

    return "\n".join(lines)


def build_hybrid_prompt(summary: PageDataSummary, mode: str, question: str | None = None) -> str:
    instructions = [
        "You are explaining a business dashboard page.",
        "Use the exported Power BI visual data as the source of truth for exact values.",
        "Use the screenshot only for layout, chart type, and visual context.",
        "Never invent numbers. If exact data is missing, say 'not available from exported visual data'.",
    ]

    if mode == "page":
        instructions.append(
            "Write a concise page explanation with these headers exactly: Theme, KPIs, Charts, Trends, Anomalies, Actions, Summary."
        )
        instructions.append("Mention exact KPI values and top or bottom values whenever the data context provides them.")
    elif mode == "executive":
        instructions.append("Write an executive summary with these headers exactly: Key Findings, Anomalies, Actions.")
        instructions.append("Return exactly 3 findings, 2 anomalies, and 2 action points.")
    elif mode == "question":
        instructions.append("Answer the user's question using exported visual data first and screenshot context second.")
        instructions.append("Keep the answer short and practical.")
        instructions.append(f"User question: {question or 'N/A'}")
    else:
        instructions.append("Write a short explanation for each visual using the provided exact values and chart context.")

    return "\n".join(instructions) + "\n\nData context:\n" + build_data_context(summary)


def save_explanation_record(record: dict[str, Any], history_dir: Path) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = record.get("timestamp") or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    page_name = sanitize_filename(str(record.get("page_name", "page")))
    mode = sanitize_filename(str(record.get("mode", "summary")))
    output_path = history_dir / f"{timestamp}_{page_name}_{mode}.json"
    output_path.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path


def load_explanation_history(history_dir: Path) -> list[dict[str, Any]]:
    if not history_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for file_path in sorted(history_dir.glob("*.json"), reverse=True):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            payload["history_file"] = str(file_path)
            records.append(payload)
        except Exception:
            continue
    return records


def history_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "timestamp": record.get("timestamp"),
                "page_name": record.get("page_name"),
                "mode": record.get("mode"),
                "model_id": record.get("model_id"),
                "screenshot_path": record.get("screenshot_path"),
                "screenshot_source": record.get("screenshot_source"),
                "source_status": record.get("source_status"),
                "exported_visual_count": len(record.get("exported_visuals", [])),
                "failed_visual_count": len(record.get("failed_visuals", [])),
                "question": record.get("question"),
                "text": record.get("text"),
                "history_file": record.get("history_file"),
            }
        )
    return pd.DataFrame(rows)


def history_to_excel_bytes(records: list[dict[str, Any]]) -> bytes:
    dataframe = history_dataframe(records)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="history")
    output.seek(0)
    return output.read()


def build_history_record(
    page_name: str,
    page_display_name: str,
    mode: str,
    model_id: str,
    text: str,
    screenshot_path: Path | None,
    screenshot_source: str | None,
    summary: PageDataSummary,
    question: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "page_name": page_name,
        "page_display_name": page_display_name,
        "mode": mode,
        "model_id": model_id,
        "question": question,
        "text": text,
        "screenshot_path": str(screenshot_path) if screenshot_path else None,
        "screenshot_source": screenshot_source,
        "source_status": summary.transparency_lines,
        "exported_visuals": summary.exported_visuals,
        "failed_visuals": summary.failed_visuals,
        "visual_only_visuals": summary.visual_only_visuals,
        "kpis": summary.kpis,
        "top_categories": summary.top_categories,
        "low_categories": summary.low_categories,
        "trend_highlights": summary.trend_highlights,
        "anomalies": summary.anomalies,
        "action_points": summary.action_points,
        "visual_analyses": [asdict(item) for item in summary.visual_analyses],
    }

