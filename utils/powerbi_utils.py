from __future__ import annotations

import copy
import difflib
import io
import json
import os
import re
import shutil
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from PIL import Image


DEFAULT_IFRAME_URL = (
    "https://app.powerbi.com/reportEmbed"
    "?reportId=c2e9503b-c8c7-417b-8ba6-465fbbc8c5e6"
    "&autoAuth=true"
    "&ctid=ad06ef22-d6dc-4a55-b4c1-c3a158f5f147"
    "&actionBarEnabled=true"
    "&reportCopilotInEmbed=true"
)
POWER_BI_SCOPE = "https://analysis.windows.net/powerbi/api/.default"
EXPORT_POLL_INTERVAL_SECONDS = 2
EXPORT_TIMEOUT_SECONDS = 120
DEFAULT_VISUAL_EXPORT_ROWS = 3000
BRIDGE_CAPTURE_MAX_AGE_SECONDS = 1800


@dataclass(slots=True)
class PowerBIPage:
    order: int
    name: str
    display_name: str
    is_active: bool = False
    local_image_path: Path | None = None
    width: float | None = None
    height: float | None = None

    @property
    def button_label(self) -> str:
        label = self.display_name.replace("_", " ").strip()
        return label or self.name


@dataclass(slots=True)
class PowerBIVisual:
    name: str
    title: str
    visual_type: str
    layout: dict[str, float] = field(default_factory=dict)
    query_fields: list[str] = field(default_factory=list)
    source: str = "pbix"
    export_status: str = "not_attempted"
    export_error: str | None = None
    data_path: Path | None = None
    crop_path: Path | None = None
    raw_csv: str | None = None
    ocr_text: str | None = None

    @property
    def display_title(self) -> str:
        return self.title or self.name

    @property
    def is_exported(self) -> bool:
        return self.export_status == "exported" and self.data_path is not None


@dataclass(slots=True)
class PageVisualSnapshot:
    page_name: str
    page_display_name: str
    page_width: float | None
    page_height: float | None
    visuals: list[PowerBIVisual]
    source_status: str
    bridge_status: str
    captured_at: str | None = None
    bridge_message: str | None = None

    @property
    def exported_visuals(self) -> list[PowerBIVisual]:
        return [visual for visual in self.visuals if visual.is_exported]

    @property
    def failed_visuals(self) -> list[PowerBIVisual]:
        return [visual for visual in self.visuals if visual.export_status == "failed"]

    @property
    def visual_only_visuals(self) -> list[PowerBIVisual]:
        return [visual for visual in self.visuals if not visual.is_exported]


@dataclass(slots=True)
class ExportedPageImage:
    path: Path
    source: str
    message: str


@dataclass(slots=True)
class BridgeServerInfo:
    url: str
    port: int
    storage_dir: Path


@dataclass(slots=True)
class PowerBISettings:
    report_id: str
    embed_url: str
    iframe_url: str
    tenant_id: str | None = None
    group_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    embed_access_token: str | None = None
    token_type: str = "Embed"
    embed_mode: str = "view"

    @property
    def can_use_js_sdk(self) -> bool:
        return bool(self.report_id and self.embed_url and self.embed_access_token)

    @property
    def can_call_rest_api(self) -> bool:
        return bool(self.report_id and self.tenant_id and self.client_id and self.client_secret)


_BRIDGE_LOCK = threading.Lock()
_BRIDGE_SERVER: BridgeServerInfo | None = None
_BRIDGE_HTTPD: ThreadingHTTPServer | None = None


def normalize_label(value: str) -> str:
    cleaned = "".join(character.lower() for character in value if character.isalnum())
    return cleaned


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized[:120] or "item"


def _pick(mapping: Mapping[str, Any], *keys: str, default: str | None = None) -> str | None:
    for key in keys:
        if key in mapping and mapping[key]:
            return str(mapping[key])
    return default


def load_powerbi_settings(overrides: Mapping[str, Any] | None = None) -> PowerBISettings:
    merged: dict[str, Any] = dict(os.environ)
    if overrides:
        merged.update({key: value for key, value in overrides.items() if value not in (None, "")})

    iframe_url = _pick(merged, "POWERBI_IFRAME_URL", "PBI_IFRAME_URL", default=DEFAULT_IFRAME_URL) or DEFAULT_IFRAME_URL
    parsed = urlparse(iframe_url)
    query = dict(parse_qsl(parsed.query))
    report_id = _pick(
        merged,
        "POWERBI_REPORT_ID",
        "PBI_REPORT_ID",
        default=query.get("reportId", ""),
    ) or ""
    tenant_id = _pick(
        merged,
        "POWERBI_TENANT_ID",
        "PBI_TENANT_ID",
        default=query.get("ctid"),
    )
    embed_url = _pick(merged, "POWERBI_EMBED_URL", "PBI_EMBED_URL", default=iframe_url) or iframe_url
    token_type = (_pick(merged, "POWERBI_TOKEN_TYPE", "PBI_TOKEN_TYPE", default="Embed") or "Embed").title()
    embed_token = _pick(
        merged,
        "POWERBI_EMBED_TOKEN",
        "PBI_EMBED_TOKEN",
        "POWERBI_ACCESS_TOKEN",
        "PBI_ACCESS_TOKEN",
    )

    return PowerBISettings(
        report_id=report_id,
        embed_url=embed_url,
        iframe_url=iframe_url,
        tenant_id=tenant_id,
        group_id=_pick(merged, "POWERBI_GROUP_ID", "PBI_GROUP_ID"),
        client_id=_pick(merged, "POWERBI_CLIENT_ID", "PBI_CLIENT_ID"),
        client_secret=_pick(merged, "POWERBI_CLIENT_SECRET", "PBI_CLIENT_SECRET"),
        embed_access_token=embed_token,
        token_type=token_type if token_type in {"Embed", "Aad"} else "Embed",
        embed_mode=_pick(merged, "POWERBI_EMBED_MODE", default="view") or "view",
    )


def _api_base_url(settings: PowerBISettings) -> str:
    if settings.group_id:
        return f"https://api.powerbi.com/v1.0/myorg/groups/{settings.group_id}/reports/{settings.report_id}"
    return f"https://api.powerbi.com/v1.0/myorg/reports/{settings.report_id}"


def _append_query(url: str, **updates: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query.update({key: value for key, value in updates.items() if value})
    return urlunparse(parsed._replace(query=urlencode(query)))


def build_page_embed_url(settings: PowerBISettings, page_name: str | None = None) -> str:
    base = settings.embed_url or settings.iframe_url or DEFAULT_IFRAME_URL
    url = _append_query(
        base,
        pageName=page_name or "",
        navContentPaneEnabled="false",
        filterPaneEnabled="false",
    )
    return url


def _streamlit_post_message_script(payload: dict[str, Any]) -> str:
    safe_payload = json.dumps(payload)
    return (
        "window.parent.postMessage("
        f'{{isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: {payload.get("height", 760)}}}, "*");'
        f'window.parent.postMessage({safe_payload}, "*");'
    )


def build_powerbi_embed_html(
    settings: PowerBISettings,
    page_name: str | None,
    height: int = 760,
    bridge_url: str | None = None,
    enable_visual_sync: bool = False,
    export_rows_limit: int = DEFAULT_VISUAL_EXPORT_ROWS,
    page_width: float | None = None,
    page_height: float | None = None,
) -> str:
    iframe_url = build_page_embed_url(settings, page_name)
    sync_note = " Visual data sync is unavailable in iframe fallback mode." if enable_visual_sync else ""

    if not settings.can_use_js_sdk:
        return f"""
        <div style="font-family:Segoe UI, sans-serif; border:1px solid #dbe3ea; border-radius:18px; overflow:hidden; background:#fff;">
            <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 14px; background:#f7fafc; border-bottom:1px solid #e5edf5;">
                <div>
                    <div style="font-size:14px; font-weight:600; color:#0f172a;">Embedded Power BI Report</div>
                    <div style="font-size:12px; color:#64748b;">Fallback iframe mode. Add a Power BI embed token to enable the JavaScript SDK.{sync_note}</div>
                </div>
                <button onclick="document.getElementById('pbi-frame').requestFullscreen()" style="border:none; background:#0f766e; color:white; padding:8px 12px; border-radius:10px; cursor:pointer;">
                    Fullscreen
                </button>
            </div>
            <iframe
                id="pbi-frame"
                title="Power BI report"
                src="{iframe_url}"
                width="100%"
                height="{height}"
                frameborder="0"
                allowfullscreen="true"
                style="display:block; background:#fff;"
            ></iframe>
        </div>
        """

    embed_config = {
        "type": "report",
        "id": settings.report_id,
        "embedUrl": settings.embed_url,
        "accessToken": settings.embed_access_token,
        "tokenType": settings.token_type,
        "pageName": page_name,
        "permissions": "Read",
        "viewMode": settings.embed_mode.lower(),
        "settings": {
            "hideErrors": True,
            "panes": {
                "filters": {"visible": False, "expanded": False},
                "pageNavigation": {"visible": False},
            },
            "background": "Transparent",
        },
        "bridgeUrl": bridge_url,
        "enableVisualSync": bool(enable_visual_sync and bridge_url),
        "exportRowsLimit": export_rows_limit,
        "pageWidth": page_width,
        "pageHeight": page_height,
    }
    config_json = json.dumps(embed_config)

    return f"""
    <div style="font-family:Segoe UI, sans-serif; border:1px solid #dbe3ea; border-radius:18px; overflow:hidden; background:#fff;">
        <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 14px; background:#f7fafc; border-bottom:1px solid #e5edf5;">
            <div>
                <div style="font-size:14px; font-weight:600; color:#0f172a;">Embedded Power BI Report</div>
                <div id="pbi-status" style="font-size:12px; color:#64748b;">Loading report...</div>
            </div>
            <button onclick="document.getElementById('report-host').requestFullscreen()" style="border:none; background:#0f766e; color:white; padding:8px 12px; border-radius:10px; cursor:pointer;">
                Fullscreen
            </button>
        </div>
        <div id="report-host" style="width:100%; height:{height}px; background:#fff;"></div>
        <div id="pbi-error" style="display:none; padding:12px 14px; background:#fff7ed; color:#9a3412; font-size:13px;"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/powerbi-client@2.23.1/dist/powerbi.min.js"></script>
    <script>
        const powerbiService = window.powerbi;
        const pbiClient = window["powerbi-client"] || {{}};
        const models = pbiClient.models || powerbiService.models;
        const host = document.getElementById("report-host");
        const status = document.getElementById("pbi-status");
        const errorBox = document.getElementById("pbi-error");
        const rawConfig = {config_json};
        let syncStarted = false;

        const embedConfig = {{
            type: rawConfig.type,
            id: rawConfig.id,
            embedUrl: rawConfig.embedUrl,
            accessToken: rawConfig.accessToken,
            tokenType: models.TokenType[rawConfig.tokenType] || models.TokenType.Embed,
            permissions: models.Permissions.Read,
            viewMode: models.ViewMode.View,
            pageName: rawConfig.pageName,
            settings: {{
                hideErrors: true,
                background: models.BackgroundType.Transparent,
                panes: {{
                    filters: {{ visible: false, expanded: false }},
                    pageNavigation: {{ visible: false }}
                }}
            }}
        }};

        function serializableLayout(layout) {{
            if (!layout) {{
                return null;
            }}
            return {{
                x: layout.x ?? null,
                y: layout.y ?? null,
                z: layout.z ?? null,
                width: layout.width ?? null,
                height: layout.height ?? null,
                displayState: layout.displayState ?? null
            }};
        }}

        async function syncCurrentPage(report) {{
            if (!rawConfig.enableVisualSync || !rawConfig.bridgeUrl || syncStarted) {{
                return;
            }}
            syncStarted = true;
            status.innerText = "Syncing visual metadata and exported data...";

            try {{
                const activePage = await report.getActivePage();
                const visuals = await activePage.getVisuals();
                const skipTypes = new Set(["slicer", "textbox", "shape", "image", "qnaVisual", "actionButton"]);
                const capturedVisuals = [];

                for (const visual of visuals) {{
                    const item = {{
                        name: visual.name,
                        title: visual.title || visual.name,
                        type: visual.type || "unknown",
                        layout: serializableLayout(visual.layout),
                        exportStatus: "skipped",
                        exportError: null,
                        csvData: null
                    }};

                    if (!skipTypes.has(item.type)) {{
                        try {{
                            const exported = await visual.exportData(models.ExportDataType.Summarized, rawConfig.exportRowsLimit);
                            item.csvData = exported?.data || "";
                            item.exportStatus = item.csvData ? "exported" : "empty";
                        }} catch (exportError) {{
                            item.exportStatus = "failed";
                            item.exportError = exportError?.message || String(exportError);
                        }}
                    }}

                    capturedVisuals.push(item);
                }}

                const payload = {{
                    reportId: rawConfig.id,
                    pageName: activePage.name,
                    pageDisplayName: activePage.displayName || activePage.name,
                    capturedAt: new Date().toISOString(),
                    pageWidth: rawConfig.pageWidth || activePage.defaultSize?.width || null,
                    pageHeight: rawConfig.pageHeight || activePage.defaultSize?.height || null,
                    visuals: capturedVisuals
                }};

                const response = await fetch(`${{rawConfig.bridgeUrl}}/page-capture`, {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify(payload)
                }});

                if (!response.ok) {{
                    throw new Error(`Bridge server returned ${{response.status}}`);
                }}

                const exportedCount = capturedVisuals.filter(item => item.exportStatus === "exported").length;
                status.innerText = `Report ready • ${{exportedCount}}/${{capturedVisuals.length}} visuals synced`;
            }} catch (captureError) {{
                console.warn(captureError);
                syncStarted = false;
                status.innerText = `Report ready • visual sync failed`;
            }}
        }}

        powerbiService.reset(host);
        const report = powerbiService.embed(host, embedConfig);

        report.on("loaded", async function() {{
            status.innerText = "Report loaded";
            { _streamlit_post_message_script({"height": height + 24}) }
            try {{
                if (rawConfig.pageName) {{
                    await report.setPage(rawConfig.pageName);
                }}
            }} catch (pageError) {{
                console.warn(pageError);
            }}
        }});

        report.on("rendered", async function() {{
            status.innerText = "Report ready";
            await syncCurrentPage(report);
        }});

        report.on("error", function(event) {{
            const message = event?.detail?.message || "Power BI embed failed.";
            errorBox.style.display = "block";
            errorBox.innerText = message;
            status.innerText = "Embed error";
        }});
    </script>
    """


def ensure_bridge_server(storage_dir: Path) -> BridgeServerInfo:
    global _BRIDGE_SERVER, _BRIDGE_HTTPD

    storage_dir.mkdir(parents=True, exist_ok=True)
    with _BRIDGE_LOCK:
        if _BRIDGE_SERVER is not None:
            return _BRIDGE_SERVER

        class BridgeRequestHandler(BaseHTTPRequestHandler):
            def _send_headers(self, status_code: int = 200, content_type: str = "application/json") -> None:
                self.send_response(status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_OPTIONS(self) -> None:
                self._send_headers(status_code=204)

            def do_GET(self) -> None:
                if self.path.rstrip("/") == "/health":
                    self._send_headers()
                    self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
                    return
                self._send_headers(status_code=404)
                self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

            def do_POST(self) -> None:
                if self.path.rstrip("/") != "/page-capture":
                    self._send_headers(status_code=404)
                    self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))
                    return

                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    raw_body = self.rfile.read(content_length)
                    payload = json.loads(raw_body.decode("utf-8"))
                    report_id = sanitize_filename(str(payload.get("reportId", "report")))
                    page_name = sanitize_filename(str(payload.get("pageName", "page")))
                    page_dir = storage_dir / report_id
                    page_dir.mkdir(parents=True, exist_ok=True)
                    capture_path = page_dir / f"{page_name}.json"
                    payload["receivedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    capture_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
                except Exception as exc:
                    self._send_headers(status_code=400)
                    self.wfile.write(json.dumps({"status": "error", "message": str(exc)}).encode("utf-8"))
                    return

                self._send_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))

            def log_message(self, format: str, *args) -> None:
                return

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), BridgeRequestHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        _BRIDGE_HTTPD = httpd
        _BRIDGE_SERVER = BridgeServerInfo(
            url=f"http://127.0.0.1:{httpd.server_port}",
            port=httpd.server_port,
            storage_dir=storage_dir,
        )
        return _BRIDGE_SERVER


def discover_local_pngs(workdir: Path) -> list[Path]:
    return sorted(path for path in workdir.glob("*.png") if path.is_file())


def _best_local_image_match(page: PowerBIPage, png_files: list[Path]) -> Path | None:
    if not png_files:
        return None

    lookup: dict[str, Path] = {}
    for png_path in png_files:
        lookup[normalize_label(png_path.stem)] = png_path

    preferred_keys = [
        normalize_label(page.display_name),
        normalize_label(page.button_label),
        normalize_label(page.name),
    ]
    for key in preferred_keys:
        if key in lookup:
            return lookup[key]

    choices = list(lookup)
    for key in preferred_keys:
        match = difflib.get_close_matches(key, choices, n=1, cutoff=0.45)
        if match:
            return lookup[match[0]]
    return None


def attach_local_images(pages: list[PowerBIPage], workdir: Path) -> list[PowerBIPage]:
    png_files = discover_local_pngs(workdir)
    used_paths: set[Path] = set()

    for page in pages:
        matched = _best_local_image_match(page, [path for path in png_files if path not in used_paths])
        if matched:
            page.local_image_path = matched
            used_paths.add(matched)

    remaining_pages = [page for page in pages if page.local_image_path is None]
    remaining_pngs = [path for path in png_files if path not in used_paths]
    for page, png_path in zip(sorted(remaining_pages, key=lambda item: item.order), remaining_pngs):
        page.local_image_path = png_path

    return pages


def _literal_to_string(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    literal = value.get("Literal")
    if not isinstance(literal, Mapping):
        return None
    raw_value = literal.get("Value")
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text


def _extract_title(visual_payload: Mapping[str, Any]) -> str | None:
    candidates = []
    visual_section = visual_payload.get("visual", {})
    for container_name in ("visualContainerObjects", "objects"):
        container = visual_section.get(container_name, {})
        title_items = container.get("title", [])
        if isinstance(title_items, list):
            candidates.extend(title_items)

    for candidate in candidates:
        properties = candidate.get("properties", {})
        text_expr = properties.get("text", {}).get("expr") if isinstance(properties, Mapping) else None
        title = _literal_to_string(text_expr)
        if title:
            return title
    return None


def _extract_query_fields(visual_payload: Mapping[str, Any]) -> list[str]:
    query_fields: list[str] = []
    query_state = visual_payload.get("visual", {}).get("query", {}).get("queryState", {})
    if not isinstance(query_state, Mapping):
        return query_fields

    for role_name, role_payload in query_state.items():
        projections = role_payload.get("projections", []) if isinstance(role_payload, Mapping) else []
        for projection in projections:
            if not isinstance(projection, Mapping):
                continue
            reference = projection.get("nativeQueryRef") or projection.get("queryRef")
            if reference:
                query_fields.append(f"{role_name}: {reference}")
    return query_fields


def _parse_visual_payload(visual_payload: Mapping[str, Any], source: str = "pbix") -> PowerBIVisual:
    position = visual_payload.get("position", {}) if isinstance(visual_payload, Mapping) else {}
    layout = {
        "x": float(position.get("x", 0) or 0),
        "y": float(position.get("y", 0) or 0),
        "width": float(position.get("width", 0) or 0),
        "height": float(position.get("height", 0) or 0),
        "z": float(position.get("z", 0) or 0),
    }
    visual_section = visual_payload.get("visual", {}) if isinstance(visual_payload, Mapping) else {}
    visual_name = str(visual_payload.get("name", "visual"))
    visual_type = str(visual_section.get("visualType", visual_payload.get("type", "unknown")))
    query_fields = _extract_query_fields(visual_payload)
    derived_title = " | ".join(field.split(":", 1)[-1].strip().split(".")[-1] for field in query_fields[:2])
    title = _extract_title(visual_payload) or derived_title or visual_name
    return PowerBIVisual(
        name=visual_name,
        title=title,
        visual_type=visual_type,
        layout=layout,
        query_fields=query_fields,
        source=source,
    )


@lru_cache(maxsize=2)
def _load_pbix_catalog(pbix_key: str) -> dict[str, Any]:
    pbix_path = Path(pbix_key.split("::", 1)[0])
    if not pbix_path.exists():
        return {"pages": [], "page_visuals": {}}

    with zipfile.ZipFile(pbix_path) as archive:
        page_order_payload = json.loads(archive.read("Report/definition/pages/pages.json"))
        active_page = page_order_payload.get("activePageName")
        page_order = page_order_payload.get("pageOrder", [])
        pages: list[PowerBIPage] = []
        page_visuals: dict[str, list[PowerBIVisual]] = {}

        for index, page_id in enumerate(page_order):
            page_payload = json.loads(archive.read(f"Report/definition/pages/{page_id}/page.json"))
            page = PowerBIPage(
                order=index,
                name=str(page_payload.get("name", page_id)),
                display_name=str(page_payload.get("displayName", page_id)),
                is_active=str(page_payload.get("name", page_id)) == active_page,
                width=float(page_payload.get("width", 0) or 0),
                height=float(page_payload.get("height", 0) or 0),
            )
            pages.append(page)

            prefix = f"Report/definition/pages/{page_id}/visuals/"
            visuals = []
            for visual_file in sorted(name for name in archive.namelist() if name.startswith(prefix) and name.endswith("/visual.json")):
                visual_payload = json.loads(archive.read(visual_file))
                visuals.append(_parse_visual_payload(visual_payload, source="pbix"))
            page_visuals[page.name] = visuals

    return {"pages": pages, "page_visuals": page_visuals}


def _pbix_cache_key(pbix_path: Path) -> str:
    resolved = pbix_path.resolve()
    if not resolved.exists():
        return str(resolved)
    return f"{resolved}::{resolved.stat().st_mtime_ns}"


def parse_pbix_pages(pbix_path: Path, workdir: Path) -> list[PowerBIPage]:
    catalog = _load_pbix_catalog(_pbix_cache_key(pbix_path))
    pages = copy.deepcopy(catalog.get("pages", []))
    return attach_local_images(pages, workdir)


def get_pbix_visuals(pbix_path: Path, page_name: str) -> list[PowerBIVisual]:
    catalog = _load_pbix_catalog(_pbix_cache_key(pbix_path))
    return copy.deepcopy(catalog.get("page_visuals", {}).get(page_name, []))


def get_pbix_page_lookup(pbix_path: Path) -> dict[str, PowerBIPage]:
    catalog = _load_pbix_catalog(_pbix_cache_key(pbix_path))
    return {page.name: copy.deepcopy(page) for page in catalog.get("pages", [])}


def load_pages_from_mapping(raw_pages: Any, workdir: Path) -> list[PowerBIPage]:
    if not isinstance(raw_pages, list):
        return []

    pages: list[PowerBIPage] = []
    for index, item in enumerate(raw_pages):
        if not isinstance(item, Mapping):
            continue
        page = PowerBIPage(
            order=int(item.get("order", index)),
            name=str(item.get("name", item.get("pageName", f"page-{index}"))),
            display_name=str(item.get("display_name", item.get("displayName", item.get("name", f"Page {index + 1}")))),
            is_active=bool(item.get("is_active", False)),
            width=float(item.get("width", 0) or 0),
            height=float(item.get("height", 0) or 0),
        )
        pages.append(page)
    return attach_local_images(sorted(pages, key=lambda page: page.order), workdir)


def _merge_local_page_details(pages: list[PowerBIPage], pbix_path: Path, workdir: Path) -> list[PowerBIPage]:
    local_lookup = get_pbix_page_lookup(pbix_path)
    merged_pages: list[PowerBIPage] = []
    for page in pages:
        local_page = local_lookup.get(page.name)
        if local_page:
            page.width = local_page.width
            page.height = local_page.height
            page.display_name = local_page.display_name or page.display_name
            page.is_active = page.is_active or local_page.is_active
        merged_pages.append(page)
    return attach_local_images(merged_pages, workdir)


def get_report_pages(settings: PowerBISettings, pbix_path: Path, workdir: Path, catalog_json: str | None = None) -> list[PowerBIPage]:
    if catalog_json:
        try:
            pages = load_pages_from_mapping(json.loads(catalog_json), workdir)
            if pages:
                return pages
        except json.JSONDecodeError:
            pass

    try:
        if settings.can_call_rest_api:
            pages = fetch_pages_from_powerbi(settings)
            if pages:
                return _merge_local_page_details(pages, pbix_path, workdir)
    except Exception:
        pass

    return parse_pbix_pages(pbix_path, workdir)


def _request_service_principal_token(settings: PowerBISettings) -> str:
    if not settings.can_call_rest_api:
        raise RuntimeError("Power BI REST credentials are missing.")

    token_url = f"https://login.microsoftonline.com/{settings.tenant_id}/oauth2/v2.0/token"
    response = requests.post(
        token_url,
        data={
            "client_id": settings.client_id,
            "client_secret": settings.client_secret,
            "scope": POWER_BI_SCOPE,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("Microsoft Entra returned no Power BI access token.")
    return token


def _rest_headers(settings: PowerBISettings) -> dict[str, str]:
    token = _request_service_principal_token(settings)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def fetch_pages_from_powerbi(settings: PowerBISettings) -> list[PowerBIPage]:
    response = requests.get(f"{_api_base_url(settings)}/pages", headers=_rest_headers(settings), timeout=30)
    response.raise_for_status()
    payload = response.json()
    pages: list[PowerBIPage] = []
    for page in sorted(payload.get("value", []), key=lambda item: int(item.get("order", 0))):
        pages.append(
            PowerBIPage(
                order=int(page.get("order", 0)),
                name=str(page.get("name", "")),
                display_name=str(page.get("displayName", page.get("name", ""))),
            )
        )
    return pages


def export_page_to_png(settings: PowerBISettings, page: PowerBIPage, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = _rest_headers(settings)
    payload = {
        "format": "PNG",
        "powerBIReportConfiguration": {
            "pages": [{"pageName": page.name}],
            "settings": {"locale": "en-US"},
        },
    }
    export_response = requests.post(
        f"{_api_base_url(settings)}/ExportTo",
        headers=headers,
        data=json.dumps(payload),
        timeout=30,
    )
    export_response.raise_for_status()
    export_id = export_response.json().get("id")
    if not export_id:
        raise RuntimeError("Power BI did not return an export job identifier.")

    deadline = time.time() + EXPORT_TIMEOUT_SECONDS
    while time.time() < deadline:
        status_response = requests.get(
            f"{_api_base_url(settings)}/exports/{export_id}",
            headers=headers,
            timeout=30,
        )
        status_response.raise_for_status()
        status_payload = status_response.json()
        status = status_payload.get("status")
        if status == "Succeeded":
            file_response = requests.get(
                f"{_api_base_url(settings)}/exports/{export_id}/file",
                headers=headers,
                timeout=60,
            )
            file_response.raise_for_status()
            output_path = output_dir / f"{page.name}.png"
            content = file_response.content
            if zipfile.is_zipfile(io.BytesIO(content)):
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    first_png = next((name for name in archive.namelist() if name.lower().endswith(".png")), None)
                    if not first_png:
                        raise RuntimeError("Power BI export returned a ZIP with no PNG pages inside.")
                    output_path.write_bytes(archive.read(first_png))
            else:
                output_path.write_bytes(content)
            return output_path
        if status == "Failed":
            raise RuntimeError(status_payload.get("error", {}).get("message", "Power BI export failed."))
        time.sleep(EXPORT_POLL_INTERVAL_SECONDS)

    raise TimeoutError("Timed out while waiting for the Power BI PNG export to finish.")


def resolve_page_image(settings: PowerBISettings, page: PowerBIPage, output_dir: Path) -> ExportedPageImage:
    output_dir.mkdir(parents=True, exist_ok=True)

    if settings.can_call_rest_api:
        try:
            export_path = export_page_to_png(settings, page, output_dir)
            return ExportedPageImage(
                path=export_path,
                source="power-bi-export",
                message="Power BI REST export succeeded for the selected page.",
            )
        except Exception as exc:
            if page.local_image_path and page.local_image_path.exists():
                fallback_path = output_dir / f"{page.name}-fallback.png"
                shutil.copy2(page.local_image_path, fallback_path)
                return ExportedPageImage(
                    path=fallback_path,
                    source="local-fallback",
                    message=f"Power BI export failed, so the app used the local page PNG instead: {exc}",
                )
            raise

    if page.local_image_path and page.local_image_path.exists():
        fallback_path = output_dir / f"{page.name}-fallback.png"
        shutil.copy2(page.local_image_path, fallback_path)
        return ExportedPageImage(
            path=fallback_path,
            source="local-fallback",
            message="Power BI REST export is not configured, so the app used the local page PNG.",
        )

    raise RuntimeError(
        "No page image is available. Configure Power BI export credentials or keep the local PNG page snapshots beside the app."
    )


def _capture_file_path(storage_dir: Path, report_id: str, page_name: str) -> Path:
    report_folder = sanitize_filename(report_id or "report")
    page_file = sanitize_filename(page_name or "page")
    return storage_dir / report_folder / f"{page_file}.json"


def load_bridge_capture(settings: PowerBISettings, page: PowerBIPage, storage_dir: Path) -> dict[str, Any] | None:
    capture_path = _capture_file_path(storage_dir, settings.report_id or "report", page.name)
    if not capture_path.exists():
        return None

    try:
        payload = json.loads(capture_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    captured_at = payload.get("capturedAt")
    if captured_at:
        try:
            parsed_time = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
            if parsed_time.tzinfo is None:
                parsed_time = parsed_time.replace(tzinfo=timezone.utc)
            age_seconds = (datetime.now(timezone.utc) - parsed_time.astimezone(timezone.utc)).total_seconds()
            payload["_is_stale"] = age_seconds > BRIDGE_CAPTURE_MAX_AGE_SECONDS
        except Exception:
            payload["_is_stale"] = False
    else:
        payload["_is_stale"] = False
    return payload


def _save_exported_csv(export_dir: Path, page_name: str, visual_name: str, csv_data: str) -> Path:
    visual_dir = export_dir / sanitize_filename(page_name)
    visual_dir.mkdir(parents=True, exist_ok=True)
    csv_path = visual_dir / f"{sanitize_filename(visual_name)}.csv"
    csv_path.write_text(csv_data, encoding="utf-8")
    return csv_path


def get_page_visual_snapshot(
    settings: PowerBISettings,
    page: PowerBIPage,
    pbix_path: Path,
    bridge_storage_dir: Path,
    export_dir: Path,
) -> PageVisualSnapshot:
    pbix_visuals = get_pbix_visuals(pbix_path, page.name)
    pbix_lookup = {visual.name: visual for visual in pbix_visuals}
    bridge_payload = load_bridge_capture(settings, page, bridge_storage_dir)

    if bridge_payload and isinstance(bridge_payload.get("visuals"), list):
        visuals: list[PowerBIVisual] = []
        for visual_payload in bridge_payload["visuals"]:
            if not isinstance(visual_payload, Mapping):
                continue
            visual_name = str(visual_payload.get("name", "visual"))
            base_visual = pbix_lookup.get(visual_name)
            title = str(visual_payload.get("title") or (base_visual.title if base_visual else visual_name))
            visual_type = str(visual_payload.get("type") or (base_visual.visual_type if base_visual else "unknown"))
            layout_payload = visual_payload.get("layout") if isinstance(visual_payload.get("layout"), Mapping) else {}
            layout = {
                "x": float(layout_payload.get("x", base_visual.layout.get("x", 0) if base_visual else 0) or 0),
                "y": float(layout_payload.get("y", base_visual.layout.get("y", 0) if base_visual else 0) or 0),
                "width": float(layout_payload.get("width", base_visual.layout.get("width", 0) if base_visual else 0) or 0),
                "height": float(layout_payload.get("height", base_visual.layout.get("height", 0) if base_visual else 0) or 0),
                "z": float(layout_payload.get("z", base_visual.layout.get("z", 0) if base_visual else 0) or 0),
            }
            raw_csv = visual_payload.get("csvData")
            data_path = None
            if raw_csv:
                data_path = _save_exported_csv(export_dir, page.name, visual_name, str(raw_csv))

            visuals.append(
                PowerBIVisual(
                    name=visual_name,
                    title=title,
                    visual_type=visual_type,
                    layout=layout,
                    query_fields=base_visual.query_fields if base_visual else [],
                    source="powerbi-js-sdk",
                    export_status=str(visual_payload.get("exportStatus", "not_attempted")),
                    export_error=str(visual_payload.get("exportError")) if visual_payload.get("exportError") else None,
                    data_path=data_path,
                    raw_csv=str(raw_csv) if raw_csv else None,
                )
            )

        existing_names = {visual.name for visual in visuals}
        for visual in pbix_visuals:
            if visual.name not in existing_names:
                visuals.append(visual)

        bridge_status = "stale" if bridge_payload.get("_is_stale") else "fresh"
        source_status = "sdk-export" if any(visual.is_exported for visual in visuals) else "sdk-metadata-only"
        return PageVisualSnapshot(
            page_name=page.name,
            page_display_name=page.display_name,
            page_width=float(bridge_payload.get("pageWidth") or page.width or 0) or None,
            page_height=float(bridge_payload.get("pageHeight") or page.height or 0) or None,
            visuals=sorted(visuals, key=lambda item: (item.layout.get("y", 0), item.layout.get("x", 0), item.layout.get("z", 0))),
            source_status=source_status,
            bridge_status=bridge_status,
            captured_at=str(bridge_payload.get("capturedAt")) if bridge_payload.get("capturedAt") else None,
            bridge_message=(
                "Visual metadata and data were synced from the embedded Power BI session."
                if source_status == "sdk-export"
                else "Visual metadata synced from Power BI, but exports were unavailable for this page."
            ),
        )

    return PageVisualSnapshot(
        page_name=page.name,
        page_display_name=page.display_name,
        page_width=page.width,
        page_height=page.height,
        visuals=pbix_visuals,
        source_status="pbix-metadata",
        bridge_status="missing" if settings.can_use_js_sdk else "unavailable",
        captured_at=None,
        bridge_message=(
            "Visual export data is not available yet. The app is using PBIX visual metadata and screenshot context."
        ),
    )


def save_visual_crops(page_image_path: Path, page: PowerBIPage, snapshot: PageVisualSnapshot, crop_dir: Path) -> PageVisualSnapshot:
    if not page_image_path.exists():
        return snapshot
    if not snapshot.page_width or not snapshot.page_height:
        return snapshot

    crop_dir.mkdir(parents=True, exist_ok=True)
    page_crop_dir = crop_dir / sanitize_filename(page.name)
    page_crop_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(page_image_path) as image:
        rgb_image = image.convert("RGB")
        scale_x = rgb_image.width / snapshot.page_width
        scale_y = rgb_image.height / snapshot.page_height

        for visual in snapshot.visuals:
            width = visual.layout.get("width", 0)
            height = visual.layout.get("height", 0)
            if width <= 0 or height <= 0:
                continue

            left = max(0, int(visual.layout.get("x", 0) * scale_x))
            top = max(0, int(visual.layout.get("y", 0) * scale_y))
            right = min(rgb_image.width, int((visual.layout.get("x", 0) + width) * scale_x))
            bottom = min(rgb_image.height, int((visual.layout.get("y", 0) + height) * scale_y))

            if right <= left or bottom <= top:
                continue

            crop_path = page_crop_dir / f"{sanitize_filename(visual.name)}.png"
            rgb_image.crop((left, top, right, bottom)).save(crop_path)
            visual.crop_path = crop_path

    return snapshot
