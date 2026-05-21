"""
Streamlit dashboard: date-range OCR stats, success/failure pie chart,
and failure_reason bar chart (nulls excluded).
"""

from __future__ import annotations

import csv
import html
import io
from datetime import date, datetime, timedelta
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from main import (
    ATTR_CHANNEL,
    ATTR_DATE,
    ATTR_FAILURE_REASON,
    ATTR_PROCESSED_AT,
    DEFAULT_FILTER_ATTR,
    DEFAULT_REGION,
    DEFAULT_STATUS_ATTR,
    DEFAULT_TABLE,
    channel_stats,
    coerce_to_date,
    date_range_set,
    dynamodb_resource,
    failure_reason_breakdown_rows,
    failure_reason_counts,
    fetch_records,
    filter_by_dates,
    generate_stats,
    group_failure_reason_counts,
    normalize_channel,
    normalize_status,
)

DATE_MODE_SINGLE = "Single day"
DATE_MODE_RANGE = "Date range"


def format_date_filter_label(start: date, end: date) -> str:
    if start == end:
        return start.isoformat()
    return f"{start.isoformat()} to {end.isoformat()}"


def resolve_date_filter(
    mode: str,
    single_day: date | None,
    range_start: date | None,
    range_end: date | None,
) -> tuple[date, date] | str:
    """
    Return (start, end) inclusive, or an error message string.

    Single day: one date. Range: both dates required; end must be after start.
    """
    if mode == DATE_MODE_SINGLE:
        if single_day is None:
            return "Please select a date."
        return single_day, single_day

    if range_start is None:
        return "Please select a start date."
    if range_end is None:
        return "Please select an end date."
    if range_end == range_start:
        return (
            "Start and end date must be different. "
            "Use Single day to analyze one calendar day."
        )
    if range_end <= range_start:
        return "Start date must be before end date."
    return range_start, range_end


@st.cache_data(ttl=300, show_spinner="Scanning DynamoDB...")
def load_table_items(region: str, table_name: str, status_attr: str) -> list[dict]:
    dynamodb = dynamodb_resource(region)
    table = dynamodb.Table(table_name)
    items, _, _ = fetch_records(table, status_attr, include_failure_reason=True)
    return items


def status_pie_chart(success: int, failed: int) -> alt.Chart:
    df = pd.DataFrame(
        {
            "status": ["Success", "Failed"],
            "count": [success, failed],
        }
    )
    df["legend"] = df["status"] + " (" + df["count"].map("{:,}".format) + ")"
    color_scale = alt.Scale(
        domain=df["legend"].tolist(),
        range=["#22c55e", "#ef4444"],
    )
    base = alt.Chart(df).encode(
        theta=alt.Theta("count:Q", stack=True),
        color=alt.Color(
            "legend:N",
            scale=color_scale,
            legend=alt.Legend(title="Status", orient="right"),
        ),
        tooltip=[
            alt.Tooltip("status:N", title="Status"),
            alt.Tooltip("count:Q", title="Count", format=","),
        ],
    )
    pie = base.mark_arc(innerRadius=50)
    labels = base.mark_text(radius=95, size=13, fill="white").encode(
        text=alt.Text("count:Q", format=","),
    )
    return (pie + labels).properties(title="Success vs Failed", height=280)


def failure_reason_bar_chart(counts: dict[str, int]) -> alt.Chart | None:
    if not counts:
        return None
    df = (
        pd.DataFrame(
            [{"failure_reason": reason, "count": count} for reason, count in counts.items()]
        )
        .sort_values("count", ascending=True)
    )
    base = alt.Chart(df).encode(
        x=alt.X("count:Q", title="Count"),
        y=alt.Y("failure_reason:N", sort="-x", title="Failure reason"),
        tooltip=[
            alt.Tooltip("failure_reason:N", title="Reason"),
            alt.Tooltip("count:Q", title="Count", format=","),
        ],
    )
    bars = base.mark_bar(color="#6366f1")
    labels = base.mark_text(align="left", baseline="middle", dx=4, color="#e2e8f0").encode(
        text=alt.Text("count:Q", format=","),
    )
    chart = (bars + labels).properties(
        title="Failures by reason (grouped, null excluded)",
        height=max(280, len(df) * 36),
    )
    return chart.configure_axis(labelLimit=0).configure_view(stroke=None)


def apply_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .stAppDeployButton,
        .stDeployButton {
            display: none !important;
        }

        html, body {
            overflow-y: auto !important;
            height: auto !important;
        }
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        .stApp {
            overflow-y: auto !important;
            min-height: 100vh;
            height: auto !important;
        }

        section[data-testid="stMain"],
        [data-testid="stMain"] > div,
        section.main,
        section.main > div.block-container,
        [data-testid="stMainBlockContainer"] {
            overflow: visible !important;
            max-height: none !important;
            padding-bottom: 5rem !important;
        }

        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        [data-testid="stSidebarUserContent"],
        [data-testid="stSidebarContent"] {
            overflow-y: auto !important;
            padding-top: 0.75rem !important;
            padding-bottom: 2rem !important;
        }

        /* Date pickers live in main content - room below so calendars open downward. */
        div[data-testid="stForm"] .stDateInput,
        div[data-testid="stForm"] .stDateInput > div,
        div[data-testid="stForm"] [data-testid="stVerticalBlock"],
        div[data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"] {
            overflow: visible !important;
            margin-bottom: 0.35rem;
        }
        div[data-testid="stForm"] [data-baseweb="popover"] {
            z-index: 999999 !important;
        }
        [data-baseweb="calendar"] {
            min-width: 18rem;
            overflow: visible !important;
        }

        .dashboard-hero {
            margin-bottom: 0.25rem;
        }
        .dashboard-hero h1 {
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.35rem;
        }
        .dashboard-hero p {
            color: rgba(250, 250, 250, 0.72);
            font-size: 1rem;
            margin: 0 0 1rem 0;
            line-height: 1.5;
        }

        div[data-testid="stForm"] {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            padding: 1.25rem 1.35rem 1.1rem;
            background: linear-gradient(
                165deg,
                rgba(30, 41, 59, 0.45) 0%,
                rgba(15, 23, 42, 0.25) 100%
            );
            margin-bottom: 1.25rem;
        }
        div[data-testid="stForm"] > div > p {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: rgba(148, 163, 184, 0.95);
            margin-bottom: 0.75rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 10px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stMetric"] label {
            font-size: 0.8rem;
            color: rgba(203, 213, 225, 0.9);
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.65rem;
            font-weight: 700;
        }

        .empty-state {
            border: 1px dashed rgba(148, 163, 184, 0.35);
            border-radius: 12px;
            padding: 2rem 1.5rem;
            text-align: center;
            color: rgba(226, 232, 240, 0.85);
            margin: 1rem 0 2rem;
            line-height: 1.6;
        }
        .empty-state strong {
            color: #f8fafc;
        }

        .sidebar-section-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: rgba(148, 163, 184, 0.9);
            margin: 0.5rem 0 0.35rem;
        }

        .export-bar {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 10px;
            padding: 0.85rem 1rem;
            margin: 0.5rem 0 1.25rem;
        }
        .export-bar p {
            margin: 0 0 0.65rem 0;
            font-size: 0.9rem;
            color: rgba(226, 232, 240, 0.88);
        }

        [data-testid="stSidebarContent"]::-webkit-scrollbar,
        section.main::-webkit-scrollbar,
        [data-testid="stAppViewContainer"]::-webkit-scrollbar {
            width: 10px;
        }
        [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb,
        section.main::-webkit-scrollbar-thumb,
        [data-testid="stAppViewContainer"]::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.55);
            border-radius: 6px;
        }

        /* Chart columns: avoid stretching short charts to match a tall neighbor. */
        [data-testid="stHorizontalBlock"] [data-testid="column"] {
            align-content: flex-start;
        }
        [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] {
            gap: 0.25rem;
        }
        .section-heading {
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0.35rem 0 0.25rem 0;
            line-height: 1.3;
        }

        .breakdown-table-wrap {
            overflow: auto;
            max-height: 560px;
            margin-top: 0.25rem;
            border: 1px solid rgba(250, 250, 250, 0.16);
            border-radius: 0.5rem;
        }
        .breakdown-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }
        .breakdown-table th,
        .breakdown-table td {
            border: 1px solid rgba(250, 250, 250, 0.12);
            padding: 0.5rem 0.75rem;
            vertical-align: top;
            text-align: left;
        }
        .breakdown-table th {
            position: sticky;
            top: 0;
            background: rgb(14, 17, 23);
            z-index: 1;
            font-weight: 600;
        }
        .breakdown-table .cell-scroll {
            max-height: 140px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.45;
        }
        .breakdown-table .cell-numeric {
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_calendar_placement_script() -> None:
    """Nudge BaseWeb popovers to open below date inputs when possible."""
    st.components.v1.html(
        """
        <script>
        (function () {
            const parent = window.parent.document;
            const ROOT = parent.querySelector('[data-testid="stForm"]');
            if (!ROOT) return;

            function nudgePopovers() {
                parent.querySelectorAll('[data-baseweb="popover"]').forEach((pop) => {
                    if (!pop.querySelector('[data-baseweb="calendar"]')) return;
                    const anchor = ROOT.querySelector(".stDateInput input:focus")
                        || ROOT.querySelector(".stDateInput [aria-expanded='true']");
                    if (!anchor) return;
                    const rect = anchor.getBoundingClientRect();
                    const popRect = pop.getBoundingClientRect();
                    if (popRect.top < rect.bottom) return;
                    const gap = 6;
                    pop.style.top = (rect.bottom + gap) + "px";
                    pop.style.bottom = "auto";
                    pop.style.transform = "none";
                });
            }

            const obs = new MutationObserver(() => {
                requestAnimationFrame(nudgePopovers);
            });
            obs.observe(parent.body, { childList: true, subtree: true });
            parent.addEventListener("click", () => {
                setTimeout(nudgePopovers, 50);
            }, true);
        })();
        </script>
        """,
        height=0,
    )


CHART_SCROLL_MAX = 380

_BREAKDOWN_COLUMNS: list[tuple[str, str, bool]] = [
    ("failure_reason", "Failure reason (grouped)", True),
    ("count", "Count", False),
    ("share_%", "Share %", False),
    ("full_error_message", "Full error message", True),
]

_CHANNEL_COLUMNS: list[tuple[str, str, bool]] = [
    ("channel", "Channel", True),
    ("total", "Total", False),
    ("success", "Success", False),
    ("failed", "Failed", False),
]


def _format_breakdown_cell(column: str, value: object) -> str:
    if column in ("count", "total", "success", "failed"):
        return f"{int(value):,}"
    if column == "share_%":
        return f"{float(value):.1f}"
    return html.escape("" if value is None else str(value))


def render_scrollable_breakdown_table(breakdown: pd.DataFrame) -> None:
    """HTML table with per-cell scroll for long text (error messages, reasons)."""
    header_cells = "".join(
        f"<th>{html.escape(label)}</th>" for _, label, _ in _BREAKDOWN_COLUMNS
    )
    body_rows: list[str] = []
    for _, row in breakdown.iterrows():
        cells: list[str] = []
        for key, _, scrollable in _BREAKDOWN_COLUMNS:
            content = _format_breakdown_cell(key, row[key])
            if scrollable:
                cells.append(f'<td><div class="cell-scroll">{content}</div></td>')
            else:
                cells.append(f'<td class="cell-numeric">{content}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = (
        "<div class=\"breakdown-table-wrap\">"
        '<table class="breakdown-table">'
        + f"<thead><tr>{header_cells}</tr></thead>"
        + f"<tbody>{''.join(body_rows)}</tbody>"
        + "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_channel_stats_table(channel_df: pd.DataFrame) -> None:
    header_cells = "".join(
        f"<th>{html.escape(label)}</th>" for _, label, _ in _CHANNEL_COLUMNS
    )
    body_rows: list[str] = []
    for _, row in channel_df.iterrows():
        cells: list[str] = []
        for key, _, scrollable in _CHANNEL_COLUMNS:
            content = _format_breakdown_cell(key, row[key])
            if scrollable:
                cells.append(f'<td><div class="cell-scroll">{content}</div></td>')
            else:
                cells.append(f'<td class="cell-numeric">{content}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = (
        "<div class=\"breakdown-table-wrap\">"
        '<table class="breakdown-table">'
        + f"<thead><tr>{header_cells}</tr></thead>"
        + f"<tbody>{''.join(body_rows)}</tbody>"
        + "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _altair_chart_height(chart: alt.Chart) -> int:
    spec = chart.to_dict()
    h = spec.get("height")
    return int(h) if h is not None else 280


def render_scrollable_chart(chart: alt.Chart) -> None:
    chart_height = _altair_chart_height(chart)
    if chart_height <= CHART_SCROLL_MAX:
        st.altair_chart(chart, use_container_width=True)
        return
    with st.container(height=CHART_SCROLL_MAX, border=False):
        st.altair_chart(chart, use_container_width=True)


def _csv_filename(prefix: str, start: date, end: date) -> str:
    if start == end:
        return f"{prefix}_{start.isoformat()}.csv"
    return f"{prefix}_{start.isoformat()}_to_{end.isoformat()}.csv"


def _write_channel_breakdown_csv(
    w: csv.writer,
    channel_rows: list[dict[str, Any]] | None,
) -> None:
    if not channel_rows:
        return
    w.writerow([])
    w.writerow(["Channel breakdown"])
    w.writerow(["Channel", "Total", "Success", "Failed"])
    for row in channel_rows:
        w.writerow(
            [
                row["channel"],
                int(row["total"]),
                int(row["success"]),
                int(row["failed"]),
            ]
        )


def build_summary_breakdown_csv(
    *,
    start: date,
    end: date,
    filter_by: str,
    table_name: str,
    stats: dict[str, int],
    success_rate: float,
    records_scanned: int,
    breakdown: pd.DataFrame | None,
    channel_rows: list[dict[str, Any]] | None = None,
) -> bytes:
    """CSV: filter metadata, summary metrics, and failure breakdown table."""
    buf = io.StringIO()
    w = csv.writer(buf)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w.writerow(["OCR Processing Dashboard Export"])
    w.writerow(["Generated at", generated])
    w.writerow(["Date range", format_date_filter_label(start, end)])
    w.writerow(["Date field", filter_by])
    w.writerow(["DynamoDB table", table_name])
    w.writerow(["Records scanned (full table)", records_scanned])
    w.writerow([])

    w.writerow(["Summary"])
    w.writerow(["Metric", "Value"])
    w.writerow(["Total records (filtered)", stats["total"]])
    w.writerow(["Success", stats["success"]])
    w.writerow(["Failed", stats["failed"]])
    w.writerow(["Success rate (%)", f"{success_rate:.1f}"])

    _write_channel_breakdown_csv(w, channel_rows)

    if breakdown is not None and not breakdown.empty:
        w.writerow([])
        w.writerow(["Failure reason breakdown"])
        w.writerow(
            [
                "Failure reason (grouped)",
                "Count",
                "Share %",
                "Full error message",
            ]
        )
        for _, row in breakdown.iterrows():
            w.writerow(
                [
                    row["failure_reason"],
                    int(row["count"]),
                    float(row["share_%"]),
                    row["full_error_message"],
                ]
            )

    return buf.getvalue().encode("utf-8-sig")


def build_full_report_csv(
    *,
    start: date,
    end: date,
    filter_by: str,
    table_name: str,
    stats: dict[str, int],
    success_rate: float,
    records_scanned: int,
    breakdown: pd.DataFrame | None,
    channel_rows: list[dict[str, Any]] | None,
    filtered_items: list[dict[str, Any]],
    status_attr: str,
) -> bytes:
    """Single CSV: metadata, summary, breakdown table, and filtered records."""
    buf = io.StringIO()
    w = csv.writer(buf)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w.writerow(["OCR Processing Dashboard Report"])
    w.writerow(["Generated at", generated])
    w.writerow(["Date range", format_date_filter_label(start, end)])
    w.writerow(["Date field", filter_by])
    w.writerow(["DynamoDB table", table_name])
    w.writerow(["Records scanned (full table)", records_scanned])
    w.writerow([])

    w.writerow(["Summary"])
    w.writerow(["Metric", "Value"])
    w.writerow(["Total records (filtered)", stats["total"]])
    w.writerow(["Success", stats["success"]])
    w.writerow(["Failed", stats["failed"]])
    w.writerow(["Success rate (%)", f"{success_rate:.1f}"])

    _write_channel_breakdown_csv(w, channel_rows)

    if breakdown is not None and not breakdown.empty:
        w.writerow([])
        w.writerow(["Failure reason breakdown"])
        w.writerow(
            [
                "Failure reason (grouped)",
                "Count",
                "Share %",
                "Full error message",
            ]
        )
        for _, row in breakdown.iterrows():
            w.writerow(
                [
                    row["failure_reason"],
                    int(row["count"]),
                    float(row["share_%"]),
                    row["full_error_message"],
                ]
            )

    w.writerow([])
    w.writerow(["Filtered records"])
    w.writerow(
        [
            "date",
            "processed_at",
            "filter_date",
            "channel",
            "status",
            "status_normalized",
            "failure_reason",
        ]
    )
    for item in filtered_items:
        w.writerow(
            [
                item.get(ATTR_DATE),
                item.get(ATTR_PROCESSED_AT),
                coerce_to_date(item.get(filter_by)),
                normalize_channel(item.get(ATTR_CHANNEL)),
                item.get(status_attr),
                normalize_status(item.get(status_attr)),
                item.get(ATTR_FAILURE_REASON),
            ]
        )

    return buf.getvalue().encode("utf-8-sig")


def build_filtered_records_csv(
    items: list[dict[str, Any]],
    *,
    status_attr: str,
    filter_by: str,
) -> bytes:
    """CSV: one row per filtered record."""
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "date": item.get(ATTR_DATE),
                "processed_at": item.get(ATTR_PROCESSED_AT),
                "filter_date": coerce_to_date(item.get(filter_by)),
                "channel": normalize_channel(item.get(ATTR_CHANNEL)),
                "status": item.get(status_attr),
                "status_normalized": normalize_status(item.get(status_attr)),
                "failure_reason": item.get(ATTR_FAILURE_REASON),
            }
        )
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8-sig")


def render_download_report_button(
    *,
    start: date,
    end: date,
    filter_by: str,
    table_name: str,
    stats: dict[str, int],
    success_rate: float,
    records_scanned: int,
    breakdown: pd.DataFrame | None,
    channel_rows: list[dict[str, Any]] | None,
    filtered_items: list[dict[str, Any]],
    status_attr: str,
) -> None:
    report_csv = build_full_report_csv(
        start=start,
        end=end,
        filter_by=filter_by,
        table_name=table_name,
        stats=stats,
        success_rate=success_rate,
        records_scanned=records_scanned,
        breakdown=breakdown,
        channel_rows=channel_rows,
        filtered_items=filtered_items,
        status_attr=status_attr,
    )
    st.download_button(
        label="Download report",
        data=report_csv,
        file_name=_csv_filename("ocr_report", start, end),
        mime="text/csv",
        type="primary",
        use_container_width=True,
        help="CSV with summary, channel breakdown, failure breakdown, and all records for the selected date range.",
    )


def _range_date_bounds() -> tuple[date | None, date | None]:
    """Return (max_start, min_end) so start is always strictly before end."""
    range_start = st.session_state.get("filter_range_start")
    range_end = st.session_state.get("filter_range_end")
    max_start = range_end - timedelta(days=1) if range_end else None
    min_end = range_start + timedelta(days=1) if range_start else None
    return max_start, min_end


def _on_range_start_change() -> None:
    start = st.session_state.get("filter_range_start")
    end = st.session_state.get("filter_range_end")
    if start is not None and end is not None and start >= end:
        st.session_state.filter_range_end = None


def _on_range_end_change() -> None:
    start = st.session_state.get("filter_range_start")
    end = st.session_state.get("filter_range_end")
    if start is not None and end is not None and end <= start:
        st.session_state.filter_range_start = None


def _on_date_mode_change() -> None:
    """Drop values from the mode the user left so fields don't carry over."""
    mode = st.session_state.date_mode
    if mode == DATE_MODE_SINGLE:
        st.session_state.pop("filter_range_start", None)
        st.session_state.pop("filter_range_end", None)
    else:
        st.session_state.pop("filter_single_day", None)


def _init_session_defaults() -> None:
    defaults = {
        "report_ready": False,
        "date_mode": DATE_MODE_SINGLE,
        "_prev_date_mode": DATE_MODE_SINGLE,
        "filter_single_day": None,
        "filter_range_start": None,
        "filter_range_end": None,
        "filter_by_field": DEFAULT_FILTER_ATTR,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar_settings() -> tuple[str, str]:
    with st.sidebar:
        st.header("Settings")
        st.markdown(
            '<p class="sidebar-section-label">Data source</p>',
            unsafe_allow_html=True,
        )
        table_name = st.text_input(
            "DynamoDB table",
            value=DEFAULT_TABLE,
            help="Table scanned for OCR records.",
        )
        status_attr = st.text_input(
            "Status attribute",
            value=DEFAULT_STATUS_ATTR,
            help="Attribute used for success / failed counts.",
        )
    return table_name, status_attr


def render_filter_form() -> tuple[str, date | None, date | None, date | None, str, bool]:
    """Date filters in main content so calendar popovers open below the field."""
    view_report = False

    st.markdown("**Choose dates**")
    date_mode = st.radio(
        "How do you want to filter?",
        options=[DATE_MODE_SINGLE, DATE_MODE_RANGE],
        horizontal=True,
        key="date_mode",
        on_change=_on_date_mode_change,
        help="Single day for one calendar day, or a start-end range where end is after start.",
    )
    st.session_state._prev_date_mode = date_mode

    if date_mode == DATE_MODE_SINGLE:
        st.date_input(
            "Select date",
            value=None,
            key="filter_single_day",
            help="Pick the calendar day to analyze.",
        )
    else:
        max_start, min_end = _range_date_bounds()
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.date_input(
                "Start date",
                value=None,
                key="filter_range_start",
                max_value=max_start,
                on_change=_on_range_start_change,
                help="First day of the range; must be before the end date.",
            )
        with d_col2:
            st.date_input(
                "End date",
                value=None,
                key="filter_range_end",
                min_value=min_end,
                on_change=_on_range_end_change,
                help="Last day of the range; must be after the start date.",
            )

    st.markdown("**2. Match records using**")
    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        st.selectbox(
            "Date field",
            options=[ATTR_DATE, ATTR_PROCESSED_AT],
            index=0 if DEFAULT_FILTER_ATTR == ATTR_DATE else 1,
            key="filter_by_field",
            help="Which attribute defines the calendar day for filtering.",
        )
    with f_col2:
        st.write("")
        st.caption(
            f"`{ATTR_DATE}` = calendar day | `{ATTR_PROCESSED_AT}` = timestamp day"
        )

    view_report = st.button(
        "View report",
        type="primary",
        use_container_width=True,
    )

    single_day = st.session_state.filter_single_day
    range_start = st.session_state.filter_range_start
    range_end = st.session_state.filter_range_end
    filter_by = st.session_state.filter_by_field

    return date_mode, single_day, range_start, range_end, filter_by, view_report


def main() -> None:
    st.set_page_config(
        page_title="OCR Analytics",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_dashboard_styles()
    apply_calendar_placement_script()
    _init_session_defaults()

    st.markdown(
        """
        <div class="dashboard-hero">
            <h1>OCR Processing Dashboard</h1>
            <p>View success and failure rates for OCR records. Pick a date below, then generate your report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    table_name, status_attr = render_sidebar_settings()
    date_mode, single_day, range_start, range_end, filter_by, view_report = (
        render_filter_form()
    )

    if view_report:
        resolved = resolve_date_filter(
            date_mode, single_day, range_start, range_end
        )
        if isinstance(resolved, str):
            st.error(resolved)
            st.session_state.report_ready = False
        else:
            st.session_state.report_ready = True
            st.session_state.last_filter = {
                "date_mode": date_mode,
                "single_day": single_day,
                "range_start": range_start,
                "range_end": range_end,
                "filter_by": filter_by,
                "table_name": table_name,
                "status_attr": status_attr,
            }

    if not st.session_state.report_ready:
        st.markdown(
            """
            <div class="empty-state">
                <strong>Get started</strong><br>
                Choose <strong>Single day</strong> or <strong>Date range</strong>, select your date(s) in the panel above
                (calendars open below the field), then click <strong>View report</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    saved = st.session_state.get("last_filter", {})
    date_mode = saved.get("date_mode", date_mode)
    single_day = saved.get("single_day", single_day)
    range_start = saved.get("range_start", range_start)
    range_end = saved.get("range_end", range_end)
    filter_by = saved.get("filter_by", filter_by)
    table_name = saved.get("table_name", table_name)
    status_attr = saved.get("status_attr", status_attr)

    resolved = resolve_date_filter(date_mode, single_day, range_start, range_end)
    if isinstance(resolved, str):
        st.warning(resolved)
        return
    start, end = resolved

    try:
        target_dates = date_range_set(start, end)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        all_items = load_table_items(DEFAULT_REGION, table_name, status_attr)
    except Exception as exc:
        st.error(f"Could not load data from DynamoDB: {exc}")
        return

    if not all_items:
        st.warning("No records returned. Check table name, region, and credentials.")
        return

    filtered = filter_by_dates(all_items, filter_by, target_dates)
    stats = generate_stats(filtered, status_attr)
    channel_rows = channel_stats(filtered, status_attr)
    reason_counts = group_failure_reason_counts(failure_reason_counts(filtered))

    success_rate = (
        (stats["success"] / stats["total"] * 100) if stats["total"] else 0.0
    )

    breakdown_rows = failure_reason_breakdown_rows(filtered)
    breakdown: pd.DataFrame | None = None
    if breakdown_rows:
        breakdown = pd.DataFrame(breakdown_rows)
        breakdown["share_%"] = (
            breakdown["count"] / breakdown["count"].sum() * 100
        ).round(1)

    title_col, download_col = st.columns([4, 1])
    with title_col:
        st.subheader(f"Summary | {format_date_filter_label(start, end)}")
        st.caption(f"Filtered on `{filter_by}` | {len(all_items):,} records scanned")
    with download_col:
        st.write("")
        render_download_report_button(
            start=start,
            end=end,
            filter_by=filter_by,
            table_name=table_name,
            stats=stats,
            success_rate=success_rate,
            records_scanned=len(all_items),
            breakdown=breakdown,
            channel_rows=channel_rows,
            filtered_items=filtered,
            status_attr=status_attr,
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total records", f"{stats['total']:,}")
    col2.metric("Success", f"{stats['success']:,}")
    col3.metric("Failed", f"{stats['failed']:,}")
    col4.metric("Success rate", f"{success_rate:.1f}%")

    st.divider()
    st.markdown("#### Charts")

    pie_chart = (
        status_pie_chart(stats["success"], stats["failed"])
        if stats["success"] + stats["failed"] > 0
        else None
    )
    bar_chart = failure_reason_bar_chart(reason_counts)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        if pie_chart is None:
            st.info("No success or failed records in this range for the pie chart.")
        else:
            st.altair_chart(pie_chart, use_container_width=True)
    with chart_right:
        if bar_chart is None:
            st.info("No non-null failure reasons in this date range.")
        else:
            render_scrollable_chart(bar_chart)

    if channel_rows:
        st.markdown(
            '<p class="section-heading">Channel breakdown</p>',
            unsafe_allow_html=True,
        )
        st.caption("Total, success, and failed counts per channel for the selected date range.")
        render_channel_stats_table(pd.DataFrame(channel_rows))

    if breakdown is not None:
        st.subheader("Failure reason breakdown")
        st.caption("Scroll inside each cell to read the full failure reason or error message.")
        render_scrollable_breakdown_table(breakdown)


if __name__ == "__main__":
    main()
