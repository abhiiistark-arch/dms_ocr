"""
Scan DynamoDB OCR table, filter by selected date(s), and report totals plus
success/failed status counts.

Date filtering accepts common stored formats (YYYY-MM-DD, ISO datetimes, epoch).
"""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Iterable, Optional, Set, Tuple

import boto3
from botocore.exceptions import ClientError

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ===================================
# CONFIG
# ===================================

DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
DEFAULT_TABLE = os.environ.get("DYNAMODB_TABLE", "JIO-DC-OCR-PROD-DB")
# If your attribute is not literally "date", set DATE_ATTR or env DYNAMODB_DATE_ATTR
# `date` = calendar day only (e.g. 2026-04-02). `processed_at` = same day + timestamp.
ATTR_DATE = "date"
ATTR_PROCESSED_AT = "processed_at"
DEFAULT_FILTER_ATTR = os.environ.get("DYNAMODB_FILTER_ATTR", ATTR_DATE)
DEFAULT_STATUS_ATTR = os.environ.get("DYNAMODB_STATUS_ATTR", "status")
DEFAULT_FAILURE_REASON_ATTR = os.environ.get(
    "DYNAMODB_FAILURE_REASON_ATTR", "failure_reason"
)
ATTR_FAILURE_REASON = DEFAULT_FAILURE_REASON_ATTR
DEFAULT_CHANNEL_ATTR = os.environ.get("DYNAMODB_CHANNEL_ATTR", "channel")
ATTR_CHANNEL = DEFAULT_CHANNEL_ATTR


def dynamodb_resource(region: str):
    kwargs: dict[str, Any] = {"region_name": region}
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    session_token = os.environ.get("AWS_SESSION_TOKEN")
    if key and secret:
        kwargs["aws_access_key_id"] = key
        kwargs["aws_secret_access_key"] = secret
        if session_token:
            kwargs["aws_session_token"] = session_token
    return boto3.resource("dynamodb", **kwargs)


def parse_cli_date(s: str) -> date:
    """Parse YYYY-MM-DD from CLI."""
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def normalize_status(raw: Any) -> str:
    """Map DynamoDB status value to a canonical string for counting."""
    if raw is None:
        return ""
    if isinstance(raw, Decimal):
        raw = str(raw)
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.strip().lower()

    if s in ("success", "succeeded", "ok", "pass", "passed", "complete", "completed"):
        return "success"
    if s in ("failed", "failure", "fail", "error", "errors", "rejected"):
        return "failed"
    return s


def coerce_to_date(value: Any) -> Optional[date]:
    """
    Turn a DynamoDB attribute into a calendar date, or None if unknown.
    Handles string dates, ISO datetimes, int/Decimal epoch seconds or ms.
    """
    if value is None:
        return None

    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, Decimal):
        value = int(value) if value == int(value) else float(value)

    if isinstance(value, (int, float)):
        n = float(value)
        # Heuristic: milliseconds vs seconds
        if n > 1e12:
            n = n / 1000.0
        if n > 1e11:
            # Still huge — unlikely epoch; give up
            return None
        return datetime.fromtimestamp(n, tz=timezone.utc).date()

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Calendar-only field: "2026-04-02"
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass
        # processed_at style: "2026-04-02 14:30:00" or "2026-04-02T14:30:00.123Z"
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
        ):
            try:
                return datetime.strptime(s.replace("Z", "+0000"), fmt).date()
            except ValueError:
                pass
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            try:
                return datetime.strptime(s[:10], "%Y-%m-%d").date()
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        except ValueError:
            return None

    return None


def date_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--start-date",
        help="Range start (YYYY-MM-DD). Use with --end-date or alone for one day.",
    )
    p.add_argument(
        "--end-date",
        help="Range end inclusive (YYYY-MM-DD). Defaults to --start-date if omitted.",
    )
    p.add_argument(
        "--dates",
        help="Comma-separated exact calendar days (YYYY-MM-DD) to include.",
    )


def resolve_target_dates(ns: argparse.Namespace) -> Optional[Set[date]]:
    """
    Return set of dates to keep, or None meaning 'no date filter / all data'.
    """
    if ns.dates:
        out: Set[date] = set()
        for part in ns.dates.split(","):
            part = part.strip()
            if part:
                out.add(parse_cli_date(part))
        return out

    if ns.start_date:
        start = parse_cli_date(ns.start_date)
        end = parse_cli_date(ns.end_date) if ns.end_date else start
        if end < start:
            raise SystemExit("--end-date must be on or after --start-date")
        cur = start
        rng: Set[date] = set()
        while cur <= end:
            rng.add(cur)
            cur = date.fromordinal(cur.toordinal() + 1)
        return rng

    return None


def fetch_records(
    table,
    status_attr: str,
    *,
    include_failure_reason: bool = True,
) -> Tuple[list[dict[str, Any]], int, int]:
    """
    Scan table with projection. Returns (items, total_scanned, total_returned).
    Fetches `date`, `processed_at`, status, channel, and optionally failure_reason.
    """
    items: list[dict[str, Any]] = []
    total_scanned = 0
    total_count = 0

    expr_names = {
        "#date": ATTR_DATE,
        "#proc": ATTR_PROCESSED_AT,
        "#s": status_attr,
        "#ch": ATTR_CHANNEL,
    }
    projection = "#date, #proc, #s, #ch"
    if include_failure_reason:
        expr_names["#fr"] = ATTR_FAILURE_REASON
        projection += ", #fr"

    try:
        response = table.scan(
            ProjectionExpression=projection,
            ExpressionAttributeNames=expr_names,
            ReturnConsumedCapacity="TOTAL",
        )
    except ClientError as e:
        print(f"DynamoDB Error: {e.response['Error']['Message']}")
        return items, 0, 0

    while True:
        total_scanned += int(response.get("ScannedCount") or 0)
        total_count += int(response.get("Count") or 0)
        items.extend(response.get("Items") or [])

        lek = response.get("LastEvaluatedKey")
        if not lek:
            break
        try:
            response = table.scan(
                ProjectionExpression=projection,
                ExpressionAttributeNames=expr_names,
                ExclusiveStartKey=lek,
                ReturnConsumedCapacity="TOTAL",
            )
        except ClientError as e:
            print(f"DynamoDB Error: {e.response['Error']['Message']}")
            break

    return items, total_scanned, total_count


def filter_by_dates(
    items: Iterable[dict[str, Any]],
    date_attr: str,
    allowed: Optional[Set[date]],
) -> list[dict[str, Any]]:
    if allowed is None:
        return list(items)

    out: list[dict[str, Any]] = []
    for item in items:
        raw = item.get(date_attr)
        d = coerce_to_date(raw)
        if d is None:
            continue
        if d in allowed:
            out.append(item)
    return out


def normalize_channel(raw: Any) -> str:
    if raw is None:
        return "(unknown)"
    if isinstance(raw, Decimal):
        raw = str(raw)
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.strip()
    return s if s else "(unknown)"


def channel_stats(
    items: list[dict[str, Any]],
    status_attr: str,
    channel_attr: str = ATTR_CHANNEL,
) -> list[dict[str, Any]]:
    """Per-channel total, success, and failed counts."""
    buckets: dict[str, dict[str, int]] = {}
    for item in items:
        ch = normalize_channel(item.get(channel_attr))
        if ch not in buckets:
            buckets[ch] = {"total": 0, "success": 0, "failed": 0}
        buckets[ch]["total"] += 1
        st = normalize_status(item.get(status_attr))
        if st == "success":
            buckets[ch]["success"] += 1
        elif st == "failed":
            buckets[ch]["failed"] += 1

    rows = [{"channel": ch, **counts} for ch, counts in buckets.items()]
    rows.sort(key=lambda r: r["total"], reverse=True)
    return rows


def generate_stats(
    items: list[dict[str, Any]],
    status_attr: str,
) -> dict[str, int]:
    success_count = 0
    failed_count = 0

    for item in items:
        st = normalize_status(item.get(status_attr))
        if st == "success":
            success_count += 1
        elif st == "failed":
            failed_count += 1

    return {
        "total": len(items),
        "success": success_count,
        "failed": failed_count,
    }


def normalize_failure_reason(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, Decimal):
        raw = str(raw)
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.strip()
    return s if s else None


def failure_reason_counts(
    items: list[dict[str, Any]],
    failure_attr: str = ATTR_FAILURE_REASON,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        reason = normalize_failure_reason(item.get(failure_attr))
        if reason is None:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def canonical_failure_reason(reason: str) -> str:
    """Collapse variant messages (e.g. per-field) into one display label."""
    s = reason.strip()
    if not s:
        return s
    lower = s.lower()
    families = (
        (("missing mandatory", "missing mandato", "mandatory field"), "Missing mandatory field"),
        (("duplicate policy",), "Duplicate policy"),
        (("invalid format", "format error", "invalid date"), "Invalid format"),
        (("timeout", "timed out"), "Timeout"),
        (("ocr", "extraction failed"), "OCR / extraction error"),
    )
    for keywords, label in families:
        if any(k in lower for k in keywords):
            return label
    for sep in (":", " - ", " — "):
        if sep in s:
            head = s.split(sep, 1)[0].strip()
            if len(head) >= 12:
                return head
    return s


def group_failure_reason_counts(counts: dict[str, int]) -> dict[str, int]:
    grouped: dict[str, int] = {}
    for reason, count in counts.items():
        key = canonical_failure_reason(reason)
        grouped[key] = grouped.get(key, 0) + count
    return grouped


def failure_reason_breakdown_rows(
    items: list[dict[str, Any]],
    failure_attr: str = ATTR_FAILURE_REASON,
) -> list[dict[str, Any]]:
    """
    Table rows: grouped failure label, count, and full raw error text(s).
    Multiple raw messages under one group are listed with per-variant counts.
    """
    raw_counts = failure_reason_counts(items, failure_attr)
    by_group: dict[str, dict[str, int]] = {}
    for raw, count in raw_counts.items():
        group = canonical_failure_reason(raw)
        by_group.setdefault(group, {})
        by_group[group][raw] = by_group[group].get(raw, 0) + count

    rows: list[dict[str, Any]] = []
    for group, variants in by_group.items():
        total = sum(variants.values())
        sorted_variants = sorted(variants.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_variants) == 1:
            full_message = sorted_variants[0][0]
        else:
            full_message = "\n\n".join(
                f"{msg} ({cnt:,})" for msg, cnt in sorted_variants
            )
        rows.append(
            {
                "failure_reason": group,
                "count": total,
                "full_error_message": full_message,
            }
        )
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


def date_range_set(start: date, end: date) -> Set[date]:
    if end < start:
        raise ValueError("end date must be on or after start date")
    cur = start
    out: Set[date] = set()
    while cur <= end:
        out.add(cur)
        cur = date.fromordinal(cur.toordinal() + 1)
    return out


def print_sample(items: list[dict[str, Any]], status_attr: str, n: int) -> None:
    print(f"\n--- Sample of up to {n} raw items ---\n")
    for i, it in enumerate(items[:n]):
        d_raw = it.get(ATTR_DATE)
        p_raw = it.get(ATTR_PROCESSED_AT)
        print(
            f"[{i}] date={d_raw!r} -> {coerce_to_date(d_raw)} | "
            f"processed_at={p_raw!r} -> {coerce_to_date(p_raw)} | "
            f"status={it.get(status_attr)!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DynamoDB OCR stats: filter by day / range / explicit dates.",
    )
    date_selection_args(parser)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument(
        "--filter-by",
        choices=(ATTR_DATE, ATTR_PROCESSED_AT),
        default=DEFAULT_FILTER_ATTR,
        help=(
            "Which attribute to match against your selected day(s). "
            "Default: date (calendar day, e.g. 2026-04-02). "
            "Use processed_at to filter by the timestamp field's calendar day."
        ),
    )
    parser.add_argument("--status-attr", default=DEFAULT_STATUS_ATTR)
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Print N scanned rows' raw date/status for debugging empty filters.",
    )
    ns = parser.parse_args()

    target_dates = resolve_target_dates(ns)

    dynamodb = dynamodb_resource(ns.region)
    table = dynamodb.Table(ns.table)

    items, _, _ = fetch_records(table, ns.status_attr)
    if not items and ns.sample == 0:
        print("No records returned. Check table name, region, and credentials.")
        return

    if ns.sample > 0:
        print_sample(items, ns.status_attr, ns.sample)

    filtered = filter_by_dates(items, ns.filter_by, target_dates)
    stats = generate_stats(filtered, ns.status_attr)

    if target_dates:
        print(f"Date filter ({ns.filter_by}): {sorted(target_dates)}")

    print(f"Total  : {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed : {stats['failed']}")

    by_channel = channel_stats(filtered, ns.status_attr)
    if by_channel:
        print("\n--- By channel ---")
        print(f"{'Channel':<24} {'Total':>8} {'Success':>8} {'Failed':>8}")
        for row in by_channel:
            print(
                f"{row['channel']:<24} {row['total']:>8,} "
                f"{row['success']:>8,} {row['failed']:>8,}"
            )


if __name__ == "__main__":
    main()
