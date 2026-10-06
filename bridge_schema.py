"""bridge_schema.py — dependency-free API contract for the NetraSetu bridge.

No fastapi / matlab imports here so unit tests can import this anywhere.
The bridge imports it; tests import it; the HTML doc mirrors it.

Schema version is bumped only on incompatible change; additive nullable
fields do NOT bump the major version.
"""

API_VERSION = "1.0-stage7"

# Legacy flat keys: frozen compatibility aliases, always present (null
# when the MATLAB side has no value). Same meaning as the canonical
# nested groups below - never independently computed.
LEGACY_KEYS = (
    "status", "errorMessage", "focus", "entropy", "roiPassed",
    "dl", "conf", "rule", "evidence", "nv", "gradCamOnDisc",
)

# Additive Stage-4/5/6 provenance keys (null when absent on older MATLAB).
ADDITIVE_KEYS = (
    "qualityDecision", "qualityReasons", "qualityGuidance",
    "qualityCalibrated", "ruleStatus", "nvStatus",
    "landmarkStatus", "explanationStatus", "disagreement",
    "temperature", "modelPresent",
)

QUALITY_DECISIONS = ("PASS", "BORDERLINE", "FAIL", "ERROR")
RULE_STATUSES = ("SUFFICIENT", "INSUFFICIENT_EVIDENCE", "PROXY", "INVALID")
NV_STATUSES = ("PROXY_POSITIVE", "NOT_DETECTED", "UNAVAILABLE", "INVALID")
EXPLAIN_STATUSES = ("VALID", "DEGRADED", "UNAVAILABLE", "INVALID")
TEMP_STATES = ("CALIBRATED_VALID", "CALIBRATED_MISMATCH",
               "UNCALIBRATED_FALLBACK", "UNAVAILABLE")
GRADE_RELATIONSHIPS = ("AGREE", "NUMERIC_DISAGREE", "NOT_COMPARABLE")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_CONTENT_PREFIX = "image/"


def to_null_number(v):
    """MATLAB NaN -> None; numbers pass through; anything else -> None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN has no JSON representation


def to_null_str(v):
    if v is None:
        return None
    try:
        return str(v)
    except (TypeError, ValueError):
        return None


def to_str_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    try:
        return [str(x) for x in list(v)]
    except TypeError:
        return []


def to_null_bool(v):
    if v is None:
        return None
    return bool(v)


def check_size_ok(nbytes):
    """True when a request body of nbytes may be retained/processed."""
    try:
        return 0 <= int(nbytes) <= MAX_UPLOAD_BYTES
    except (TypeError, ValueError):
        return False
