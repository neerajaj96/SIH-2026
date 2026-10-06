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

# Security configuration (fail-closed; read once at startup).
API_KEY_ENV = "SIH_API_KEY"
CORS_ORIGINS_ENV = "SIH_CORS_ORIGINS"
RATE_LIMIT_ENV = "SIH_RATE_LIMIT_PER_MIN"
TRUSTED_PROXY_ENV = "SIH_TRUSTED_PROXY"
DEFAULT_RATE_LIMIT_PER_MIN = 30
MAX_IMAGE_DIMENSION = 12000  # pixels per side; larger is rejected as abusive/malformed

# Minimal magic signatures (provable-invalidity only - NOT a decodability
# proof; undecodable-but-well-formed files remain decoder-gated).
MAGIC_SIGNATURES = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
)


def detect_image_kind(head):
    """Returns 'jpeg'/'png'/'gif'/'bmp' or None (unrecognized prefix)."""
    for sig, kind in MAGIC_SIGNATURES:
        if bytes(head).startswith(sig):
            return kind
    return None


def probe_jpeg_dimensions(head):
    """Parses JPEG SOF dimensions from a header prefix, or None when the
    prefix does not (yet) contain a parseable frame header. Never raises."""
    try:
        data = bytes(head)
        if not data.startswith(b"\xff\xd8\xff"):
            return None
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                return None
            while i < len(data) and data[i] == 0xFF:
                i += 1
            if i + 1 >= len(data):
                return None
            marker, ln = data[i], (data[i + 1] << 8) + data[i + 2]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                if i + 7 >= len(data):
                    return None
                return (int((data[i + 7] << 8) + data[i + 8]),
                        int((data[i + 5] << 8) + data[i + 6]))
            if ln < 2:
                return None
            i += 1 + ln
        return None
    except (IndexError, ValueError, TypeError):
        return None


def probe_png_dimensions(head):
    """Parses PNG IHDR (width, height) or None. Never raises."""
    try:
        data = bytes(head)
        if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        if data[12:16] != b"IHDR":
            return None
        import struct as _st
        w, h = _st.unpack(">II", data[16:24])
        return (w, h)
    except (ValueError, TypeError):
        return None


def valid_request_id(v):
    """Client X-Request-ID allowlist: 1-64 chars of [A-Za-z0-9._-]."""
    if not isinstance(v, str):
        return False
    if not (1 <= len(v) <= 64):
        return False
    return all(c.isascii() and (c.isalnum() or c in "._-") for c in v)


def new_request_id():
    import uuid as _uuid
    return "r-" + _uuid.uuid4().hex[:16]


SAFE_ERROR_DETAILS = {
    "oversize": "Image exceeds 15 MB limit.",
    "badtype": "Upload must be an image file.",
    "badmagic": "Upload is not a recognized image file.",
    "baddims": "Image dimensions are invalid or exceed limits.",
    "busy": "Screening engine busy; retry shortly.",
    "unauth": "Missing or invalid API key.",
    "unconfigured": "Screening unavailable: bridge API key not configured.",
    "ratelimit": "Rate limit exceeded; retry shortly.",
    "engine": "MATLAB engine unavailable.",
    "failed": "Screening failed.",
}


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
