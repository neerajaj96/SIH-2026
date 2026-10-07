#!/usr/bin/env python3
"""test_flagship_ux.py — Wave-2 NetraSetu flagship regression (stdlib only).

Pins: XSS hardening (coerced escapeHtml, thumb allowlist, escaped reviewer
grade), crypto ReqID, /screen timeout, FAIL/BORDERLINE queue+render states,
clinical trace, diagnostics panel, 8-stage timeline, a11y (dialog/Esc/tabs
keyboard/dropzone keyboard/aria-live), responsive CSS, queue TTL+clear-all,
TXT export (never fake PDF), preserved LIVE/OFFLINE/DEMO contracts.

Run: python3 tests/test_flagship_ux.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} :: {detail}")


def txt(rel):
    with open(os.path.join(ROOT, rel), errors="ignore") as f:
        return f.read()


def test_xss():
    h = txt("netrasetu.html")
    check("escapeHtml coerces non-strings", "String(s == null" in h, "throws on null")
    check("thumb allowlist", "safeThumbUrl" in h and "data:image/jpeg;base64" in h, "unvalidated src")
    check("queue thumb validated", h.count("safeThumbUrl(c.thumb)") >= 2, "queue XSS")
    check("detail thumb validated", "safeThumbUrl(c.thumb)" in h or "detailThumb" in h, "detail XSS")
    check("thumb escaped on render", 'escapeHtml(thumbUrl)' in h or 'escapeHtml(safeThumbUrl' in h, "attr breakout")
    check("reviewer grade escaped", "escapeHtml(c.reviewerGrade)" in h, "grade injection")
    check("queue id escaped", 'escapeHtml(c.id)' in h, "id injection")


def test_robustness():
    h = txt("netrasetu.html")
    check("crypto ReqID first", "crypto.randomUUID" in h, "Math.random only")
    check("screen timeout", "AbortController" in h and "90000" in h, "hung request")
    check("FAIL queues", "quality failure" in h, "FAIL dropped")
    check("explanation queues", "explanation ' + String(exs)" in h or "explanation \" + " in h or "explanation ' +" in h, "degraded silent")
    check("FAIL render card", "not gradable" in h, "FAIL looks like PASS")
    check("BORDERLINE distinct", "provisional pending human review" in h, "borderline==pass")


def test_trace_diag():
    h = txt("netrasetu.html")
    check("clinical trace", "Clinical reasoning trace" in h, "no trace")
    check("landmarks shown", "Optic disc landmark" in h and "Fovea landmark" in h, "landmark gap")
    check("quadrants shown", "Quadrants:" in h, "quadrant gap")
    check("model state shown", "trained weights present" in h, "model gap")
    check("attention caveat", "not proof of disease" in h, "heatmap overclaim")
    check("diagnostics panel", "renderDiagnostics" in h and "diagBody" in h, "no diagnostics")
    check("key never displayed", "never displayed" in h, "key leak risk")
    check("latency shown", "Latency" in h, "no latency")
    check("TXT export", "exportResultTxt" in h and "netrasetu-result.txt" in h, "no export")
    check("no fake PDF", h.lower().count("pdf") <= 2, "fake PDF risk")


def test_timeline_a11y():
    h = txt("netrasetu.html")
    check("8-stage timeline", h.count('data-i="7"') >= 1 and "Clinical rule" in h and "Disagreement" in h, "4-stage only")
    check("loop covers 8", "PIPELINE_STAGES" in h, "loop drift")
    check("step null-safe", "if(!el)" in h, "null crash")
    check("dialog semantics", 'role="dialog"' in h and 'aria-modal="true"' in h, "no dialog")
    check("Esc closes", "Escape" in h, "no Esc")
    check("tab arrows", "ArrowRight" in h, "no tab keyboard")
    check("queue keyboard", "role=\"button\" tabindex=\"0\"" in h, "mouse-only queue")
    check("dropzone keyboard", "aria-label', 'Upload" in h or 'aria-label", "Upload' in h, "dropzone gap")
    check("aria-live states", h.count('aria-live="polite"') >= 3, "silent updates")
    check("responsive pipeline", "@media (max-width:460px)" in h, "mobile squeeze")
    check("responsive compare", ".compare{grid-template-columns:1fr}" in h, "compare overflow")
    check("reduced motion kept", "prefers-reduced-motion" in h, "motion gap")


def test_queue_demo():
    h = txt("netrasetu.html")
    check("TTL purge", "LOCAL_QUEUE_TTL_MS" in h, "indefinite PHI")
    check("clear-all", "clearQueueBtn" in h, "no purge control")
    check("DEMO banner kept", "Explicit demo data" in h, "demo unlabeled")
    check("OFFLINE no mock", "No mock result shown" in h, "fabrication risk")
    check("LIVE path kept", "fetch(bridgeURL()" in h, "live break")


TESTS = [test_xss, test_robustness, test_trace_diag, test_timeline_a11y, test_queue_demo]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== flagship UX: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
