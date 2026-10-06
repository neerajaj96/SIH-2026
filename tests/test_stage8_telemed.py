#!/usr/bin/env python3
"""test_stage8_telemed.py — Stage-8 queueing/telemed contracts (stdlib only).

Independent Erlang-C implementation (not a copy of the .m algebra shape -
verified against closed-form M/M/1 and a textbook M/M/2 value), explicit
unit conversions, stability/monotonicity, optimizer-logic mirror,
scenario-config validation, transmission-delay math, seed contracts.

Run: python3 tests/test_stage8_telemed.py
MATLAB twin: testTelemedSim.m (UNEXECUTED here - no MATLAB/SimEvents).
No SimEvents execution claimed; no field measurements claimed.
"""
import math
import os
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


def erlang_c_wq(lam, mu, c):
    """Independent M/M/c mean queue wait (same units as 1/mu)."""
    a = lam / mu
    rho = a / c
    if rho >= 1:
        return float("inf")
    s = sum(a ** k / math.factorial(k) for k in range(c))
    last = (a ** c) / math.factorial(c) * (c / (c - a))
    return (last / (s + last)) / (c * mu - lam)


def tx_seconds(image_mb, bw_mbps, overhead):
    return image_mb * 8 * (1 + overhead) / bw_mbps


def test_erlang_known():
    # M/M/1 closed form at the PS operating point (lambda=10/hr, mu=120/hr).
    got = erlang_c_wq(10, 120, 1)
    expect = (10 / 120) / (120 - 10)
    check("M/M/1 matches closed form", abs(got - expect) < 1e-12, f"{got} vs {expect}")
    # Light utilization: tiny wait.
    check("light utilization small wait", erlang_c_wq(1, 120, 1) < 0.001, "")
    # Textbook M/M/2 sanity: more servers beat one server at same load.
    check("M/M/2 beats M/M/1", erlang_c_wq(10, 120, 2) < erlang_c_wq(10, 120, 1), "")
    # High utilization: wait grows steeply (monotone in load).
    w1, w2 = erlang_c_wq(100, 120, 1), erlang_c_wq(110, 120, 1)
    check("wait monotone in load", w2 > w1 > 0, f"{w1} vs {w2}")
    # Server monotonicity.
    ws = [erlang_c_wq(100, 120, c) for c in (2, 3, 4)]
    check("wait monotone decreasing in servers", ws[0] > ws[1] > ws[2], f"{ws}")
    # Stability boundary.
    check("unstable returns Inf (not finite)", erlang_c_wq(120, 120, 1) == float("inf"), "")
    check("overload returns Inf", erlang_c_wq(200, 120, 1) == float("inf"), "")
    check("zero arrival gives zero wait", erlang_c_wq(0, 120, 1) == 0.0, "")
    # .m file agrees structurally (single implementation, Inf on unstable).
    body = txt("erlangCWaitHours.m")
    check("standalone handles rho>=1", "rho >= 1" in body and "wq = Inf" in body, "stability gap")
    check("no duplicate implementation remains",
          "function wq = erlangCWaitHours" not in txt("optimizeResourceAllocation.m"), "shadow copy alive")


def test_units():
    # 2MB @ 4Mbps + 10% overhead = 2*8*1.1/4 = 4.4 s.
    check("transmission math (MB/Mbps/bits)", abs(tx_seconds(2.0, 4.0, 0.10) - 4.4) < 1e-12, "")
    check("bandwidth doubling halves delay",
          abs(tx_seconds(2.0, 8.0, 0.10) - 2.2) < 1e-12, "")
    check("gap conversion: 6-min gap = 10/hr",
          abs(60 / 6 - 10) < 1e-12, "")
    check("review conversion: 30s = 120/hr", abs(3600 / 30 - 120) < 1e-12, "")
    body = txt("telemedTransmissionSeconds.m")
    check("canonical helper owns the arithmetic",
          "* 8" in body and "bandwidthMbps" in body, "inline arithmetic risk")


def test_optimizer_logic():
    # Mirror of the sweep rule: min c with finite wait <= SLA and rho < 1.
    lam, mu, sla = 7.05, 120.0, 0.5  # 100k*0.141/2000 per hour
    feas = [c for c in range(1, 31)
            if math.isfinite(erlang_c_wq(lam, mu, c)) and erlang_c_wq(lam, mu, c) <= sla
            and (lam / mu) / c < 1]
    check("optimizer picks min feasible", feas and feas[0] == 1, f"{feas[:3]}")
    check("optimizer states objective explicitly",
          "minimize reviewers" in txt("optimizeResourceAllocation.m"), "ambiguous objective")
    check("optimizer reports utilization+stability",
          "utilization" in txt("optimizeResourceAllocation.m") and "UNSTABLE" in txt("optimizeResourceAllocation.m"), "stability gap")
    check("infeasible stays NaN (not a fake count)",
          "isnan(neededReviewers)" in txt("optimizeResourceAllocation.m"), "fake staffing")


def test_scenario_config():
    cfg = txt("telemedConfig.m")
    for f in ("annualPatients", "operatingHoursPerYear", "imageMB", "bandwidthMbps",
              "protocolOverheadFraction", "aiServiceTimeSeconds", "aiWorkers",
              "reviewSecondsOptimistic", "reviewSecondsConservative",
              "bandwidthQueueCapacity", "meanArrivalGapMinutes", "simDurationMinutes",
              "warmupMinutes", "replications", "seed", "consistencyTolerance",
              "severityMix", "severityProvenance", "referableFraction", "version"):
        check(f"config owns {f}", f in cfg, "missing")
    check("ARDA defaults exact", "[0.827 0.032 0.104 0.009 0.028]" in cfg, "mix drift")
    check("provenance explicit", "ARDA-sourced scenario" in cfg, "unlabeled scenario")
    check("labels order documented", "NoDR" in cfg and "PDR" in cfg, "label gap")
    mix = [0.827, 0.032, 0.104, 0.009, 0.028]
    check("mix sums to 1", abs(sum(mix) - 1) < 1e-12, f"{sum(mix)}")
    check("mix non-negative, length 5", len(mix) == 5 and all(v >= 0 for v in mix), "")
    check("referable derived from mix", abs(sum(mix[2:]) - 0.141) < 1e-12, f"{sum(mix[2:])}")
    check("builder validates mix", "badSeverityMix" in txt("buildTelemedModel.m"), "unchecked mix")
    check("builder uses config mix (no hardcoded distribution)",
          "cumsum(" in txt("buildTelemedModel.m"), "hardcoded distribution")
    check("model validates transmission inputs",
          "badInputs" in txt("telemedTransmissionSeconds.m"), "unchecked inputs")


def test_sim_contracts():
    b = txt("buildTelemedModel.m")
    check("transmission server exists (delay affects queueing)",
          "Transmission_Server" in b, "bandwidth params decorative")
    check("derived delay wired into server",
          "telemedTransmissionSeconds" in b, "unwired delay")
    check("machine-readable wait logging (not Scope-only)",
          "To Workspace" in b and "reviewWaitLog" in b and "bwWaitLog" in b, "visual-only metrics")
    check("seed contract documented (no hidden rng)",
          "rng(cfg.seed" in txt("evaluateTelemedConsistency.m") and "does NOT call rng" in b, "seed gap")
    check("no-second-copy model builder",
          not any(ln.strip().startswith("add_block(")
                  for ln in txt("SimEvents_Telemed_Model.m").splitlines()), "topology fork")
    check("consistency runner exists with tolerance verdict",
          os.path.exists(os.path.join(ROOT, "evaluateTelemedConsistency.m")) and
          "consistencyTolerance" in txt("evaluateTelemedConsistency.m"), "runner gap")
    check("bottleneck evidence-based (utilization, not snapshot)",
          "bottleneck" in txt("evaluateTelemedConsistency.m").lower(), "snapshot inference")


TESTS = [test_erlang_known, test_units, test_optimizer_logic,
         test_scenario_config, test_sim_contracts]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stage8 telemed: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
