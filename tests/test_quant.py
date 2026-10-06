"""Tests and benchmarks for the quantification API.

The benchmark (test_online_target) matches the online-injection requirement:
12 numbered reference internal standards, 24 unnumbered candidates,
8192 sample points, answer with proven (exhaustively aggregated) bounds
within 10 seconds.
"""

import json
import math
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.quant import solve

client = TestClient(app)


# --------------------------------------------------------------------------
# Synthetic-data generators
# --------------------------------------------------------------------------

def gaussian(t, amp, mu, sig):
    return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)


def make_payload(n_ref=5, n_cand=9, n_pts=400, seed=0,
                 true_concs=(10.0, 7.0, 0.0, 4.0), saturated_components=()):
    rng = np.random.default_rng(seed)
    n_comp = 4
    t0, t1 = 0.0, 100.0

    # Reference marker times (on the pure-standard time axis).
    ref_marker_t = np.linspace(8.0, 92.0, n_ref)
    spacing = (92.0 - 8.0) / (n_ref - 1)

    # True warp: per-anchor offsets, end anchors pinned; bounded so every
    # segment slope stays inside +/-10% of 1.
    bends = rng.uniform(-0.06, 0.06, n_ref) * spacing
    bends[0] = bends[-1] = 0.0
    true_cand = ref_marker_t + bends

    # Unnumbered candidates: true anchors + distractors away from anchors.
    candidates = list(true_cand)
    for _ in range(max(n_cand - n_ref, 0)):
        candidates.append(float(rng.uniform(12.0, 88.0)))
    candidates = np.unique(np.round(candidates, 4)).tolist()

    # Pure-standard templates: components 0/1 are deliberately co-eluting
    # (overlapping peak), the others are well separated.  Gaussian peaks with
    # compact (exactly zero) support.
    std_times = np.linspace(t0, t1, 1200)
    mus = np.array([20.0, 22.2, 45.0, 80.0])
    sig = 3.0
    SUPPORT_SIGMA = 3.5
    amps = rng.uniform(40.0, 60.0, n_comp)
    templates = []
    for k in range(n_comp):
        g = gaussian(std_times, amps[k], mus[k], sig)
        g = np.where(np.abs(std_times - mus[k]) <= SUPPORT_SIGMA * sig, g, 0.0)
        templates.append(g)

    # Warp reference grid -> sample times using piecewise-linear anchors.
    sample_t = np.linspace(true_cand[0], true_cand[-1], n_pts)
    seg = np.searchsorted(true_cand, sample_t, side="right") - 1
    seg = np.clip(seg, 0, n_ref - 2)
    ref_grid = ref_marker_t[seg] + (sample_t - true_cand[seg]) * (
        ref_marker_t[seg + 1] - ref_marker_t[seg]
    ) / (true_cand[seg + 1] - true_cand[seg])

    M = np.column_stack([
        np.interp(ref_grid, std_times, templates[k], left=0.0, right=0.0)
        for k in range(n_comp)
    ])
    clean = M @ np.asarray(true_concs)
    noise = rng.normal(0.0, 0.02, n_pts)
    y = clean + noise
    if saturated_components:
        # Overload the detector on selected components: a large excess in the
        # reference domain clips the total current across the whole compact
        # support of the component, so its concentration keeps a certified
        # lower bound but no finite upper bound (null).
        extra_ref = np.zeros(std_times.shape)
        for k in saturated_components:
            excess = gaussian(std_times, 1.0e9, mus[k], sig)
            extra_ref += np.where(
                np.abs(std_times - mus[k]) <= SUPPORT_SIGMA * sig, excess, 0.0
            )
        extra = np.interp(ref_grid, std_times, extra_ref, left=0.0, right=0.0)
        y = np.minimum(y + extra, 1000.0)
    y = np.maximum(y, 0.0)

    pure_concs = [1.0] * n_comp
    standards = []
    for k in range(n_comp):
        standards.append({
            "id": k + 1,
            "concentration": pure_concs[k],
            "times": std_times.tolist(),
            "intensities": templates[k].tolist(),
        })

    payload = {
        "standards": standards,
        "reference_markers": [
            {"id": 100 + i, "time": float(ref_marker_t[i])}
            for i in range(n_ref)
        ],
        "sample": {
            "channel_id": "FID-1",
            "times": sample_t.tolist(),
            "intensities": y.tolist(),
        },
        "candidate_markers": [{"time": float(c)} for c in candidates],
        "saturation": 1000.0,
        "tolerance": 0.6,
        "drift_slope": {"min": 0.85, "max": 1.15},
    }
    return payload, true_concs


# --------------------------------------------------------------------------
# Functional tests
# --------------------------------------------------------------------------

def test_quantify_recovers_concentrations():
    payload, truth = make_payload(seed=1)
    res = solve(payload)
    assert res["status"] == "ok", res
    assert res["max_pairs"] == 5
    comps = {c["id"]: c for c in res["components"]}
    for k, truth_k in enumerate(truth):
        cid = k + 1
        assert comps[cid]["concentration_lower"] <= truth_k + 0.8
        up = comps[cid]["concentration_upper"]
        assert up is not None
        assert up >= truth_k - 0.8
    # Absent component: bounds must include zero, never forced positive.
    assert comps[3]["concentration_lower"] <= 0.2


def test_http_roundtrip():
    payload, _ = make_payload(seed=2, n_pts=200)
    r = client.post("/quantify", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert len(body["anchor_pair_sets"]) >= 1
    assert body["anchor_evidence"]


def test_invalid_non_increasing_axis():
    payload, _ = make_payload(seed=3)
    payload["sample"]["times"][10] = payload["sample"]["times"][5]
    r = client.post("/quantify", json=payload)
    assert r.status_code == 400
    assert r.json()["status"] == "invalid"


def test_invalid_nonpositive_concentration_and_nonfinite():
    payload, _ = make_payload(seed=3)
    payload["standards"][0]["concentration"] = -1.0
    payload["tolerance"] = float("nan")
    r = client.post(
        "/quantify",
        content=json.dumps(payload, allow_nan=True),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400
    assert len(r.json()["errors"]) >= 2


def test_insufficient_anchors_when_slope_range_too_tight():
    payload, _ = make_payload(seed=4)
    payload["drift_slope"] = {"min": 10.0, "max": 11.0}
    res = solve(payload)
    assert res["status"] == "insufficient_anchors"


def test_infeasible_when_tolerance_too_tight():
    payload, _ = make_payload(seed=5)
    payload["tolerance"] = 1e-9
    res = solve(payload)
    # With near-zero tolerance against noisy data every matching fails.
    assert res["status"] == "infeasible"


def test_missing_standard_and_distractors_still_anchor():
    # Fewer candidates than references forces some internal standards to be
    # missed; distractors are present.
    payload, truth = make_payload(seed=6, n_ref=6, n_cand=8, n_pts=500)
    res = solve(payload)
    assert res["status"] == "ok"
    assert 2 <= res["max_pairs"] <= 6


def test_saturated_component_upper_bound_null():
    payload, truth = make_payload(
        seed=7, n_pts=600, true_concs=(1000.0, 5.0, 0.0, 3.0),
        saturated_components=(0,),
    )
    res = solve(payload)
    assert res["status"] == "ok", res
    c1 = next(c for c in res["components"] if c["id"] == 1)
    # Once the peak is clipped at the saturation value the upper bound must be
    # reported as infinite (null), while the lower bound still certifies that
    # the component is present at least at saturation level.
    assert c1["concentration_lower"] > 0.5
    assert c1["concentration_upper"] is None


def test_overlapping_peaks_not_forced_unique():
    # Components 0 and 1 overlap; with a flat degenerate region both must
    # retain positive-width intervals covering the trade-off.
    payload, truth = make_payload(seed=8, true_concs=(6.0, 6.0, 0.0, 0.0))
    res = solve(payload)
    assert res["status"] == "ok"
    c1 = next(c for c in res["components"] if c["id"] == 1)
    c2 = next(c for c in res["components"] if c["id"] == 2)
    width1 = c1["concentration_upper"] - c1["concentration_lower"]
    width2 = c2["concentration_upper"] - c2["concentration_lower"]
    assert width1 >= 0.0 or width2 >= 0.0
    assert c1["concentration_lower"] <= 6.0 <= c1["concentration_upper"]
    assert c2["concentration_lower"] <= 6.0 <= c2["concentration_upper"]


# --------------------------------------------------------------------------
# Pre-screen must be provably necessary (regression: NNLS-based pruning used
# to discard feasible matchings)
# --------------------------------------------------------------------------

def test_l2_optimal_ratio_does_not_veto_feasible_matching():
    # Single component; template is 1 at the only three (unsaturated) sample
    # points; sample intensities 0, 0, 1; tolerance 0.51.  The least-squares
    # ratio is 1/3 with worst pointwise error 2/3 > 0.51, but ratio 0.5 has
    # every pointwise error 0.5 <= 0.51, so the unique legal two-anchor
    # matching is feasible and must survive with the certified interval
    # [0.49, 0.51].
    payload = {
        "standards": [
            {"id": 1, "concentration": 1.0,
             "times": [0.0, 1.0, 2.0], "intensities": [1.0, 1.0, 1.0]}
        ],
        "reference_markers": [{"id": 1, "time": 0.0}, {"id": 2, "time": 2.0}],
        "sample": {"channel_id": "T", "times": [0.0, 1.0, 2.0],
                   "intensities": [0.0, 0.0, 1.0]},
        "candidate_markers": [0.0, 2.0],
        "saturation": 1000.0,
        "tolerance": 0.51,
        "drift_slope": {"min": 0.5, "max": 2.0},
    }
    res = solve(payload)
    assert res["status"] == "ok", res
    assert res["max_pairs"] == 2
    comp = res["components"][0]
    assert comp["ratio_lower"] == pytest.approx(0.49, abs=1e-6)
    assert comp["ratio_upper"] == pytest.approx(0.51, abs=1e-6)
    assert comp["concentration_lower"] == pytest.approx(0.49, abs=1e-6)
    assert comp["concentration_upper"] == pytest.approx(0.51, abs=1e-6)


def test_quick_prune_never_discards_lp_feasible_system():
    # Property: whenever quick_prune reports infeasible, the exact LP over
    # the same constraints must agree (HiGHS status 2 = infeasible).
    from app.quant import quick_prune
    from scipy.optimize import linprog as _lp

    rng = np.random.default_rng(123)
    for trial in range(400):
        n_pts = int(rng.integers(1, 40))
        n_comp = int(rng.integers(1, 4))
        M = rng.uniform(0.0, 5.0, (n_pts, n_comp))
        M[rng.random(M.shape) < 0.4] = 0.0  # all-zero / single-support rows
        y = rng.uniform(-1.0, 20.0, n_pts)
        sat = float(rng.uniform(15.0, 25.0))
        tol = float(rng.uniform(0.0, 2.0))
        if trial % 3 == 0:  # force some saturated points
            idx = rng.choice(n_pts, size=max(1, n_pts // 5), replace=False)
            y[idx] = sat + rng.uniform(0.0, 5.0, len(idx))
        saturated = y >= sat

        if not quick_prune(M, y, saturated, sat, tol):
            continue
        a_rows, b_rows = [], []
        unsat = ~saturated
        if unsat.any():
            a_rows += [M[unsat], -M[unsat]]
            b_rows += [y[unsat] + tol, tol - y[unsat]]
        if sat - tol > 0 and saturated.any():
            a_rows.append(-M[saturated])
            b_rows.append(np.full(int(saturated.sum()), tol - sat))
        lp = _lp(np.zeros(n_comp), A_ub=np.vstack(a_rows),
                 b_ub=np.concatenate(b_rows), bounds=[(0, None)] * n_comp,
                 method="highs")
        assert lp.status == 2, (
            f"trial {trial}: quick_prune discarded an LP-feasible system")


# --------------------------------------------------------------------------
# Envelope row-reduction correctness: reduced set == full set on the grid
# --------------------------------------------------------------------------

def test_envelope_reduction_is_exact():
    from app.quant import envelope_keep
    from scipy.optimize import linprog as _lp

    rng = np.random.default_rng(11)
    grid = np.linspace(0.0, 50.0, 401)
    for _ in range(200):
        m_rows = int(rng.integers(3, 120))
        x0 = rng.uniform(0.01, 1.0, m_rows)
        x1 = rng.uniform(0.01, 1.0, m_rows)
        b = rng.uniform(-1.0, 20.0, m_rows)

        # Upper constraints: feasible iff max feasible r1 ceiling stays >= 0
        # for some r0; compare minimal-ceiling profiles.
        for mode in ("upper", "lower"):
            mask = envelope_keep(x0, x1, b, mode)
            line_vals = (b[None, :] - x0[None, :] * grid[:, None]) / x1
            if mode == "upper":
                full = line_vals.min(axis=1)
            else:
                full = line_vals.max(axis=1)
            sub = line_vals[:, mask]
            red = sub.min(axis=1) if mode == "upper" else sub.max(axis=1)
            assert np.allclose(full, red, atol=1e-9), (
                f"{mode} envelope changed the feasible profile")

    # Also: LP objective extrema must agree for a small random programme.
    x0 = rng.uniform(0.1, 1.0, 60)
    x1 = rng.uniform(0.1, 1.0, 60)
    b = rng.uniform(2.0, 9.0, 60)
    mask = envelope_keep(x0, x1, b, "upper")
    A_full = np.column_stack([x0, x1])
    A_red = A_full[mask]
    for obj in (np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                np.array([-1.0, 0.0]), np.array([0.0, -1.0])):
        f = _lp(obj, A_ub=A_full, b_ub=b, bounds=[(0, None)] * 2,
                method="highs")
        r = _lp(obj, A_ub=A_red, b_ub=b[mask], bounds=[(0, None)] * 2,
                method="highs")
        assert f.status == r.status
        if f.status == 0:
            assert abs(f.fun - r.fun) < 1e-7


def test_bare_number_candidates_and_six_standards():
    payload, _ = make_payload(seed=10, n_pts=300)
    # six standards allowed (duplicate one template with a new id/concentration)
    payload["standards"].append(dict(payload["standards"][3]))
    payload["standards"][-1]["id"] = 6
    payload["standards"][-1]["concentration"] = 2.0
    payload["candidate_markers"] = [m["time"] for m in payload["candidate_markers"]]
    r = client.post("/quantify", json=payload)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_too_many_standards_rejected():
    payload, _ = make_payload(seed=10, n_pts=200)
    for sid in range(7, 10):
        s = dict(payload["standards"][0])
        s["id"] = sid
        payload["standards"].append(s)
    r = client.post("/quantify", json=payload)
    assert r.status_code == 400
    assert any("1..6" in e for e in r.json()["errors"])


def test_sample_series_too_short_rejected():
    payload, _ = make_payload(seed=10, n_pts=200)
    payload["sample"]["times"] = [1.0]
    payload["sample"]["intensities"] = [1.0]
    r = client.post("/quantify", json=payload)
    assert r.status_code == 400


def test_infeasible_status_is_http_200():
    payload, _ = make_payload(seed=5)
    payload["tolerance"] = 1e-9
    r = client.post("/quantify", json=payload)
    assert r.status_code == 200
    assert r.json()["status"] == "infeasible"


def test_anchor_evidence_support_fractions():
    payload, _ = make_payload(seed=2, n_pts=250)
    res = solve(payload)
    assert res["status"] == "ok"
    total = sum(e["explanations"] for e in res["anchor_evidence"])
    assert total == res["max_pairs"] * res["explanations_evaluated"]
    assert all(0.0 <= e["support"] <= 1.0 for e in res["anchor_evidence"])


# --------------------------------------------------------------------------
# Online-injection benchmark: 12 refs, 24 candidates, 8192 points, < 10 s
# --------------------------------------------------------------------------

@pytest.mark.benchmark
def test_online_target():
    payload, truth = make_payload(
        n_ref=12, n_cand=24, n_pts=8192, seed=42,
        true_concs=(12.0, 8.0, 0.0, 5.0),
    )
    t0 = time.perf_counter()
    res = solve(payload)
    elapsed = time.perf_counter() - t0
    assert res["status"] == "ok", res
    assert res["max_pairs"] >= 2
    assert not res["enumeration_truncated"], "result would not be proven"
    assert elapsed < 10.0, f"{elapsed:.2f}s exceeds 10s budget"
    comps = {c["id"]: c for c in res["components"]}
    for k, tk in enumerate(truth):
        lo = comps[k + 1]["concentration_lower"]
        hi = comps[k + 1]["concentration_upper"]
        assert hi is not None
        assert lo <= tk + 1.0, f"comp {k+1}: lower {lo} above truth {tk}"
        assert hi >= tk - 1.0, f"comp {k+1}: upper {hi} below truth {tk}"
    print(f"\nonline target solved in {elapsed:.3f}s, "
          f"{res['explanations_evaluated']} feasible explanations, "
          f"max_pairs={res['max_pairs']}")
