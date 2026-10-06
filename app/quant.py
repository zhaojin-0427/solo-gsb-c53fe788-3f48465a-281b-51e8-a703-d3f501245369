"""Chromatographic quantification core.

Pipeline
--------
1. Validate the payload (finite numbers, strictly increasing time axes, ...).
2. Build a DAG of possible anchor pairs (numbered reference internal-standard
   times -> unnumbered candidate times on the sample channel).  An arc between
   two pairs exists when the reference points are order preserving, neither
   reference nor candidate is reused, and the segment slope

       ds/dr = (sample_time_2 - sample_time_1) / (ref_time_2 - ref_time_1)

   lies within the allowed drift-slope range.
3. Longest-path DP gives the maximum achievable pair count.  Matchings of
   that length are enumerated (lengths are tried in descending order so that
   the *feasible* pair count is maximised).
4. For each matching, a common piecewise-linear time warp maps every pure
   standard template onto the sample grid.  Sample component ratios
   r = c_sample / c_pure are found with linear programmes:
       * unsaturated points: |M r - y| <= tolerance
       * saturated points:    M r >= saturation - tolerance
         (only a lower bound is required once the detector saturates)
       * r >= 0
5. Across every feasible, pair-maximal explanation we aggregate:
       * component concentration lower/upper bounds (null = +infinity),
       * anchor evidence (how often each pair appears).
   Overlapping peaks are never assigned uniquely - the LP polytope keeps the
   full trade-off as concentration intervals.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import nnls

# Matching enumeration depth is bounded by the (potentially large) number of
# reference markers; keep a comfortable recursion budget.
sys.setrecursionlimit(20_000)

# Guard rails for the exhaustive enumeration (the 8192-point online target is
# reached well below these limits; they only protect against pathological
# requests with millions of combinatorial matchings).
_ENUM_DEADLINE_S = 9.0
_ENUM_PATH_CAP = 500_000


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(
        v, bool
    )


def _finite(v: Any) -> bool:
    return _is_num(v) and math.isfinite(float(v))


def _extract_series(obj: Any, errors: list[str], label: str):
    if not isinstance(obj, dict):
        errors.append(f"{label} must be an object with 'times' and 'intensities'")
        return None
    t_raw, y_raw = obj.get("times"), obj.get("intensities")
    if not isinstance(t_raw, list) or not isinstance(y_raw, list):
        errors.append(f"{label}.times and {label}.intensities must be arrays")
        return None
    if len(t_raw) != len(y_raw):
        errors.append(f"{label}: times/intensities length mismatch")
        return None
    if len(t_raw) < 2:
        errors.append(f"{label}: at least two points are required")
        return None
    t = np.empty(len(t_raw))
    for i, v in enumerate(t_raw):
        if not _finite(v):
            errors.append(f"{label}.times[{i}] is not a finite number")
        else:
            t[i] = float(v)
    y = np.empty(len(y_raw))
    for i, v in enumerate(y_raw):
        if not _finite(v):
            errors.append(f"{label}.intensities[{i}] is not a finite number")
        else:
            y[i] = float(v)
    if errors:
        return None
    if np.any(np.diff(t) <= 0):
        errors.append(f"{label}.times must be strictly increasing")
        return None
    return t, y


def validate_and_extract(payload: Any) -> tuple[dict | None, list[str]]:
    """Return (typed inputs, errors)."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return None, ["request body must be a JSON object"]

    raw_std = payload.get("standards")
    if not isinstance(raw_std, list) or not (1 <= len(raw_std) <= 6):
        errors.append("standards must be an array of 1..6 pure standards")
        raw_std = []
    standards = []
    used_ids: set[int] = set()
    for k, st in enumerate(raw_std):
        label = f"standards[{k}]"
        if not isinstance(st, dict):
            errors.append(f"{label} must be an object")
            continue
        sid = st.get("id", k + 1)
        if not _is_num(sid) or int(sid) != sid:
            errors.append(f"{label}.id must be an integer")
            continue
        sid = int(sid)
        if sid in used_ids:
            errors.append(f"{label}.id {sid} is duplicated")
            continue
        used_ids.add(sid)
        c = st.get("concentration")
        if not _finite(c) or float(c) <= 0:
            errors.append(f"{label}.concentration must be a finite positive number")
            continue
        series = _extract_series(st, errors, label)
        if series is None:
            continue
        standards.append({"id": sid, "concentration": float(c),
                          "times": series[0], "intensities": series[1]})

    raw_ref = payload.get("reference_markers")
    if not isinstance(raw_ref, list) or not raw_ref:
        errors.append("reference_markers must be a non-empty array")
        raw_ref = []
    ref_ids: list[int] = []
    ref_times: list[float] = []
    seen_ref_ids: set[int] = set()
    prev_t: float | None = None
    for i, m in enumerate(raw_ref):
        label = f"reference_markers[{i}]"
        if not isinstance(m, dict):
            errors.append(f"{label} must be an object")
            continue
        mid, mt = m.get("id"), m.get("time")
        if not _is_num(mid) or int(mid) != mid:
            errors.append(f"{label}.id must be an integer")
            continue
        mid = int(mid)
        if mid in seen_ref_ids:
            errors.append(f"{label}.id {mid} is duplicated")
            continue
        if not _finite(mt):
            errors.append(f"{label}.time must be a finite number")
            continue
        if prev_t is not None and float(mt) <= prev_t:
            errors.append("reference_markers times must be strictly increasing")
            continue
        seen_ref_ids.add(mid)
        prev_t = float(mt)
        ref_ids.append(mid)
        ref_times.append(float(mt))

    sample_series = _extract_series(payload.get("sample"), errors, "sample")
    if sample_series is not None:
        sample_t, sample_y = sample_series
        sample_channel = str(payload["sample"].get("channel_id", "quant"))
    else:
        sample_t = sample_y = np.empty(0)
        sample_channel = "quant"

    raw_cand = payload.get("candidate_markers")
    if not isinstance(raw_cand, list) or not raw_cand:
        errors.append("candidate_markers must be a non-empty array")
        raw_cand = []
    cand_times: list[float] = []
    for i, m in enumerate(raw_cand):
        label = f"candidate_markers[{i}]"
        if _finite(m):
            cand_times.append(float(m))
        elif isinstance(m, dict) and _finite(m.get("time")):
            cand_times.append(float(m["time"]))
        else:
            errors.append(f"{label} must carry a finite 'time'")
    # Candidates are unnumbered; their physical order matters, so sort+de-dup.
    cand_times = sorted(set(cand_times))

    sat = payload.get("saturation")
    if not _finite(sat) or float(sat) <= 0:
        errors.append("saturation must be a finite positive number")
        sat = math.nan
    tol = payload.get("tolerance")
    if not _finite(tol) or float(tol) < 0:
        errors.append("tolerance must be a finite non-negative number")
        tol = math.nan
    slope = payload.get("drift_slope")
    if not isinstance(slope, dict) or not _finite(slope.get("min")) or not _finite(
        slope.get("max")
    ):
        errors.append("drift_slope must be {'min': finite, 'max': finite}")
        slope_min = slope_max = math.nan
    else:
        slope_min, slope_max = float(slope["min"]), float(slope["max"])
        if slope_min > slope_max:
            errors.append("drift_slope.min must be <= drift_slope.max")
        if slope_min <= 0:
            errors.append("drift_slope.min must be positive")

    if errors:
        return None, errors

    return {
        "standards": standards,
        "ref_ids": np.asarray(ref_ids, dtype=np.int64),
        "ref_times": np.asarray(ref_times, dtype=np.float64),
        "sample_channel": sample_channel,
        "sample_t": sample_t,
        "sample_y": sample_y,
        "cand_times": np.asarray(cand_times, dtype=np.float64),
        "saturation": float(sat),
        "tolerance": float(tol),
        "slope_min": slope_min,
        "slope_max": slope_max,
    }, []


def envelope_keep(x0: np.ndarray, x1: np.ndarray, b: np.ndarray,
                  mode: str) -> np.ndarray:
    """Co-eluting pair constraints on r0, r1 >= 0.

    Constraint row i:  x0_i r0 + x1_i r1 ~ b_i.
    mode 'upper' (<=): keep rows whose line is on the lower envelope
    (minimum achievable r1 at some r0 >= 0); the others are dominated.
    mode 'lower' (>=): keep the upper envelope.
    Returns a boolean mask over the input rows.
    """
    m, c = -x0 / x1, b / x1
    order = np.argsort(m, kind="stable")
    m, c = m[order], c[order]
    uniq: list[int] = []
    for i in range(len(m)):
        if uniq and m[i] == m[uniq[-1]]:
            better = (c[i] < c[uniq[-1]]) if mode == "upper" else (
                c[i] > c[uniq[-1]])
            if better:
                uniq[-1] = i
        else:
            uniq.append(i)
    ua = np.array(uniq)
    m, c = m[ua], c[ua]
    hull: list[int] = []
    for i in range(len(m)):
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            x_1i = (c[i1] - c[i]) / (m[i] - m[i1])
            x_12 = (c[i1] - c[i2]) / (m[i2] - m[i1])
            # Lower envelope: pop while the new intersection overtakes from
            # the left.  Upper envelope mirrors the inequality.
            if (x_1i <= x_12) if mode == "lower" else (x_1i >= x_12):
                hull.pop()
            else:
                break
        hull.append(i)
    keep = np.zeros(len(order), dtype=bool)
    keep[order[ua[np.array(hull)]]] = True
    return keep


def solve(payload: Any) -> dict:
    started = time.perf_counter()
    data, errors = validate_and_extract(payload)
    if errors:
        return {"status": "invalid", "errors": errors}

    ref_t = data["ref_times"]
    ref_ids = data["ref_ids"]
    cand_t = data["cand_times"]
    s_min, s_max = data["slope_min"], data["slope_max"]
    n_ref, n_cand = len(ref_t), len(cand_t)

    # ------------------------------------------------------------------ #
    # DAG of anchor pairs.  Every single pair is a valid node (a pair by
    # itself has no segment slope); arcs carry the slope constraint.
    # ------------------------------------------------------------------ #
    succ: dict[tuple[int, int], list[tuple[int, int]]] = {}
    nodes: list[tuple[int, int]] = []
    for i in range(n_ref):
        for j in range(n_cand):
            nodes.append((i, j))
            outs: list[tuple[int, int]] = []
            for i2 in range(i + 1, n_ref):
                dt = ref_t[i2] - ref_t[i]
                for j2 in range(j + 1, n_cand):
                    s = (cand_t[j2] - cand_t[j]) / dt
                    if s_min <= s <= s_max:
                        outs.append((i2, j2))
            succ[(i, j)] = outs

    # Longest-path DP: best[node] = longest matching length starting at node.
    best = {node: 1 for node in nodes}
    for i in range(n_ref - 1, -1, -1):
        for j in range(n_cand - 1, -1, -1):
            b = 1
            for s in succ[(i, j)]:
                v = best[s] + 1
                if v > b:
                    b = v
            best[(i, j)] = b
    longest = max(best.values()) if best else 0

    if longest < 2:
        return {"status": "insufficient_anchors",
                "reason": "fewer than two order-preserving anchor pairs with "
                          "admissible drift slopes",
                "longest_valid_matching": int(longest)}

    # Exact-length path counts (small 3-D DP) used to report level sizes.
    cnt = np.zeros((longest + 1, n_ref, n_cand), dtype=np.float64)
    for (i, j) in nodes:
        cnt[1, i, j] = 1.0
    for r in range(2, longest + 1):
        for i in range(n_ref - 1, -1, -1):
            for j in range(n_cand - 1, -1, -1):
                cnt[r, i, j] = sum(
                    cnt[r - 1, i2, j2] for (i2, j2) in succ[(i, j)])

    # ------------------------------------------------------------------ #
    # Per-matching feasibility + ratio bounds.
    # ------------------------------------------------------------------ #
    std = data["standards"]
    n_comp = len(std)
    std_t = [s["times"] for s in std]
    std_y = [s["intensities"] for s in std]
    pure_c = np.array([s["concentration"] for s in std], dtype=np.float64)
    std_ids = [s["id"] for s in std]

    sample_t = data["sample_t"]
    sample_y = data["sample_y"]
    n_pts = len(sample_t)
    sat = data["saturation"]
    tol = data["tolerance"]
    saturated = sample_y >= sat
    deadline = started + _ENUM_DEADLINE_S

    def build_matrix(path: list[tuple[int, int]]) -> np.ndarray:
        """Warp every template onto the sample grid for one matching."""
        ri = np.array([p[0] for p in path])
        ci = np.array([p[1] for p in path])
        rseg, cseg = ref_t[ri], cand_t[ci]
        t0, t1 = cseg[0], cseg[-1]

        seg = np.searchsorted(cseg, sample_t, side="right") - 1
        seg = np.clip(seg, 0, len(cseg) - 2)
        ref_grid = rseg[seg] + (sample_t - cseg[seg]) * (
            rseg[seg + 1] - rseg[seg]
        ) / (cseg[seg + 1] - cseg[seg])
        inside = (sample_t >= t0) & (sample_t <= t1)

        M = np.zeros((n_pts, n_comp))
        for k in range(n_comp):
            vals = np.interp(ref_grid, std_t[k], std_y[k], left=0.0, right=0.0)
            M[:, k] = np.where(inside, vals, 0.0)
        return M

    def quick_prune(M: np.ndarray) -> bool:
        """Cheap *provably necessary* conditions before the exact LP.

        True  => the matching is provably infeasible (safe to discard).
        False => undecided here; the exact LP must arbitrate.

        Two sound criteria are used:

        * A saturated row whose template entries are all <= 0 can never
          reach ``saturation - tolerance`` (when positive) with r >= 0.
        * NNLS minimises ||Mu r - yu||_2 over r >= 0.  Any pointwise
          feasible r (max |Mu r - yu| <= tol) would achieve an L2 norm
          of at most sqrt(n) * tol, so an NNLS optimum above that bound
          proves infeasibility.

        The max pointwise residual of the least-squares fit itself
        proves nothing: NNLS trades a few large deviations for many
        small ones, so its fit can violate the tolerance while another
        ratio vector (the Chebyshev one) satisfies every row.  That
        statistic must not be used to reject a matching - uncertain
        matchings go to the exact LP below.
        """
        if sat - tol > 0 and saturated.any():
            Ms = M[saturated]
            if Ms.shape[0] and np.any(Ms.max(axis=1) <= 0.0):
                return True
        unsat = ~saturated
        if unsat.any():
            Mu, yu = M[unsat], sample_y[unsat]
            scale = max(float(np.linalg.norm(yu)), 1.0)
            r_nnls, _ = nnls(Mu / scale, yu / scale, maxiter=200)
            resid_l2 = float(np.linalg.norm(Mu @ r_nnls - yu))
            l2_bound = math.sqrt(len(yu)) * tol
            # Margin: NNLS may stop marginally above the true optimum;
            # only a clear overshoot is a proof, borderline cases fall
            # through to the LP.
            if resid_l2 > l2_bound * (1.0 + 1e-9) + 1e-9:
                return True
        return False

    def grouped_rows(A: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
        groups: dict[tuple[int, ...], list[int]] = {}
        nz = A > 0
        for row_i in range(A.shape[0]):
            key = tuple(np.nonzero(nz[row_i])[0].tolist())
            groups.setdefault(key, []).append(row_i)
        return {k: np.asarray(v) for k, v in groups.items()}

    def build_lp(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reduced inequality set A r <= b with identical feasible polytope.

        Rows are grouped by non-zero support.  Within a group only tightest
        rows can bind: 1-D support -> one row; 2-D support (a co-eluting
        pair) -> envelope lines; >= 3-D support -> keep all.
        """
        A_blocks: list[np.ndarray] = []
        b_blocks: list[np.ndarray] = []

        unsat_idx = np.nonzero(~saturated)[0]
        if len(unsat_idx):
            Mu, yu = M[unsat_idx], sample_y[unsat_idx]
            up: list[int] = []
            lo: list[int] = []
            for key, idx in grouped_rows(Mu).items():
                if key == ():
                    # Rows of zeros: upper is vacuous; lower can be infeasible.
                    j = int(idx[np.argmin(yu[idx])])
                    lo.append(j)
                    continue
                if len(key) == 1:
                    col = key[0]
                    up.append(int(idx[np.argmin((yu[idx] + tol) / Mu[idx, col])]))
                    lo.append(int(idx[np.argmax((yu[idx] - tol) / Mu[idx, col])]))
                elif len(key) == 2:
                    c0, c1 = key
                    up.extend(idx[envelope_keep(
                        Mu[idx, c0], Mu[idx, c1], yu[idx] + tol, "upper")].tolist())
                    lo.extend(idx[envelope_keep(
                        Mu[idx, c0], Mu[idx, c1], yu[idx] - tol, "lower")].tolist())
                else:
                    up.extend(idx.tolist())
                    lo.extend(idx.tolist())
            up, lo = sorted(set(up)), sorted(set(lo))
            A_blocks += [Mu[up], -Mu[lo]]
            b_blocks += [yu[up] + tol, tol - yu[lo]]

        sat_idx = np.nonzero(saturated)[0]
        if sat - tol > 0 and len(sat_idx):
            Ms = M[sat_idx]
            keep: list[int] = []
            for key, idx in grouped_rows(Ms).items():
                if key == ():
                    keep.append(int(idx[0]))
                    continue
                if len(key) == 1:
                    col = key[0]
                    keep.append(int(idx[np.argmax((sat - tol) / Ms[idx, col])]))
                elif len(key) == 2:
                    c0, c1 = key
                    keep.extend(idx[envelope_keep(
                        Ms[idx, c0], Ms[idx, c1],
                        np.full(len(idx), sat - tol), "lower")].tolist())
                else:
                    keep.extend(idx.tolist())
            keep = sorted(set(keep))
            A_blocks.append(-Ms[keep])
            b_blocks.append(np.full(len(keep), tol - sat))

        return np.vstack(A_blocks), np.concatenate(b_blocks)

    def exact_bounds(A_ub: np.ndarray, b_ub: np.ndarray):
        bounds = [(0.0, None)] * n_comp
        feas = linprog(np.zeros(n_comp), A_ub=A_ub, b_ub=b_ub,
                       bounds=bounds, method="highs")
        if feas.status == 2:
            return None
        out = np.empty((n_comp, 2))
        for k in range(n_comp):
            cvec = np.zeros(n_comp)
            cvec[k] = 1.0
            r_min = linprog(cvec, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                            method="highs")
            out[k, 0] = 0.0 if r_min.status in (2, 3) else r_min.fun
            r_max = linprog(-cvec, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                            method="highs")
            if r_max.status == 3:
                out[k, 1] = math.inf
            elif r_max.status == 2:
                out[k, 1] = out[k, 0]
            else:
                out[k, 1] = -r_max.fun
        out[:, 0] = np.maximum(out[:, 0], 0.0)
        out[:, 1] = np.maximum(out[:, 1], out[:, 0])
        return out

    def evaluate(path: list[tuple[int, int]]) -> dict | None:
        M = build_matrix(path)
        if quick_prune(M):
            return None
        A_ub, b_ub = build_lp(M)
        ext = exact_bounds(A_ub, b_ub)
        if ext is None:
            return None
        pairs = []
        for (i_, j_) in path:
            pairs.append({"ref_id": int(ref_ids[i_]),
                          "ref_time": float(ref_t[i_]),
                          "sample_time": float(cand_t[j_]),
                          "candidate_index": int(j_)})
        for a, b in zip(pairs[:-1], pairs[1:]):
            a["segment_slope_to_next"] = (
                b["sample_time"] - a["sample_time"]
            ) / (b["ref_time"] - a["ref_time"])
        return {"pairs": pairs, "lower": ext[:, 0], "upper": ext[:, 1]}

    explanations: list[dict] = []
    best_length = 0
    total_matchings_at_length = 0
    truncated = False
    leaves_seen = 0
    path_stack: list[tuple[int, int]] = []

    def dfs(node: tuple[int, int], rem: int) -> None:
        nonlocal truncated, leaves_seen
        path_stack.append(node)
        if rem == 1:
            leaves_seen += 1
            if leaves_seen <= _ENUM_PATH_CAP and time.perf_counter() < deadline:
                expl = evaluate(path_stack)
                if expl is not None:
                    explanations.append(expl)
            else:
                truncated = True
            path_stack.pop()
            return
        for s in succ[node]:
            if best[s] >= rem - 1:
                dfs(s, rem - 1)
                if truncated:
                    break
        path_stack.pop()

    for length in range(longest, 1, -1):
        explanations = []
        truncated = False
        leaves_seen = 0
        total_matchings_at_length = int(
            sum(cnt[length, i, j] for (i, j) in nodes))
        for node in nodes:
            if best[node] >= length:
                dfs(node, length)
            if truncated:
                break
        if explanations:
            best_length = length
            break

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not explanations:
        result: dict = {
            "reason": "no anchor matching with at least two pairs admits "
                      "concentrations meeting the tolerance/saturation "
                      "constraints",
            "longest_valid_matching": int(longest),
            "elapsed_ms": round(elapsed_ms, 3),
        }
        if truncated:
            result["status"] = "exhaustion"
            result["reason"] = (
                "search budget exhausted before all anchor matchings could be "
                "proved; no feasible explanation found in the enumerated "
                "subset.  Tighten drift_slope or reduce candidate_markers for "
                "a certified answer.")
            result["matchings_of_that_length"] = total_matchings_at_length
            return result
        result["status"] = "infeasible"
        return result

    glob_lo = np.full(n_comp, math.inf)
    glob_hi = np.full(n_comp, -math.inf)
    for e in explanations:
        glob_lo = np.minimum(glob_lo, e["lower"])
        glob_hi = np.maximum(glob_hi, e["upper"])

    pair_support: dict[tuple[int, int], int] = {}
    for e in explanations:
        for p in e["pairs"]:
            key = (p["ref_id"], p["candidate_index"])
            pair_support[key] = pair_support.get(key, 0) + 1
    n_expl = len(explanations)

    def bound_json(v: float):
        return None if not math.isfinite(v) else float(v)

    components = []
    for k in range(n_comp):
        components.append({
            "id": std_ids[k],
            "pure_concentration": float(pure_c[k]),
            "ratio_lower": float(glob_lo[k]),
            "ratio_upper": bound_json(glob_hi[k]),
            "concentration_lower": float(glob_lo[k] * pure_c[k]),
            "concentration_upper": bound_json(glob_hi[k] * pure_c[k]),
        })

    anchor_evidence = []
    for (rid, jndx), supp in sorted(pair_support.items(),
                                    key=lambda kv: (-kv[1], kv[0])):
        i = int(np.where(ref_ids == rid)[0][0])
        anchor_evidence.append({
            "ref_id": int(rid),
            "ref_time": float(ref_t[i]),
            "sample_time": float(cand_t[jndx]),
            "candidate_index": int(jndx),
            "explanations": supp,
            "support": supp / n_expl,
        })

    return {
        "status": "ok",
        "sample_channel": data["sample_channel"],
        "max_pairs": int(best_length),
        "longest_valid_matching": int(longest),
        "explanations_evaluated": n_expl,
        "matchings_of_that_length": total_matchings_at_length,
        "enumeration_truncated": truncated,
        "components": components,
        "anchor_pair_sets": [e["pairs"] for e in explanations],
        "anchor_evidence": anchor_evidence,
        "elapsed_ms": round(elapsed_ms, 3),
    }
