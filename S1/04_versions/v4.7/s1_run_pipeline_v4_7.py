"""
s1_run_pipeline_v4_7.py
Sequential conditional LHS -> iterative filter -> ROSS -> post-sim checks.

v4.4 CHANGES (2026-08-22):
  1. Disk OD/width ranges follow the old document global coverage ranges.
  2. Isotropic bearings only; damping [50, 1000] N.s/m, stiffness
     [1e6, 5e8] N/m.
  3. Disk/bearing positions use sequential conditional LHS.
  4. H6 bearing consistency: base log10 stiffness +-30% perturbations.
  5. Materials aligned with ROSS available_materials.toml / old document.
  6. H3 end margin stays at 0.1L (code version).
  7. Post-sim checks: cs1 in [50, 950000] RPM, partial/error statuses,
     and data quality report.
  8. Iterative rejection/resampling until n accepted samples are produced.
  9. Reproducibility: code_hash, parameter range table, material table,
     ROSS version and Python version in summary.json.

v4.5 CHANGES:
  A. critical_speeds() now returns FORWARD-WHIRL modes only.
     run_critical_speed(num_modes=N) returns N/2 damped whirl frequencies
     alternating backward/forward.  v4.4 took the three lowest without
     filtering, so cs1-cs3 mixed precession directions (backward/forward/
     backward in 88% of a 60-row pilot check, other patterns in the rest).
  B. num_modes raised 12 -> 24.  Pilot test on 40 rotors: fraction yielding
     three forward modes was 50.0% (12), 82.5% (16), 92.5% (20), 100% (24).
     Cost per rotor rises roughly 3x.
  C. "partial" now means fewer than THREE forward-whirl critical speeds
     (was fewer than two mixed-direction values).
  D. Soft criteria S1-S3 are recorded as labels instead of rejecting
     samples; six extra columns are written per row.


v4.6 CHANGES:
  A. Primary targets are the first THREE positive, damped, forward-whirl
     critical speeds.  Backward, mixed and non-lateral modes are retained
     in a separate long-form modal CSV for later complete-spectrum analysis.
  B. All raw modal results are preserved: wn, wd, direction, log decrement,
     damping ratio, and the original mode index.
  C. Soft criteria S1-S3 remain non-rejecting labels.
  D. Rejection-reason and prefilter-reason counters are restored in
     sampling statistics and summary.json; a rejection histogram is saved.

v4.7 CHANGES:
  A. NUM_MODES raised 24 -> 48 (24 whirl pairs).  The 24-mode budget
     censored cs4/cs5/cs6 (coverage 73% / 26% / 0.3%); at 48 modes the
     same rotors reach 100% / 100% / 99% while cs1-cs3 are bit-identical.
  B. Label columns widened from cs_1..cs_3 to cs_1..cs_6.
     MIN_FORWARD_MODES keeps the v4.6 success gate (>= 3 forward modes)
     so the v4.7 dataset is not smaller than v4.6, only better labelled.
"""
import os
# Relabeling is much faster with Numba JIT enabled.  Keep the historical
# no-JIT behavior as the default, but make the fast path explicit and
# reproducible through ROSS_ENABLE_NUMBA_JIT=1.
_enable_numba_jit = os.environ.get("ROSS_ENABLE_NUMBA_JIT", "0") == "1"
os.environ["NUMBA_DISABLE_JIT"] = "0" if _enable_numba_jit else "1"
import numba
numba.config.DISABLE_JIT = not _enable_numba_jit

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import json
import csv
import time
import hashlib
import platform
import traceback
import multiprocessing as mp

import numpy as np
import ross as rs
from scipy.optimize import newton as _newton

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import importlib.util as _iu

_cf_path = os.path.join(_here, "s1_constraint_filter_v4_7.py")
_cf_spec = _iu.spec_from_file_location("s1_constraint_filter_v4_7", _cf_path)
_cf = _iu.module_from_spec(_cf_spec)
_cf_spec.loader.exec_module(_cf)

lhs_v4                   = _cf.lhs_v4
X_to_samples_v4          = _cf.X_to_samples_v4
filter_v4                = _cf.filter_v4
to_features_v4           = _cf.to_features_v4
generate_filtered_samples = _cf.generate_filtered_samples
N_DIMS                   = _cf.N_DIMS
MATERIALS                = _cf.MATERIALS
K_MIN                    = _cf.K_MIN
K_MAX                    = _cf.K_MAX
C_MIN                    = _cf.C_MIN
C_MAX                    = _cf.C_MAX
H3_MARGIN_FRAC           = _cf.H3_MARGIN_FRAC
H4_MIN_SEP               = _cf.H4_MIN_SEP
H6_PERT                  = _cf.H6_PERT

_qc_path = os.path.join(_here, "s1_quality_check_v4_7.py")
_qc_spec = _iu.spec_from_file_location("s1_quality_check_v4_7", _qc_path)
_qc = _iu.module_from_spec(_qc_spec)
_qc_spec.loader.exec_module(_qc)
quality_check = _qc.quality_check

VERSION = "v4.7"
ERROR_LOG_NAME = "ross_errors_v4.7.log"
CS_MIN_RPM = 50.0
CS_MAX_RPM = 950000.0
N_PRIMARY_CS = 6        # label columns cs_1..cs_6
MIN_FORWARD_MODES = 3   # success gate kept at the v4.6 value


# ----------------------------------------------------------------------
# Code hash / reproducibility
# ----------------------------------------------------------------------

def _code_hash():
    h = hashlib.sha256()
    for p in (__file__, _cf_path, _qc_path):
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:16]


def _parameter_ranges():
    return {
        "shaft_length_m": [0.3, 2.0],
        "shaft_od_m": [0.02, 0.15],
        "shaft_id_m": [0.0, 0.05],
        "disk_od_m": [_cf.DISK_OD_MIN, _cf.DISK_OD_MAX],
        "disk_width_m": [_cf.DISK_W_MIN, _cf.DISK_W_MAX],
        "bearing_k_base_log10_npm": [float(np.log10(K_MIN)), float(np.log10(K_MAX))],
        "bearing_k_perturbation": [-H6_PERT, H6_PERT],
        "bearing_c_ns_per_m": [C_MIN, C_MAX],
        "h3_margin_frac": H3_MARGIN_FRAC,
        "h4_separation_m": H4_MIN_SEP,
    }


def _material_table():
    return {name: dict(mat) for name, mat in MATERIALS.items()}


# ----------------------------------------------------------------------
# Unequal-length element discretization (same core as v4.3)
# ----------------------------------------------------------------------

def _build_unequal_length_elements(s, material, n_base=15):
    """Build shaft elements with nodes exactly at disk/bearing positions."""
    sl = s["sl"]
    base_dx = sl / n_base
    tolerance = 1e-3

    z_feats = []
    for i in range(s["nd"]):
        z_feats.append(s["dp"][i])
    for i in range(s["nb"]):
        z_feats.append(s["bp"][i])

    if not z_feats:
        z_boundaries = [0.0, sl]
    else:
        z_feats.sort()
        merged = [z_feats[0]]
        for z in z_feats[1:]:
            if z - merged[-1] < tolerance:
                merged[-1] = (merged[-1] + z) / 2.0
            else:
                merged.append(z)
        z_boundaries = [0.0]
        for z in merged:
            if z > tolerance and z < sl - tolerance:
                z_boundaries.append(z)
        z_boundaries.append(sl)

    elms = []
    node_z = [z_boundaries[0]]
    node_idx = 0

    for seg_i in range(len(z_boundaries) - 1):
        z_a = z_boundaries[seg_i]
        z_b = z_boundaries[seg_i + 1]
        seg_len = z_b - z_a
        if seg_len <= 0:
            continue
        n_sub = max(1, int(np.ceil(seg_len / base_dx)))
        sub_dx = seg_len / n_sub
        for sub_i in range(n_sub):
            elms.append(rs.ShaftElement(
                n=node_idx,
                L=sub_dx,
                idl=s["id"],
                odl=max(s["od"], 0.01),
                material=material,
            ))
            node_idx += 1
            node_z.append(z_a + (sub_i + 1) * sub_dx)

    def find_node(z_target):
        best_i = 0
        best_d = abs(node_z[0] - z_target)
        for i, nz in enumerate(node_z):
            d = abs(nz - z_target)
            if d < best_d:
                best_d = d
                best_i = i
            else:
                if nz > z_target + best_d:
                    break
        return best_i

    return elms, find_node, len(node_z) - 1


def _get_material(name):
    mat = MATERIALS[name]
    try:
        from ross.materials import Material
        return Material(name=name, E=mat["E"], G_s=mat["G_s"], rho=mat["rho"])
    except (ImportError, AttributeError):
        pass
    try:
        from ross.material import Material
        return Material(name=name, E=mat["E"], G_s=mat["G_s"], rho=mat["rho"])
    except (ImportError, AttributeError):
        pass
    import ross as _rs
    for src in [lambda: _rs.steel, lambda: _rs.Steel(), lambda: _rs.materials.steel]:
        try:
            obj = src()
            obj.name = name
            obj.E = mat["E"]
            obj.G_s = mat["G_s"]
            obj.rho = mat["rho"]
            return obj
        except (AttributeError, ImportError, TypeError):
            continue
    raise ImportError("Could not construct Material object. Upgrade ROSS.")


def build_ross_rotor(s, n_base=15):
    import ross as rs
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    material = _get_material(s["material"])
    elms, find_node, n_nodes = _build_unequal_length_elements(s, material, n_base)

    disks = []
    for i in range(s["nd"]):
        node = find_node(s["dp"][i])
        try:
            disk = rs.DiskElement.from_geometry(
                n=node, material=material,
                width=s["disk_w_list"][i], i_d=0.0, o_d=s["disk_od_list"][i],
            )
        except TypeError:
            disk = rs.DiskElement.from_geometry(
                n=node, material=material,
                width=s["disk_w_list"][i],
                inner_diameter=0.0, outer_diameter=s["disk_od_list"][i],
            )
        disks.append(disk)

    bearings = []
    for i in range(s["nb"]):
        node = find_node(s["bp"][i])
        bearings.append(rs.BearingElement(
            n=node, kxx=s["bk_list"][i], cxx=s["bc_list"][i],
        ))

    return rs.Rotor(elms, disks, bearings)


NUM_MODES = 48          # 24 modes = 12 whirl pairs; only 0.3% of rotors reach cs6
                        # 24 -> 100% in a 40-rotor pilot test


def _safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _direction_text(value):
    if value is None:
        return None
    text = str(value)
    return text if text and text.lower() != "none" else None


def _eigen_frequency_arrays(rotor, speed, num_modes):
    """Return the same first wn/wd arrays as Rotor.run_modal, without shapes."""
    evalues, _ = rotor._eigen(speed, num_modes=num_modes, sparse=True)
    wn_len = num_modes // 2
    wn = np.absolute(evalues)[:wn_len]
    wd = np.imag(evalues)[:wn_len]
    return wn, wd


def _eigen_data(rotor, speed, num_modes):
    """Return eigenfrequencies and eigenpairs once for a speed."""
    evalues, evectors = rotor._eigen(speed, num_modes=num_modes, sparse=True)
    wn_len = num_modes // 2
    wn = np.absolute(evalues)[:wn_len]
    wd = np.imag(evalues)[:wn_len]
    return wn, wd, evalues, evectors


def _orbit_kappa(ru_e, rv_e):
    """Return the sign-bearing orbit parameter used by ROSS' Orbit class."""
    ru = abs(ru_e)
    rv = abs(rv_e)
    nu = np.angle(ru_e)
    nv = np.angle(rv_e)
    matrix = np.array(
        [[ru * np.cos(nu), -ru * np.sin(nu)],
         [rv * np.cos(nv), -rv * np.sin(nv)]]
    )
    h = matrix @ matrix.T
    eigenvalues = np.linalg.eigvalsh(h)
    minor = np.sqrt(max(float(eigenvalues[0]), 0.0))
    major = np.sqrt(max(float(eigenvalues[1]), 0.0))
    if major == 0:
        return 0.0
    diff = nv - nu
    if diff < -np.pi:
        diff += 2.0 * np.pi
    elif diff > np.pi:
        diff -= 2.0 * np.pi
    if diff == 0.0 or diff == np.pi:
        return 0.0
    if 0.0 < diff < np.pi:
        return -minor / major
    return minor / major


def _whirl_direction_from_eigenvector(rotor, vector):
    """Replicate ModalResults.whirl_direction for one eigenvector cheaply."""
    vector = np.asarray(vector, dtype=complex)
    number_dof = rotor.number_dof
    modex = vector[0::number_dof]
    modey = vector[1::number_dof]
    if not len(modex) or not len(modey):
        return None
    ixmax = int(np.argmax(np.abs(modex)))
    iymax = int(np.argmax(np.abs(modey)))
    if abs(modey[iymax]) > abs(modex[ixmax]):
        vector = vector / modey[iymax]
    else:
        vector = vector / modex[ixmax]

    if number_dof == 6:
        size = len(vector)
        vector_norm = np.linalg.norm(np.abs(vector))
        if vector_norm == 0:
            return None
        norm_vec = np.abs(vector) / vector_norm
        nonzero_dofs = np.where(norm_vec > 0.08)[0]
        dofs_count = [
            int(np.isin(np.arange(i, size, number_dof), nonzero_dofs).sum())
            for i in range(number_dof)
        ]
        count_sum = sum(dofs_count)
        if count_sum and dofs_count[2] / count_sum > 0.9:
            return None
        if count_sum and dofs_count[5] / count_sum > 0.9:
            return None
    elif number_dof == 1:
        return None

    directions = []
    for node in range(len(rotor.nodes)):
        start = number_dof * node
        if start + 1 >= len(vector):
            break
        kappa = _orbit_kappa(vector[start], vector[start + 1])
        directions.append("Forward" if kappa > 0 else "Backward")
    if not directions:
        return None
    if all(direction == "Forward" for direction in directions):
        return "Forward"
    if all(direction == "Backward" for direction in directions):
        return "Backward"
    return "Mixed"


def _fast_critical_speed_result(rotor, num_modes=NUM_MODES, rtol=0.005):
    """Fast synchronous-wd critical-speed extraction for forward relabeling.

    The primary v4.7 labels depend only on the damped roots (s = wd).  This
    implementation skips the separate undamped-root solve and keeps the final
    eigenpair from the damped Newton iteration, so direction and damping use
    exactly the same endpoint without recalculating mode shapes.
    """
    _, wd_initial, _, _ = _eigen_data(rotor, 0.0, num_modes)
    wn = np.zeros_like(wd_initial)
    wd = np.zeros_like(wd_initial)
    final_eigenpairs = [None] * len(wd)
    for i in range(len(wd)):
        def damped_root(speed, index=i):
            _, wd_values, evalues, evectors = _eigen_data(
                rotor, speed, num_modes
            )
            final_eigenpairs[index] = (evalues, evectors)
            return speed - wd_values[index]

        wd[i] = _newton(
            func=damped_root,
            x0=wd_initial[i],
            rtol=rtol,
        )

    log_dec = np.zeros_like(wn)
    damping_ratio = np.zeros_like(wn)
    whirl_direction = []
    for i, _ in enumerate(wd):
        evalues, evectors = final_eigenpairs[i]
        wn[i] = float(np.absolute(evalues[i]))
        abs_evalues = np.absolute(evalues)
        damping_ratio[i] = float((-np.real(evalues[i]) / abs_evalues[i]))
        denom = np.sqrt(max(1.0 - damping_ratio[i] ** 2, 0.0))
        log_dec[i] = float(2.0 * np.pi * damping_ratio[i] / denom) if denom else 0.0
        whirl_direction.append(
            _whirl_direction_from_eigenvector(rotor, evectors[: rotor.ndof, i])
        )

    return (
        np.asarray(wn),
        np.asarray(wd),
        np.asarray(log_dec),
        np.asarray(damping_ratio),
        np.asarray(whirl_direction, dtype=object),
    )


def critical_speed_result(rotor, num_modes=NUM_MODES):
    """Return (sorted forward-whirl speeds, all raw modal rows).

    The raw rows deliberately retain every mode returned by ROSS, including
    backward, mixed, and non-lateral modes.  Only positive damped forward
    modes are used for the primary cs1-cs3 labels.
    """
    if os.environ.get("ROSS_FAST_RELABEL", "0") == "1":
        wn_rad, wd_rad, log_dec, damping_ratio, whirl = (
            _fast_critical_speed_result(rotor, num_modes=num_modes)
        )
        rpm_factor = 60.0 / (2.0 * np.pi)
        wn = wn_rad * rpm_factor
        wd = wd_rad * rpm_factor
    else:
        res = rotor.run_critical_speed(num_modes=num_modes)
        wn = res.wn("rpm")
        wd = res.wd("rpm")
        whirl = res.whirl_direction
        log_dec = res.log_dec
        damping_ratio = res.damping_ratio

    raw = []
    for idx in range(len(wd)):
        direction = _direction_text(whirl[idx]) if idx < len(whirl) else None
        wd_value = _safe_float(wd[idx])
        raw.append({
            "mode_index": idx + 1,
            "mode_order": idx,
            "wn_rpm": _safe_float(wn[idx]) if idx < len(wn) else None,
            "wd_rpm": wd_value,
            "whirl_direction": direction,
            "log_dec": _safe_float(log_dec[idx]) if idx < len(log_dec) else None,
            "damping_ratio": _safe_float(damping_ratio[idx]) if idx < len(damping_ratio) else None,
        })

    forward = sorted(
        m["wd_rpm"] for m in raw
        if m["wd_rpm"] is not None
        and m["wd_rpm"] > 0
        and (m["whirl_direction"] or "").lower() == "forward"
    )
    return forward, raw


def critical_speeds(rotor, num_modes=NUM_MODES):
    """Compatibility helper: primary forward-whirl critical speeds only."""
    return critical_speed_result(rotor, num_modes=num_modes)[0]


def evaluate_sample(sample_id, s, cs_min=CS_MIN_RPM, cs_max=CS_MAX_RPM,
                    num_modes=NUM_MODES):
    """Run one ROSS evaluation and return (row, ok, raw_modal_rows)."""
    try:
        rotor = build_ross_rotor(s)
        forward, raw = critical_speed_result(rotor, num_modes=num_modes)
        row = {
            "sample_id": sample_id,
            "L_m": round(s["sl"], 4),
            "OD_m": round(s["od"], 6),
            "ID_m": round(s["id"], 6),
            "material": s["material"],
            "bearing_type": s["bearing_type"],
            "n_disks": s["nd"],
            "n_bearings": s["nb"],
            "status": "success",
            "forward_mode_count": len(forward),
            "raw_mode_count": len(raw),
            "qc_note": "",
        }
        soft = s.get("soft") or {}
        for key in ("S1_volume_ratio", "S2_support_sag", "S3_bearing_span"):
            row[key] = bool(soft.get(key)) if key in soft else None
        for key in ("volume_ratio", "support_sag_m", "bearing_span_frac"):
            row[key] = round(soft[key], 6) if key in soft else None

        if len(forward) < MIN_FORWARD_MODES:
            row["status"] = "partial"
            row["qc_note"] = (
                f"Fewer than {MIN_FORWARD_MODES} forward-whirl critical speeds found"
            )
        elif forward[0] < cs_min or forward[0] > cs_max:
            row["status"] = "rejected_cs1"
            row["qc_note"] = (
                f"cs1={forward[0]:.1f} outside "
                f"[{cs_min:.0f},{cs_max:.0f}] RPM"
            )

        for j, value in enumerate(forward[:N_PRIMARY_CS]):
            row[f"cs_{j + 1}_rpm"] = round(value, 1)
        for j in range(len(forward[:N_PRIMARY_CS]), N_PRIMARY_CS):
            row[f"cs_{j + 1}_rpm"] = None

        direction_counts = {}
        for modal in raw:
            direction = modal["whirl_direction"] or "None"
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        row["modal_direction_counts"] = json.dumps(
            direction_counts, ensure_ascii=False, sort_keys=True
        )

        modal_rows = []
        for modal in raw:
            out = dict(modal)
            out["sample_id"] = sample_id
            out["status"] = row["status"]
            modal_rows.append(out)
        return row, row["status"] == "success", modal_rows
    except Exception as e:
        row = {
            "sample_id": sample_id,
            "status": "error",
            "error_msg": str(e)[:200],
            "qc_note": "ROSS simulation error",
            "L_m": round(s.get("sl", 0), 4),
            "OD_m": round(s.get("od", 0), 6),
            "n_disks": s.get("nd", 0),
            "n_bearings": s.get("nb", 0),
        }
        return row, False, []


# ----------------------------------------------------------------------
# Worker with post-simulation checks
# ----------------------------------------------------------------------

def _process_one_sample(args):
    i, s, cs_min, cs_max = args
    row, ok, modal_rows = evaluate_sample(i, s, cs_min, cs_max)
    return i, row, ok, modal_rows


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def _merge_stats(total, extra):
    """Accumulate scalar and nested-counter sampling statistics."""
    for key, value in extra.items():
        if isinstance(value, dict):
            target = total.setdefault(key, {})
            for sub_key, sub_value in value.items():
                target[sub_key] = target.get(sub_key, 0) + sub_value
        elif isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        else:
            total[key] = value


def _save_rejection_histogram(samp_stats, out_path):
    """Save a compact rejection/prefilter histogram when matplotlib exists."""
    reasons = dict(samp_stats.get("rejection_reasons", {}))
    prefilters = dict(samp_stats.get("prefilter_reasons", {}))
    keys = list(reasons.keys()) + [k for k in prefilters if k not in reasons]
    if not keys:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    values = [reasons.get(k, 0) + prefilters.get(k, 0) for k in keys]
    order = np.argsort(values)
    keys = [keys[i] for i in order]
    values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.42 * len(keys) + 1.5)))
    ax.barh(keys, values, color="#4472C4")
    ax.set_xlabel("Rejected candidates")
    ax.set_title("v4.7 rejection / prefilter reasons")
    for y, value in enumerate(values):
        ax.text(value, y, f" {value}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def run(n=5000, out_dir="output_v4.7", max_rpm=950000, seed=42, n_base=15,
        n_workers=None, max_sampling_rounds=20, max_sim_rounds=5,
        batch_factor=2.0, num_modes=NUM_MODES):
    os.makedirs(out_dir, exist_ok=True)
    error_log_path = os.path.join(out_dir, ERROR_LOG_NAME)
    if os.path.exists(error_log_path):
        os.remove(error_log_path)

    t_start = time.time()
    print("=" * 62)
    print("  S1 Pipeline v4.7: first 3 forward-whirl cs + full modal audit")
    print(f"  n_base = {n_base} | target = {n} | seed = {seed}")
    print(f"  cs1 range = [{CS_MIN_RPM:.0f}, {max_rpm:.0f}] RPM | num_modes={num_modes}")
    print("=" * 62)

    # Step 1: iterative filter/resample
    print("[1/4] Sequential conditional LHS + iterative filtering...")
    valid, samp_stats = generate_filtered_samples(
        n,
        seed=seed,
        max_rounds=max_sampling_rounds,
        batch_factor=batch_factor,
    )
    if len(valid) == 0:
        print("  No valid samples generated; aborting.")
        return
    print(
        f"  LHS rounds={samp_stats['n_rounds']}, lhs_rows={samp_stats['n_lhs_rows']}, "
        f"candidates={samp_stats['n_candidates']}, valid={samp_stats['n_valid']}, "
        f"rejected={samp_stats['n_rejected']}"
    )

    mat_counts = {}
    for s in valid:
        mat_counts[s["material"]] = mat_counts.get(s["material"], 0) + 1
    print(f"  Material distribution (first batch): {mat_counts}")
    print()

    # Step 2: ROSS simulation with iterative refill on post-sim rejection
    print("[2/4] ROSS simulation + post-simulation checks...")
    if n_workers is None:
        n_workers = 16

    accepted = []   # (sample_id, row, sample)
    all_rows = []
    all_modal_rows = []
    sample_id_counter = 0
    sim_round = 0

    while len(accepted) < n and sim_round < max_sim_rounds:
        if sim_round == 0:
            batch_valid = valid
        else:
            need = n - len(accepted)
            extra, extra_stats = generate_filtered_samples(
                need,
                seed=seed + 100000 + sim_round * 7919,
                max_rounds=max_sampling_rounds,
                batch_factor=batch_factor,
            )
            batch_valid = extra
            _merge_stats(samp_stats, extra_stats)
            print(f"  Refill round {sim_round}: generated {len(extra)} extra samples")

        if not batch_valid:
            print("  No further valid samples; stopping simulation loop.")
            break

        start_id = sample_id_counter
        sample_by_id = {start_id + j: s for j, s in enumerate(batch_valid)}
        tasks = [(start_id + j, s, CS_MIN_RPM, max_rpm)
                 for j, s in enumerate(batch_valid)]
        sample_id_counter += len(batch_valid)

        print(f"  Sim round {sim_round + 1}: {len(tasks)} samples, {n_workers} workers")
        with mp.Pool(processes=n_workers) as pool:
            for idx, (i, row, ok, modal_rows) in enumerate(
                pool.imap_unordered(_process_one_sample, tasks)
            ):
                all_rows.append(row)
                all_modal_rows.extend(modal_rows)
                if ok:
                    accepted.append((i, row, sample_by_id[i]))
                if (idx + 1) % 50 == 0 or idx + 1 == len(tasks):
                    print(f"    [{idx + 1}/{len(tasks)}] accepted={len(accepted)}")
        sim_round += 1

    accepted.sort(key=lambda x: x[0])
    accepted = accepted[:n]
    accepted_rows = [row for _, row, _ in accepted]
    accepted_samples = [s for _, _, s in accepted]
    print()

    n_ok = len(accepted_rows)
    n_partial = sum(1 for r in all_rows if r.get("status") == "partial")
    n_rejected_cs1 = sum(1 for r in all_rows if r.get("status") == "rejected_cs1")
    n_error = sum(1 for r in all_rows if r.get("status") == "error")
    n_simulated = len(all_rows)

    # Step 3: save dataset, features and audit log
    print("[3/4] Saving dataset / features / audit...")
    csv_path = os.path.join(out_dir, f"dataset_v4.7_{n}.csv")
    if accepted_rows:
        fnames = list(dict.fromkeys(k for r in accepted_rows for k in r.keys()))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fnames)
            w.writeheader()
            w.writerows(accepted_rows)

    audit_path = os.path.join(out_dir, f"audit_v4.7_{n}.csv")
    if all_rows:
        afnames = list(dict.fromkeys(k for r in all_rows for k in r.keys()))
        with open(audit_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=afnames)
            w.writeheader()
            w.writerows(all_rows)

    X_path = None
    nf = 0
    if accepted_samples:
        Xf, fn = to_features_v4(accepted_samples)
        X_path = os.path.join(out_dir, f"features_v4.7_{n}.csv")
        with open(X_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(fn)
            for row in Xf:
                w.writerow(row)
        nf = Xf.shape[1]
        print(f"  Features: {X_path} ({Xf.shape[0]} x {Xf.shape[1]})")

    modal_path = os.path.join(out_dir, f"modes_v4.7_{n}.csv")
    modal_fields = [
        "sample_id", "status", "mode_index", "mode_order", "wn_rpm",
        "wd_rpm", "whirl_direction", "log_dec", "damping_ratio",
    ]
    with open(modal_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=modal_fields)
        w.writeheader()
        for row in all_modal_rows:
            w.writerow({key: row.get(key) for key in modal_fields})
    print(f"  Raw modes: {modal_path} ({len(all_modal_rows)} rows)")

    # Step 4: quality check + summary
    print("[4/4] Quality check + reproducibility summary...")
    report_path = os.path.join(out_dir, f"quality_v4.7_{n}.json")
    if all_rows:
        report = quality_check(
            all_rows,
            report_path,
            min_cs_rpm=CS_MIN_RPM,
            max_cs_rpm=max_rpm,
            modal_rows=all_modal_rows,
        )
    else:
        report = {"n_total": 0, "checks": {}, "summary": "0 rows", "is_clean": False}

    rejection_json_path = os.path.join(out_dir, f"rejection_stats_v4.7_{n}.json")
    with open(rejection_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "rejection_reasons": samp_stats.get("rejection_reasons", {}),
            "prefilter_reasons": samp_stats.get("prefilter_reasons", {}),
        }, f, indent=2, ensure_ascii=False)
    histogram_path = _save_rejection_histogram(
        samp_stats, os.path.join(out_dir, f"rejection_histogram_v4.7_{n}.png")
    )

    total_t = time.time() - t_start
    mat_counts = {}
    for s in accepted_samples:
        mat_counts[s["material"]] = mat_counts.get(s["material"], 0) + 1
    summary = {
        "version": "v4.7",
        "n_requested": n,
        "n_base": n_base,
        "n_total": n_ok,
        "n_simulated": n_simulated,
        "n_sim_rounds": sim_round,
        "n_ok": n_ok,
        "n_partial": n_partial,
        "n_rejected_cs1": n_rejected_cs1,
        "n_error": n_error,
        "sampling_stats": samp_stats,
        "rejection_reasons": samp_stats.get("rejection_reasons", {}),
        "prefilter_reasons": samp_stats.get("prefilter_reasons", {}),
        "rejection_json": rejection_json_path,
        "rejection_histogram": histogram_path,
        "simulation_accept_rate": round(n_ok / max(n_simulated, 1), 4),
        "n_raw_modal_rows": len(all_modal_rows),
        "modal_csv": modal_path,
        "n_features": nf,
        "time_s": round(total_t, 2),
        "seed": seed,
        "material_distribution": mat_counts,
        "code_hash": _code_hash(),
        "python_version": platform.python_version(),
        "ross_version": getattr(rs, "__version__", "unknown"),
        "parameter_ranges": _parameter_ranges(),
        "materials": _material_table(),
        "post_sim_checks": {
            "cs1_min_rpm": CS_MIN_RPM,
            "cs1_max_rpm": max_rpm,
            "min_critical_speeds": MIN_FORWARD_MODES,
            "primary_target": "positive_damped_forward_whirl",
            "all_modes_preserved": True,
            "quality_is_clean": bool(report.get("is_clean")),
        },
        "error_log": error_log_path if n_error > 0 else None,
    }
    if n_error > 0:
        with open(error_log_path, "w", encoding="utf-8") as f:
            for row in all_rows:
                if row.get("status") == "error":
                    f.write(
                        f"sample_id={row.get('sample_id')} "
                        f"{row.get('error_msg', '')}\n"
                    )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Dataset:  {csv_path} ({n_ok}/{n} accepted)")
    print(f"  Audit:    {audit_path} ({n_simulated} simulated)")
    print(f"  Modes:    {modal_path} ({len(all_modal_rows)} rows)")
    if histogram_path:
        print(f"  Rejections: {histogram_path}")
    print(f"  Time: {total_t:.1f}s")
    if n_ok < n:
        print(f"  WARNING: only {n_ok}/{n} accepted after {sim_round} simulation rounds")
    if n_error > 0:
        print(f"  {n_error} ROSS errors. See '{error_log_path}' for details.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("-o", "--out-dir", default="output_v4.7")
    parser.add_argument("--max-speed", type=float, default=950000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-base", type=int, default=15)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-sampling-rounds", type=int, default=20)
    parser.add_argument("--max-sim-rounds", type=int, default=5)
    parser.add_argument("--batch-factor", type=float, default=2.0)
    parser.add_argument("--num-modes", type=int, default=NUM_MODES)
    args = parser.parse_args()
    run(args.n, args.out_dir, args.max_speed, args.seed, args.n_base,
        args.workers, args.max_sampling_rounds, args.max_sim_rounds,
        args.batch_factor, args.num_modes)
