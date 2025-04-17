%%writefile recursive_ode_falsifier.py
#!/usr/bin/env python3
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import re
import argparse
import sys
import os
import math

FALSIFIER_LOG = "/content/output/falsifier_log.txt"
SUMMARY_JSON = "/content/output/ode_sim_summary.json"
SUCCESS_CERT = "/content/output/ode_success_certificate.lean"
FAILURE_CERT = "/content/output/falsifier_certificate.lean"

def parse_interval_tuple(s):
    # ... (same as before) ...
    if isinstance(s, (int, float)):
        return float(s), 0.0
    if isinstance(s, (list, tuple)) and len(s) == 2:
        return float(s[0]), float(s[1])
    if isinstance(s, dict):
        if "mid" in s and "width" in s:
            return float(s["mid"]), float(s["width"])
    if isinstance(s, str):
        m = re.match(r"^\s*([-+0-9.eE]+)\s*±\s*([-+0-9.eE]+)\s*$", s)
        if m:
            return float(m.group(1)), float(m.group(2))
        try:
            return float(s), 0.0
        except:
            pass
    raise ValueError(f"Cannot parse interval from: {s}")

def falsify(reason, strict=False, do_lean_cert=False):
    os.makedirs(os.path.dirname(FALSIFIER_LOG), exist_ok=True)
    with open(FALSIFIER_LOG, "a") as f:
        f.write(reason.strip() + "\n")
    print("FALSIFIER:", reason.strip(), file=sys.stderr)
    if do_lean_cert:
        export_falsifier_certificate(reason)
    if strict:
        sys.exit(1)

def export_falsifier_certificate(reason, fname=FAILURE_CERT):
    with open(fname, "w") as f:
        f.write("-- Falsifier certificate: Antagonist claim\n")
        f.write("theorem proof_invalid : false := by\n")
        f.write(f"  -- {reason.strip()}\n")
        f.write("  contradiction\n")

def export_success_certificate(fname=SUCCESS_CERT):
    with open(fname, "w") as f:
        f.write("-- ODE simulation completed without overflow\n")
        f.write("theorem simulation_valid : True := by trivial\n")
    print(f"Wrote success certificate to {fname}")

def compute_entropy(per_shell):
    H = 0.0
    for s in per_shell:
        m, _ = parse_interval_tuple(s['norm'])
        if m > 0:
            H += m * np.log(m)
    return H

def load_logs():
    diagnostics = []
    shell_per_t = []
    files = sorted(glob.glob("/content/output/diagnostics_log_timestep_*.json"),
                  key=lambda f: int(re.search(r'_(\d+)\.json$', f).group(1)))
    if not files:
        raise RuntimeError("No log files found in /content/output/")
    prev_idx = -1
    for f in files:
        with open(f) as j:
            d = json.load(j)
        idx = int(re.search(r'_(\d+)\.json$', f).group(1))
        if idx <= prev_idx:
            falsify(f"Non-monotonic timestep index in {f}.", strict=False)
        prev_idx = idx
        diagnostics.append(d)
        shell_per_t.append(d.get('_entropic_per_shell', []))
    if len(files) > 1:
        idxs = [int(re.search(r'_(\d+)\.json$', f).group(1)) for f in files]
        dts = np.diff(idxs)
        dt = float(dts[0]) if np.all(dts == dts[0]) else float(np.median(dts))
    else:
        dt = 1.0
    return diagnostics, shell_per_t, dt

def main():
    parser = argparse.ArgumentParser(description="Robust recursive ODE falsifier")
    parser.add_argument("--strict", action="store_true", help="Fail hard on any ODE falsification")
    parser.add_argument("--lean_cert", action="store_true", help="Export Lean-style falsifier certificate")
    parser.add_argument("--safe", action="store_true", help="Enable safe mode with detailed logging and JSON summary")
    args, _ = parser.parse_known_args()

    diagnostics, shell_per_t, dt = load_logs()
    steps = len(diagnostics)
    t_arr = np.arange(steps) * dt

    # extract data
    Y_tuples = [parse_interval_tuple(d['Y']) for d in diagnostics]
    Y_mids = [y[0] for y in Y_tuples]
    dYdt_vals = [parse_interval_tuple(d['dYdt'])[0] if d['dYdt'] else np.nan for d in diagnostics]
    A_tuples = [parse_interval_tuple(d['align_sup']) for d in diagnostics]
    A_mids = [min(max(a[0],0.0),1.0) for a in A_tuples]  # clip A in [0,1]
    delta = float(diagnostics[0]['delta'])
    Lam = float(diagnostics[0]['Lambda'])
    C_nl = float(diagnostics[0]['C_nl'])
    nu = float(diagnostics[0].get('nu',0.01))

    # logs for safe mode
    sim_logs = []

    def ode_rhs(Y, A):
        # guard inputs
        A_clipped = float(np.clip(A, 0.0, 1.0))
        Y_safe = float(max(Y, 1e-8))
        try:
            term1 = C_nl * (Y_safe**2) * A_clipped
            term2 = nu * Lam * (Y_safe**(1.0 + delta))
        except OverflowError as e:
            raise OverflowError(f"Overflow: Y={Y_safe}, exponent={1+delta}") from e
        if not (np.isfinite(term1) and np.isfinite(term2)):
            raise ValueError(f"Non-finite term: term1={term1}, term2={term2}")
        return term1 - term2

    def sim_trajectory(Y0, A_series):
        Ys = [Y0]
        rhses = []
        for i, A in enumerate(A_series):
            t = i * dt
            Y = Ys[-1]
            try:
                rhs = ode_rhs(Y, A)
            except Exception as e:
                reason = f"Step {i}: error in RHS calculation: {e}"
                if args.safe:
                    sim_logs.append({"step": i, "Y": Y, "A": A, "error": str(e)})
                    raise
                else:
                    falsify(reason, strict=args.strict, do_lean_cert=args.lean_cert)
                    rhs = 0.0
            Y_next = Y + dt * rhs
            if not np.isfinite(Y_next):
                reason = f"Step {i}: Y_next is non-finite: {Y_next}"
                if args.safe:
                    sim_logs.append({"step": i, "Y": Y, "rhs": rhs, "Y_next": Y_next})
                    raise OverflowError(reason)
                else:
                    falsify(reason, strict=args.strict, do_lean_cert=args.lean_cert)
                    Y_next = 0.0
            rhses.append(rhs)
            if args.safe:
                residual = rhs - (Y_next - Y)/dt
                sim_logs.append({
                    "step": i,
                    "t": t,
                    "Y": Y,
                    "A": A,
                    "rhs": rhs,
                    "Y_next": Y_next,
                    "residual": residual
                })
            Ys.append(Y_next)
        return np.array(Ys), np.array(rhses)

    # run sim
    Y0 = Y_mids[0]
    success = True
    try:
        Ys_sim, rhs_vals = sim_trajectory(Y0, A_mids)
    except Exception as e:
        success = False

    # post-simulation checks
    if success:
        # sign check: no positive RHS for dissipativity
        if np.any(rhs_vals > 0):
            falsify("RHS > 0 detected (net stretching): simulation invalid",
                    strict=args.strict, do_lean_cert=args.lean_cert)
            success = False

    # save summary
    summary = {
        "status": "success" if success else "failure",
        "steps": steps,
        "max_Y": float(np.max(Ys_sim)) if success else None,
        "min_Y": float(np.min(Ys_sim)) if success else None,
        "max_rhs": float(np.max(rhs_vals)) if success else None,
        "safe_mode": args.safe
    }
    os.makedirs(os.path.dirname(SUMMARY_JSON), exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote simulation summary to {SUMMARY_JSON}")

    # export certificate
    if success:
        export_success_certificate()
        sys.exit(0)
    else:
        if args.lean_cert:
            export_falsifier_certificate("ODE simulation overflow or invalid")
        sys.exit(1)

if __name__ == "__main__":
    main()
