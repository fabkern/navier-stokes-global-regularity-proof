import glob, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv, re, warnings, argparse, sys, os

FALSIFIER_LOG = "/content/output/falsifier_log.txt"

def parse_interval_tuple(s):
    # Accepts: float/int, "X ± Y", [X, Y], (X, Y), or {"mid":X,"width":Y}
    if isinstance(s, (int, float)):
        return (float(s), 0.0)
    if isinstance(s, (list, tuple)) and len(s) == 2 and all(isinstance(x, (int, float)) for x in s):
        return (float(s[0]), float(s[1]))
    if isinstance(s, dict):
        if "mid" in s and "width" in s:
            return (float(s["mid"]), float(s["width"]))
        if "val" in s and "err" in s:
            return (float(s["val"]), float(s["err"]))
        # add more keys as needed
    if isinstance(s, str):
        m = re.match(r"^\s*([-+0-9.eE]+)\s*±\s*([-+0-9.eE]+)\s*$", s)
        if m:
            return (float(m.group(1)), float(m.group(2)))
        try:
            return (float(s), 0.0)
        except:
            pass
    raise ValueError(f"Cannot parse interval from: {s}")


def falsify(reason, strict=False, do_lean_cert=False):
    with open(FALSIFIER_LOG, "a") as f:
        f.write(reason.strip() + "\n")
    print("FALSIFIER:", reason.strip())
    if do_lean_cert:
        export_falsifier_certificate(reason)
    if strict:
        sys.exit(1)

def export_falsifier_certificate(reason, fname="/content/output/falsifier_certificate.lean"):
    with open(fname, "w") as f:
        f.write(f"-- Falsifier certificate: Antagonist claim\n")
        f.write(f"theorem proof_invalid : false := by\n  -- {reason.strip()}\n  contradiction\n")

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
        t_idx = int(re.search(r'_(\d+)\.json$', f).group(1))
        if t_idx <= prev_idx:
            falsify(f"Non-monotonic timestep index in {f}.", strict=False)
        prev_idx = t_idx
        diagnostics.append(d)
        if '_entropic_per_shell' in d:
            shell_per_t.append(d['_entropic_per_shell'])
        else:
            shell_per_t.append([])
    # Guess dt:
    if len(files)>1:
        idxs = [int(re.search(r'_(\d+)\.json$', f).group(1)) for f in files]
        dts = np.diff(idxs)
        dt = float(dts[0]) if np.all(dts==dts[0]) else np.median(dts)
    else:
        dt = 1.0
    return diagnostics, shell_per_t, dt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Fail hard on any ODE falsification.")
    parser.add_argument("--lean_cert", action="store_true", help="Export Lean-style falsifier certificate.")
    # PATCH: Allow notebook/Colab to pass extraneous args
    args, unknown = parser.parse_known_args()
    diagnostics, shell_per_t, dt = load_logs()
    steps = len(diagnostics)
    t_arr = np.arange(steps) * dt
    # Extract constants and interval data
    Y_tuples  = [parse_interval_tuple(d['Y']) for d in diagnostics]
    Y_mids = [y[0] for y in Y_tuples]
    Y_wids = [y[1] for y in Y_tuples]
    dYdt_vals = [parse_interval_tuple(d['dYdt'])[0] if d['dYdt'] else np.nan for d in diagnostics]
    A_tuples  = [parse_interval_tuple(d['align_sup']) for d in diagnostics]
    A_mids = [a[0] for a in A_tuples]
    A_wids = [a[1] for a in A_tuples]
    C_nl  = float(diagnostics[0]['C_nl'])
    nu    = float(diagnostics[0].get('nu', 0.01))
    delta = float(diagnostics[0]['delta'])
    Lam   = float(diagnostics[0]['Lambda'])
    A_lowers = [max(a[0]-a[1], 0.0) for a in A_tuples]
    A_uppers = [a[0]+a[1] for a in A_tuples]
    Y0_mid = Y_mids[0]; Y0_wid = Y_wids[0]
    # Prepare ODE sim
    def ode_step(Y, A):
        return C_nl * Y**2 * A - nu * Lam * (Y**(1+delta))
    def sim_trajectory(Y0, A_series):
        Ys = [Y0]
        rhses = []
        for i in range(steps-1):
            Y = Ys[-1]
            A = A_series[i]
            rhs = ode_step(Y, A)
            rhses.append(rhs)
            Y_next = Y + dt * rhs
            if Y_next < 0 or np.isnan(Y_next) or np.isinf(Y_next):
                falsify(f"Ode trajectory blew up at t={i*dt:.4g}, Y={Y_next}", strict=args.strict, do_lean_cert=args.lean_cert)
                Y_next = 0.0
            Ys.append(Y_next)
        # Final point for rhs (for completeness)
        rhses.append(ode_step(Ys[-1], A_series[-1]))
        return np.array(Ys), np.array(rhses)
    Ys_sim_nom, rhs_vals_nom = sim_trajectory(Y0_mid, A_mids)
    Ys_sim_lo,  _            = sim_trajectory(max(Y0_mid-Y0_wid,0), A_lowers)
    Ys_sim_hi,  _            = sim_trajectory(Y0_mid+Y0_wid, A_uppers)
    # RHS sign stability check
    if any(r > 0 for r in rhs_vals_nom):
        falsify("RHS > 0 detected: net stretching over damping!", strict=args.strict, do_lean_cert=args.lean_cert)
    # Entropy
    entropies = [compute_entropy(shell) if shell else np.nan for shell in shell_per_t]
    ent_deltas = np.diff(entropies)
    if np.any(ent_deltas > 0.5 * np.abs(entropies[:-1])):
        falsify("Entropy surged >50% between steps — possible intermittent burst", strict=args.strict, do_lean_cert=args.lean_cert)
    # Relative error
    rel_errs = np.abs(Ys_sim_nom - np.array(Y_mids)) / np.maximum(np.abs(Y_mids), 1e-14)
    max_rel_err = np.max(rel_errs)
    trend_measured = Y_mids[-1] < Y_mids[0]
    trend_sim = Ys_sim_nom[-1] < Ys_sim_nom[0]
    trend_match = (trend_measured == trend_sim)
    falsified = False
    violations = []
    for i in range(steps):
        if rel_errs[i] > 0.02:
            msg = f"Step {i}: rel_err={rel_errs[i]:.3%} exceeds Clay threshold (2%)"
            violations.append(msg)
            falsify(f"FALSIFICATION: {msg}", strict=args.strict, do_lean_cert=args.lean_cert)
            falsified = True
        if Ys_sim_nom[i]<0 or np.isnan(Ys_sim_nom[i]):
            msg = f"Step {i}: Simulated Y negative or NaN!"
            violations.append(msg)
            falsify(f"FALSIFICATION: {msg}", strict=args.strict, do_lean_cert=args.lean_cert)
            falsified = True
    if not trend_match:
        falsify("Decay trend between proof logs and ODE simulation does not match (monotonicity fail)", strict=args.strict, do_lean_cert=args.lean_cert)
        falsified = True
    # Derivative check (finite diff)
    dY_log = (np.array(Y_mids[1:]) - np.array(Y_mids[:-1]))/dt
    dY_log_full = np.concatenate([ [dYdt_vals[0]], dY_log ])
    dYdt_agree = np.max(np.abs(dY_log_full-np.array(dYdt_vals))) < 0.01*np.max(np.abs(Y_mids))
    if not dYdt_agree:
        falsify("Log-differs of measured Y(t) and provided dY/dt do not match (inconsistency)", strict=args.strict, do_lean_cert=args.lean_cert)
    # Slope bound check
    if np.any(np.abs(dY_log_full) > 1e3):
        falsify("dY/dt slope magnitude exceeds 1000 — possibly unstable timestep", strict=args.strict, do_lean_cert=args.lean_cert)
    # Plot Y(t)
    plt.figure(figsize=(7,6))
    plt.plot(t_arr, Y_mids, 'o-', label='Measured Y(t)', zorder=3)
    plt.plot(t_arr, Ys_sim_nom, 's-', label='Simulated Y(t) [Nominal]', zorder=2)
    plt.fill_between(t_arr, Ys_sim_lo, Ys_sim_hi, color='lightblue', alpha=0.3, label='Sim interval')
    plt.xlabel('t')
    plt.ylabel('Y(t)')
    plt.title('Recursive ODE (with Antagonist Intervals)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("/content/output/recursive_ode_validation.png")
    # Plot A(t)
    plt.figure(figsize=(7,2))
    plt.plot(t_arr, A_mids, label='A(t) midpoint')
    plt.fill_between(t_arr, A_lowers, A_uppers, color='orange', alpha=0.2, label='A(t) interval')
    plt.ylabel('Alignment A')
    plt.xlabel('t')
    plt.title('Alignment factor interval')
    plt.legend()
    plt.savefig("/content/output/recursive_ode_A.png")
    # Plot entropy
    plt.figure(figsize=(7,2))
    plt.plot(t_arr, entropies, label='Entropy')
    plt.xlabel('t')
    plt.ylabel('H(t)')
    plt.title('Shellwise Entropy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("/content/output/recursive_ode_entropy.png")
    # CSV
    with open("/content/output/recursive_ode_trace.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'Y_measured', 'Y_measured_width', 'Y_sim_nom', 'Y_sim_lo', 'Y_sim_hi',
                        'dYdt_measured', 'A_align_mid', 'A_lo', 'A_hi', 'entropy', 'rel_error'])
        for i in range(steps):
            writer.writerow([t_arr[i], Y_mids[i], Y_wids[i], Ys_sim_nom[i],
                            Ys_sim_lo[i], Ys_sim_hi[i], dYdt_vals[i],
                            A_mids[i], A_lowers[i], A_uppers[i], entropies[i], rel_errs[i]])
    print(f"Max relative error: {max_rel_err:.4%}")
    print("Decay trend matches: " if trend_match else "DECAY TREND MISMATCH!")
    print(f"Strict falsified: {falsified}")
    print(f"CSV and plots in /content/output/")

if __name__ == "__main__":
    main()
