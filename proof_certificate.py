%%writefile proof_certificate.py
"""
proof_certificate.py

Certificate‑driven verification of the recursive damping inequality:

    dY/dt <= -ν Λ Y^(1+δ) + C Y^2 · sup_j A_j(t)

At each timestep this module computes the residual,
checks it against a tolerance, logs a JSON‑lines “certificate”,
and emits a warning if the inequality is violated.
"""

from pathlib import Path
import json

__all__ = ["verify_recursive_damping"]

def verify_recursive_damping(
    Y: float,
    dY_dt: float,
    alignment_sup: float,
    t: float,
    *,
    nu: float,
    Lambda: float,
    C: float,
    alpha: float,
    Y_star: float,
    tolerance: float = 1e-6,
    log_path: str = "recursive_damping_certificate.jsonl"
):
    """
    Compute the residual of the recursive damping inequality at time t, check
    it against `tolerance`, log a JSON line to `log_path`, and optionally
    print a warning if the inequality fails.

    Parameters
    ----------
    Y : float
        The current vorticity norm, Y(t).
    dY_dt : float
        The time derivative dY/dt at this timestep.
    alignment_sup : float
        sup_j 𝒜_j(t), the maximal alignment factor.
    t : float
        The current time.
    nu : float
        Viscosity coefficient ν.
    Lambda : float
        Damping constant Λ.
    C : float
        Nonlinear constant C.
    alpha : float
        Scaling exponent α (so that δ = 2/α).
    Y_star : float
        Threshold for triggering contraction certificate.
    tolerance : float, optional
        Residual tolerance (default 1e‑6).
    log_path : str, optional
        Path to append the JSONL certificate (default "recursive_damping_certificate.jsonl").

    Returns
    -------
    residual : float
        The computed residual = dY_dt + νΛY^(1+δ) − C Y^2 · alignment_sup.
    contracting : bool
        True if Y > Y_star and residual ≤ tolerance.
    """

    # Compute δ = 2/α
    delta = 2.0 / alpha

    # Residual of the inequality: LHS − RHS ≤ 0  ⇒  residual = LHS − RHS
    residual = dY_dt + nu * Lambda * (Y ** (1 + delta)) - C * (Y ** 2) * alignment_sup

    # Determine if contracting
    contracting = (Y > Y_star) and (residual <= tolerance)

    # Prepare JSON entry
    entry = {
        "time":          float(t),
        "Y":             float(Y),
        "dY_dt":         float(dY_dt),
        "residual":      float(residual),
        "alignment_sup": float(alignment_sup),
        "Y_star":        float(Y_star),
        "contracting":   bool(contracting),
    }

    # Ensure output directory exists
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # Append one line of JSON
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
        fh.flush()

    # If we expected contraction but got a violation, warn
    if (Y > Y_star) and (residual > tolerance):
        print(f"[❌ FAIL] Damping violated at t={t:.6f}, residual={residual:.2e}")

    return residual, contracting
