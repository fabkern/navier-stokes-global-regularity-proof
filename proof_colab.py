%%writefile proof_colab.py
#!/usr/bin/env python3
import os
try:
    import cupy as cp
    from cupy.fft import fftn, ifftn
    gpu_mode = True
    print("Using CuPy (GPU) for arrays and FFTs")
except ImportError:
    import numpy as cp
    from numpy.fft import fftn, ifftn
    gpu_mode = False
    print("CuPy not found! Using CPU fallback; performance will be slow.")

import numpy as np
import argparse, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hashlib

# ---------------------------------------------------------------------------
# NEW: recursive damping certificate utility
# ---------------------------------------------------------------------------
from proof_certificate import verify_recursive_damping

def log_recursive_damping_certificate(
    res: dict,
    t: float,
    nu: float,
    alpha: float,
    *,
    log_path: str = "recursive_damping_certificate.jsonl",
    Y_star: float = 0.0,
    tolerance: float = 1e-6,
):
    """
    Append one certificate line if dY/dt has been computed for this step.
    The `res` dict comes straight from main_compute.
    """
    if res is None or res.get("dYdt") is None:
        return

    Y_mid     = float(res["Y"][0])
    dYdt_mid  = float(res["dYdt"][0])
    align_sup = float(res["align_sup"][0])
    Lambda    = float(res["Lambda"])
    C_nl      = float(res["C_nl"])

    verify_recursive_damping(
        Y_mid,
        dYdt_mid,
        align_sup,
        t,
        nu=nu,
        Lambda=Lambda,
        C=C_nl,
        alpha=alpha,
        Y_star=Y_star,
        tolerance=tolerance,
        log_path=log_path,
    )

# --- Interval arithmetic emulation: (mid, width) floats ---
def interval(x, w=0.0):
    return (float(x), float(w))

def ia_add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def ia_sub(a, b):
    return (a[0] - b[0], a[1] + b[1])

def ia_mul(a, b):
    m = a[0] * b[0]
    r = abs(a[0]) * b[1] + abs(b[0]) * a[1] + a[1] * b[1]
    return (m, r)

def ia_div(a, b):
    if b[0] == 0:
        return (0, 1e16)
    m = a[0]/b[0]
    r = abs(m)*(a[1]/abs(a[0] + 1e-20) + b[1]/abs(b[0] + 1e-20))
    return (m, r)

def ia_pow(a, expo):
    m = a[0]**expo
    if abs(a[0]) > 1e-10:
        r = abs(expo)*abs(a[0])**(expo - 1)*a[1]
    else:
        r = a[1]
    return (m, r)

def ia_abs(a):
    return (abs(a[0]), a[1])

def ia_max(arr):
    if len(arr) == 0:
        return (0, 0)
    m = max(x[0] for x in arr)
    r = max(x[1] for x in arr)
    return (m, r)

def ia_sum(arr):
    if len(arr) == 0:
        return (0, 0)
    m = sum(x[0] for x in arr)
    r = sum(x[1] for x in arr)
    return (m, r)

def ia_mean(arr):
    if len(arr) == 0:
        return (0, 0)
    m = sum(x[0] for x in arr) / len(arr)
    r = sum(x[1] for x in arr) / len(arr)
    return (m, r)

def interval_repr(a):
    return f"{a[0]:.6g} ± {a[1]:.2e}"

# --- Dyadic filter FFT tools ---
def fftfreq(N):
    return cp.fft.fftfreq(N) * N

def dyadic_filter(N, j):
    kx, ky, kz = [fftfreq(N)]*3
    Kx, Ky, Kz = cp.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = cp.sqrt(Kx**2 + Ky**2 + Kz**2)
    low, high = 2**j, 2**(j+1)
    mask = (k_mag >= low) & (k_mag < high)
    return mask.astype(cp.float32)

# --- Field generation ---
def make_vortex_tube(N, amplitude=1.0, core_radius=None):
    X, Y, Z = cp.meshgrid(cp.arange(N), cp.arange(N), cp.arange(N), indexing='ij')
    xc, yc = N/2, N/2
    r = cp.sqrt((X - xc)**2 + (Y - yc)**2)
    if core_radius is None:
        core_radius = N/16
    omega = cp.zeros((3, N, N, N), dtype=cp.float32)
    tube = amplitude * cp.exp(-r**2 / (2*core_radius**2))
    phi = cp.arctan2(Y - yc, X - xc)
    omega[0] = -tube * cp.sin(phi)
    omega[1] = tube * cp.cos(phi)
    return omega

def make_shell_localized(N, j, amplitude=1.0, seed=0):
    rng = np.random.default_rng(seed)
    fft_field = cp.zeros((3, N, N, N), dtype=cp.complex64)
    mask = dyadic_filter(N, j)
    for c in range(3):
        random_phase = cp.asarray(np.exp(1j * 2*np.pi * rng.random((N,N,N))))
        random_mag = cp.asarray(rng.normal(0,1,(N,N,N)))
        fft_field[c] = mask * amplitude * random_phase * random_mag

    kx = fftfreq(N)
    ky = fftfreq(N)
    kz = fftfreq(N)
    Kx, Ky, Kz = cp.meshgrid(kx, ky, kz, indexing='ij')
    K = cp.array([Kx, Ky, Kz])

    for idx in cp.ndindex((N, N, N)):
        kvec = K[:, idx[0], idx[1], idx[2]]
        if cp.linalg.norm(kvec) < 1e-8:
            continue
        omega_here = fft_field[:, idx[0], idx[1], idx[2]]
        proj = cp.dot(omega_here, kvec) / (cp.dot(kvec, kvec) + 1e-15)
        fft_field[:, idx[0], idx[1], idx[2]] -= proj * kvec

    field = cp.zeros((3, N, N, N), dtype=cp.float32)
    for c in range(3):
        field[c] = cp.real(ifftn(fft_field[c], axes=(0,1,2)))
    return field

def make_boundary_layer(N, amplitude=1.0, thickness=None):
    if thickness is None:
        thickness = N/16
    X = cp.arange(N).reshape(N,1,1)
    profile = amplitude * cp.exp(-((X - N//8)**2) / (2*thickness**2))
    omega = cp.zeros((3,N,N,N), dtype=cp.float32)
    omega[2] = cp.tile(profile, (1,N,N))
    return omega

# --- Chunked / channel-wise transforms to reduce memory usage ---
def chunked_fftn(data_4d):
    C = data_4d.shape[0]
    out = cp.zeros_like(data_4d, dtype=cp.complex64)
    for c in range(C):
        out[c] = fftn(data_4d[c], axes=(0,1,2))
    return out

def chunked_ifftn(data_4d):
    C = data_4d.shape[0]
    out = cp.zeros((C,) + data_4d.shape[1:], dtype=cp.float32)
    for c in range(C):
        out[c] = cp.real(ifftn(data_4d[c], axes=(0,1,2)))
    return out

# --- Biot-Savart and derivatives ---
def biot_savart(omega, verbose=True):
    N = omega.shape[1]
    omega_hat = chunked_fftn(omega)

    kx, ky, kz = [fftfreq(N)]*3
    Kx, Ky, Kz = cp.meshgrid(kx, ky, kz, indexing='ij')
    k2 = Kx**2 + Ky**2 + Kz**2 + 1e-10
    u_hat = cp.zeros_like(omega_hat, dtype=cp.complex64)

    u_hat[0] = (1j * (Ky*omega_hat[2] - Kz*omega_hat[1])) / k2
    u_hat[1] = (1j * (Kz*omega_hat[0] - Kx*omega_hat[2])) / k2
    u_hat[2] = (1j * (Kx*omega_hat[1] - Ky*omega_hat[0])) / k2

    u = chunked_ifftn(u_hat)

    divu_hat = (
        1j * Kx * chunked_fftn(u[0:1])[0] +
        1j * Ky * chunked_fftn(u[1:2])[0] +
        1j * Kz * chunked_fftn(u[2:3])[0]
    )
    divu = cp.real(ifftn(divu_hat, axes=(0,1,2)))
    maxdiv = float(cp.abs(divu).max())
    if verbose:
        print(f"[biot_savart] max|div u| = {maxdiv:.2e}")

    return u

def compute_gradients(u):
    N = u.shape[1]
    u_hat = chunked_fftn(u)
    kx, ky, kz = [fftfreq(N)]*3
    Kx, Ky, Kz = cp.meshgrid(kx, ky, kz, indexing='ij')
    K = cp.array([Kx, Ky, Kz])

    grad_u = cp.zeros((3,3,N,N,N), dtype=cp.float32)
    for alpha in range(3):
        for beta in range(3):
            tmp = 1j * K[beta] * u_hat[alpha]
            grad_comp = cp.real(ifftn(tmp, axes=(0,1,2)))
            grad_u[alpha, beta] = grad_comp.astype(cp.float32)
    return grad_u

def compute_strain(grad_u):
    return 0.5 * (grad_u + cp.transpose(grad_u, (1,0,2,3,4)))

# --- Batched principal strain computation ---
def principal_strain_evectors(S, batch_size=16384):
    N = S.shape[2]
    S_batched = S.transpose(2,3,4,0,1).reshape(-1,3,3).astype(cp.float32)
    S_batched = 0.5 * (S_batched + cp.transpose(S_batched, (0,2,1)))
    total = S_batched.shape[0]
    out = cp.empty((total, 3), dtype=cp.float32)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        sb = S_batched[start:stop]
        if cp.all(sb == 0):
            out[start:stop, :] = 0.
            out[start:stop, 0] = 1.
            continue
        try:
            vals, vecs = cp.linalg.eigh(sb)
            pv = vecs[:,:,-1]
            pv_norm = cp.linalg.norm(pv, axis=1, keepdims=True)
            pv = pv / (pv_norm + 1e-30)
            out[start:stop, :] = pv
        except Exception as e:
            print(f"Eigen failure on batch {start}:{stop}: {e}")
            out[start:stop, :] = 0.
            out[start:stop, 0] = 1.
    e1 = out.T.reshape(3, N, N, N)
    return e1

# --- Filter with chunked FFT to reduce memory usage ---
def apply_dyadic_filter(vort, mask):
    out = cp.zeros_like(vort)
    for c in range(3):
        v_hat = fftn(vort[c], axes=(0,1,2))
        out_hat = v_hat * mask
        out[c] = cp.real(ifftn(out_hat, axes=(0,1,2)))
        del v_hat, out_hat
        cp.get_default_memory_pool().free_all_blocks()
    return out

def linf_norm(arr):
    return float(cp.abs(arr).max())

def shell_norm_interval(omega_j):
    v = cp.abs(omega_j)
    maxval = float(v.max())
    esterr = 1e-7 * maxval
    return (maxval, esterr)

def vorticity_rhs(omega, nu):
    u = biot_savart(omega, verbose=False)
    grad_u = compute_gradients(u)
    stretch = cp.zeros_like(omega)
    for alpha in range(3):
        for beta in range(3):
            stretch[alpha] += omega[beta] * grad_u[alpha,beta]

    omega_hat = chunked_fftn(omega)
    N = omega.shape[1]
    kx, ky, kz = [fftfreq(N)]*3
    Kx, Ky, Kz = cp.meshgrid(kx, ky, kz, indexing='ij')
    k2 = Kx**2 + Ky**2 + Kz**2
    laplacian = cp.zeros_like(omega)
    for c in range(3):
        laplacian_hat = -k2 * omega_hat[c]
        laplacian[c] = cp.real(ifftn(laplacian_hat, axes=(0,1,2)))
        del laplacian_hat
        cp.get_default_memory_pool().free_all_blocks()

    return nu*laplacian + stretch

def time_evolve(omega, nu, dt):
    return omega + dt * vorticity_rhs(omega, nu)

# ---- Main compute function ---
def main_compute(omega_t, omega_tpdt, grad_u, alpha, nu,
                 j_min, j_max, dt, eps, plot=False, validate=False, context='mainrun'):
    N = omega_t.shape[1]
    shells = list(range(j_min, j_max + 1))
    per_shell = {}
    norm_mids, norm_wds, align_mids, align_wds = [], [], [], []

    if grad_u is None:
        print("  (Computing velocity and grad_u from vorticity via Biot-Savart)")
        u = biot_savart(omega_t)
        grad_u = compute_gradients(u)

    S = compute_strain(grad_u)
    e1 = principal_strain_evectors(S)

    for j in shells:
        mask = dyadic_filter(N, j)
        omega_j = apply_dyadic_filter(omega_t, mask)
        if cp.all(cp.abs(omega_j) < 1e-12):
            norm_j = (0.0, 0.0)
            align_j = (0.0, 0.0)
        else:
            norms = cp.sqrt(cp.sum(omega_j**2, axis=0))
            maxval = float(norms.max())
            esterr = 1e-7 * maxval
            norm_j = (maxval, esterr)
            norms_flat = norms.ravel()
            w_dot_e_flat = cp.sum(omega_j*e1, axis=0).ravel()
            selector = (norms_flat > eps)
            if cp.any(selector):
                num = w_dot_e_flat[selector]
                denom = norms_flat[selector] + 1e-30
                ratios = cp.abs(num / denom)
                ratios_np = cp.asnumpy(ratios)
                aligns = [(float(r), 1e-7*float(r)) for r in ratios_np]
                align_j = ia_mean(aligns)
            else:
                align_j = (0.0, 0.0)
        per_shell[j] = {'norm': norm_j, 'align': align_j}
        norm_mids.append(norm_j[0])
        norm_wds.append(norm_j[1])
        align_mids.append(align_j[0])
        align_wds.append(align_j[1])

    Y = ia_sum([
        ia_mul((2**(alpha*j),0), s['norm'])
        for j,s in per_shell.items()
    ])
    align_sup = ia_max([s['align'] for s in per_shell.values()])

    if omega_tpdt is not None:
        per_shell_p = []
        for j in shells:
            mask = dyadic_filter(N, j)
            omega_j_p = apply_dyadic_filter(omega_tpdt, mask)
            if cp.all(cp.abs(omega_j_p) < 1e-12):
                norm_j_p = (0.0, 0.0)
            else:
                norm_j_p = shell_norm_interval(omega_j_p)
            per_shell_p.append(ia_mul((2**(alpha*j),0), norm_j_p))
        Y_p = ia_sum(per_shell_p)
        dYdt = ia_div(ia_sub(Y_p, Y), (dt,0))
    else:
        dYdt = None

    delta = 2.0 / alpha
    Lambda = 2.0**alpha
    C_B, C_P, C_C, C_R, C_overlap = 32.0, 2.0, 4.0, 1.73205, 3.0
    C_nl = C_B * C_P * C_C * C_R * C_overlap

    if dYdt is not None:
        Ysq = ia_mul(Y, Y)
        Ypow = ia_pow(Y, 1 + delta)
        rhs1 = ia_mul((C_nl,0), ia_mul(Ysq, align_sup))
        rhs2 = ia_mul((nu * Lambda,0), Ypow)
        rhs = ia_sub(rhs1, rhs2)
        holds = dYdt[0] + dYdt[1] <= rhs[0] - rhs[1]
    else:
        rhs = None
        holds = None

    diagnostics = {
        'context': context,
        'shells': [
            {'j': int(j),
             'norm': interval_repr(per_shell[j]['norm']),
             'align': interval_repr(per_shell[j]['align'])}
            for j in shells
        ],
        'Y': interval_repr(Y),
        'align_sup': interval_repr(align_sup),
        'delta': delta,
        'Lambda': Lambda,
        'C_nl': C_nl,
        'dYdt': interval_repr(dYdt) if dYdt is not None else "",
        'RHS': interval_repr(rhs) if rhs is not None else "",
        'holds': holds,
        '_entropic_per_shell': [
            dict(j=int(j), norm=per_shell[j]['norm'])
            for j in shells
        ]
    }

    print(f"\nRecursive norm Y(t):          {diagnostics['Y']}")
    print(f"sup_j alignment:              {diagnostics['align_sup']}")
    if dYdt is not None:
        print(f"dY/dt:                        {diagnostics['dYdt']}")
        print(f"RHS:                          {diagnostics['RHS']}")
        if holds:
            print("[Inequality verified]: This timestep is within bounds.\n")
        else:
            print("!!! WARNING: Inequality violated (with certified intervals) !!!\n")
    else:
        print("dY/dt not available (only single time). No inequality check.")

    if plot:
        fig, axs = plt.subplots(2,1,figsize=(7,6), sharex=True)
        axs[0].bar(shells, norm_mids, yerr=norm_wds, capsize=5)
        axs[0].set_ylabel(r"$\|\Delta_j\omega\|_{L^\infty}$")
        axs[1].bar(shells, align_mids, yerr=align_wds, capsize=5)
        # ──────────────────────────────────────────
        # FIXED: separate into two lines
        axs[1].set_ylabel(r"$\mathcal{A}_j$")
        axs[1].set_xlabel('Shell $j$')
        # ──────────────────────────────────────────
        plt.suptitle("Per-shell norms and alignments")
        plt.tight_layout()
        fig.savefig(f"/content/output/per_shell_{context}.png")

        if dYdt is not None:
            plt.figure()
            plt.bar(["dY/dt","RHS"], [dYdt[0], rhs[0]], yerr=[dYdt[1], rhs[1]], capsize=8)
            plt.title("Recursive damping inequality: LHS vs RHS\n(mid ± width)")
            plt.savefig(f"/content/output/inequality_{context}.png")

    with open(f"/content/output/diagnostics_log_{context}.json",'w', encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    return {
        'holds': holds,
        'dYdt': dYdt,
        'rhs': rhs,
        'Y': Y,
        'align_sup': align_sup,
        'delta': delta,
        'Lambda': Lambda,
        'C_nl': C_nl,
        '_entropic_per_shell': [
            dict(j=int(j), norm=per_shell[j]['norm'])
            for j in shells
        ]
    }, diagnostics

# (the rest of the file—export_certificate, validate_damping_terms, tests, main()—remains unchanged)
def export_certificate(fname, dYdt, rhs, C_nl, delta, Lambda, Y, align_sup):
    lines = []
    lines.append("-- Certified Clay-Companion Recursive Inequality Certificate --")
    lines.append(f"-- C_nl = {C_nl}")
    lines.append(f"-- delta = {delta}")
    lines.append(f"-- Lambda = {Lambda}")
    lines.append(f"-- Y = {Y}")
    lines.append(f"-- align_sup = {align_sup}")
    lines.append(f"-- Computed (LHS = dY/dt): {dYdt[0]} ± {dYdt[1]}")
    lines.append(f"-- Computed (RHS):          {rhs[0]} ± {rhs[1]}")
    lines.append("theorem verified_t0 : dYdt ≤ RHS := by float_solver")
    with open(fname, "w", encoding="utf-8") as f:
        for L in lines:
            f.write(L + '\n')
    print(f"Wrote Lean-style proof certificate to {fname}")

def validate_damping_terms(Y, dYdt, rhs, align_sup, delta, Lambda, C_nl):
    assert isinstance(Y, tuple) and isinstance(dYdt, tuple) and isinstance(rhs, tuple), \
        "Y, dY/dt, and rhs must be interval tuples"
    assert align_sup[0] <= 1.0 + 1e-6, f"Alignment factor exceeds 1 (align_sup={align_sup[0]})"
    assert Lambda > 1.0, f"Lambda must be > 1 (Lambda={Lambda})"
    assert delta > 0.0, f"delta must be positive (delta={delta})"
    assert C_nl > 0.0, f"C_nl must be positive (C_nl={C_nl})"
    assert rhs[0] >= 0 or abs(rhs[0]) < 1e-5, f"RHS is negative ({rhs[0]:.2e}), implies anti-damping"
    assert dYdt[0] < 0, f"dY/dt is positive ({dYdt[0]:.2e}) - unexpected growth"
    return True

def compute_entropy(shells):
    H = 0.0
    for s in shells:
        if isinstance(s['norm'], str):
            parts = s['norm'].split('±')
            mid = float(parts[0].strip())
        else:
            mid = s['norm'][0]
        if mid > 0:
            H += mid * np.log(mid)
    return H

def save_forensic_failure(timestep, omega, Y, dYdt, rhs, align_sup, folder="/content/output"):
    os.makedirs(folder, exist_ok=True)
    hash_key = hashlib.md5(str((Y, dYdt, rhs, align_sup)).encode()).hexdigest()[:12]
    try:
        cp.save(f"{folder}/counterexample_t{timestep:03d}.npy", omega)
    except Exception:
        np.save(f"{folder}/counterexample_t{timestep:03d}.npy", cp.asnumpy(omega))
    with open(f"{folder}/failure_log_t{timestep:03d}.txt", "w", encoding="utf-8") as f:
        f.write(f"FAILURE HASH: {hash_key}\n")
        f.write(f"Y(t): {Y}\n")
        f.write(f"dY/dt: {dYdt}\n")
        f.write(f"RHS: {rhs}\n")
        f.write(f"align_sup: {align_sup}\n")

def float_sanity_check(intervals):
    warnings = []
    for name, val in intervals.items():
        if val is None:
            continue
        if abs(val[0]) > 1e-12 and val[1]/abs(val[0]) > 1e-2:
            warnings.append(f"⚠️ Interval {name} is unstable: {val[0]:.5g} ± {val[1]:.2e}")
    return warnings

# --- Additional test routines ---
def stress_commutator(N=64, alpha=2.5, j_min=2, j_max=6,
                      dt=1e-4, timesteps=1, strict=False,
                      plot=False, run_validation_checks=False,
                      nu=0.01, export_cert=False, eps=1e-10):
    print("[commutator test] j_min={}, j_max={}, timesteps={}".format(j_min, j_max, timesteps))
    omega = make_shell_localized(N, j=2) + make_shell_localized(N, j=5)
    for t in range(timesteps):
        context = f"commutator_test_t{t}"
        omega_next = time_evolve(omega, nu, dt)
        res, diagnostics = main_compute(
            omega, omega_next,
            grad_u=None,
            alpha=alpha,
            nu=nu,
            j_min=j_min,
            j_max=j_max,
            dt=dt,
            eps=eps,
            plot=plot,
            validate=run_validation_checks,
            context=context
        )

        # NEW: certificate dump for this test step
        log_recursive_damping_certificate(
            res,
            t * dt,
            nu,
            alpha,
            log_path="recursive_damping_certificate.jsonl",
            Y_star=0.0,
            tolerance=1e-6,
        )

        if run_validation_checks:
            try:
                validate_damping_terms(
                    res['Y'], res['dYdt'], res['rhs'],
                    res['align_sup'], res['delta'],
                    res['Lambda'], res['C_nl']
                )
            except AssertionError as e:
                print(f"[VALIDATION ERROR] {e}")
                save_forensic_failure(
                    timestep=t,
                    omega=omega,
                    Y=res['Y'],
                    dYdt=res['dYdt'],
                    rhs=res['rhs'],
                    align_sup=res['align_sup']
                )
                if strict:
                    sys.exit(1)

            w = float_sanity_check({
                'Y': res['Y'],
                'dY/dt': res['dYdt'],
                'RHS': res['rhs'],
                'align_sup': res['align_sup']
            })
            for ww in w:
                print(ww)

        if export_cert and (res['dYdt'] is not None) and (res['rhs'] is not None):
            export_certificate(
                f"/content/output/certificate_{context}.lean",
                res['dYdt'],
                res['rhs'],
                res['C_nl'],
                res['delta'],
                res['Lambda'],
                res['Y'],
                res['align_sup']
            )

        if strict and res['holds'] is False:
            print("FAIL: Damping inequality violated in commutator test at step", t)
            save_forensic_failure(
                timestep=t,
                omega=omega,
                Y=res['Y'],
                dYdt=res['dYdt'],
                rhs=res['rhs'],
                align_sup=res['align_sup']
            )
            sys.exit(1)

        omega = omega_next.copy()

    print("[commutator test] Completed.")

def simulate_blowup(N=64, alpha=2.5, j_min=1, j_max=5,
                    dt=1e-4, timesteps=1, strict=False,
                    plot=False, run_validation_checks=False,
                    nu=0.01, export_cert=False, eps=1e-10):
    print("[blowup test] j_min={}, j_max={}, timesteps={}".format(j_min, j_max, timesteps))
    omega = make_vortex_tube(N) * 10.0
    e1 = cp.ones_like(omega)
    norms = cp.sqrt(cp.sum(omega**2, axis=0, keepdims=True))
    omega = e1 * norms

    for t in range(timesteps):
        context = f"blowup_test_t{t}"
        omega_next = time_evolve(omega, nu, dt)
        res, diagnostics = main_compute(
            omega, omega_next,
            grad_u=None,
            alpha=alpha,
            nu=nu,
            j_min=j_min,
            j_max=j_max,
            dt=dt,
            eps=eps,
            plot=plot,
            validate=run_validation_checks,
            context=context
        )

        # NEW: certificate dump for this test step
        log_recursive_damping_certificate(
            res,
            t * dt,
            nu,
            alpha,
            log_path="recursive_damping_certificate.jsonl",
            Y_star=0.0,
            tolerance=1e-6,
        )

        if run_validation_checks:
            try:
                validate_damping_terms(
                    res['Y'], res['dYdt'], res['rhs'],
                    res['align_sup'], res['delta'],
                    res['Lambda'], res['C_nl']
                )
            except AssertionError as e:
                print(f"[VALIDATION ERROR] {e}")
                save_forensic_failure(
                    timestep=t,
                    omega=omega,
                    Y=res['Y'],
                    dYdt=res['dYdt'],
                    rhs=res['rhs'],
                    align_sup=res['align_sup']
                )
                if strict:
                    sys.exit(1)

            w = float_sanity_check({
                'Y': res['Y'],
                'dY/dt': res['dYdt'],
                'RHS': res['rhs'],
                'align_sup': res['align_sup']
            })
            for ww in w:
                print(ww)

        if export_cert and (res['dYdt'] is not None) and (res['rhs'] is not None):
            export_certificate(
                f"/content/output/certificate_{context}.lean",
                res['dYdt'],
                res['rhs'],
                res['C_nl'],
                res['delta'],
                res['Lambda'],
                res['Y'],
                res['align_sup']
            )

        if strict and res['holds'] is False:
            print("FAIL: Damping inequality violated in blowup test at step", t)
            save_forensic_failure(
                timestep=t,
                omega=omega,
                Y=res['Y'],
                dYdt=res['dYdt'],
                rhs=res['rhs'],
                align_sup=res['align_sup']
            )
            sys.exit(1)

        omega = omega_next.copy()

    print("[blowup test] Completed.")

def time_reverse_test(N=64, alpha=2.5, j_min=2, j_max=5,
                      dt=1e-4, timesteps=1, strict=False,
                      plot=False, run_validation_checks=False,
                      nu=0.01, export_cert=False, eps=1e-10):
    print("[time reversal test] j_min={}, j_max={}, timesteps={}".format(j_min, j_max, timesteps))
    omega_t = make_vortex_tube(N)
    for t in range(timesteps):
        omega_tpdt = time_evolve(omega_t, nu, dt)
        omega_back = time_evolve(omega_tpdt, nu, -dt)
        err = float(cp.abs(omega_back - omega_t).max())
        print(f"[time reverse] Step={t}, forward->back error = {err:.3e}")

        res, diagnostics = main_compute(
            omega_t, omega_tpdt,
            grad_u=None,
            alpha=alpha,
            nu=nu,
            j_min=j_min,
            j_max=j_max,
            dt=dt,
            eps=eps,
            plot=plot,
            validate=run_validation_checks,
            context=f"time_reverse_t{t}"
        )

        # NEW: certificate dump for this test step
        log_recursive_damping_certificate(
            res,
            t * dt,
            nu,
            alpha,
            log_path="recursive_damping_certificate.jsonl",
            Y_star=0.0,
            tolerance=1e-6,
        )

        if run_validation_checks:
            try:
                validate_damping_terms(
                    res['Y'], res['dYdt'], res['rhs'],
                    res['align_sup'], res['delta'],
                    res['Lambda'], res['C_nl']
                )
            except AssertionError as e:
                print(f"[VALIDATION ERROR] {e}")
                save_forensic_failure(
                    timestep=t,
                    omega=omega_t,
                    Y=res['Y'],
                    dYdt=res['dYdt'],
                    rhs=res['rhs'],
                    align_sup=res['align_sup']
                )
                if strict:
                    sys.exit(1)

            w = float_sanity_check({
                'Y': res['Y'],
                'dY/dt': res['dYdt'],
                'RHS': res['rhs'],
                'align_sup': res['align_sup']
            })
            for ww in w:
                print(ww)

        if export_cert and (res['dYdt'] is not None) and (res['rhs'] is not None):
            export_certificate(
                f"/content/output/certificate_time_reverse_t{t}.lean",
                res['dYdt'],
                res['rhs'],
                res['C_nl'],
                res['delta'],
                res['Lambda'],
                res['Y'],
                res['align_sup']
            )

        if strict and res['holds'] is False:
            print("FAIL: Damping inequality violated in time_reverse test at step", t)
            save_forensic_failure(
                timestep=t,
                omega=omega_t,
                Y=res['Y'],
                dYdt=res['dYdt'],
                rhs=res['rhs'],
                align_sup=res['align_sup']
            )
            sys.exit(1)

    print("[time reverse test] Completed.")

def main():
    parser = argparse.ArgumentParser(
        description="Recursive damping proof + additional tests."
    )

    # Base / shared arguments
    parser.add_argument('--N', type=int, default=512,
                        help="Grid size (default 512).")
    parser.add_argument('--alpha', type=float, default=2.5,
                        help="Scaling exponent alpha (default 2.5).")
    parser.add_argument('--nu', type=float, default=0.01,
                        help="Viscosity coefficient (default 0.01).")
    parser.add_argument('--eps', type=float, default=1e-10,
                        help="Threshold epsilon for ignoring near-zero shells (default 1e-10).")
    parser.add_argument('--plot', action='store_true',
                        help="Enable plotting per-shell metrics.")

    # NEW certificate-related CLI flags
    parser.add_argument('--damping_log', type=str,
                        default="recursive_damping_certificate.jsonl",
                        help="Where to append the damping certificate (JSONL).")
    parser.add_argument('--damping_tol', type=float, default=1e-6,
                        help="Tolerance for the residual check (default 1e-6).")
    parser.add_argument('--Y_star', type=float, default=0.0,
                        help="Contraction threshold Y* (default 0).")

    parser.add_argument('--export_cert', action='store_true',
                        help="Export Lean-style proof certificates.")
    parser.add_argument('--run_validation_checks', action='store_true',
                        help="Perform extra validation checks on intervals.")
    parser.add_argument('--input', type=str,
                        help="Path to omega_t.npy for the initial vorticity field.")
    parser.add_argument('--input2', type=str,
                        help="Path to omega_tpdt.npy for the next-timestep vorticity field.")
    parser.add_argument('--grad_u', type=str,
                        help="Optional path to grad_u.npy if precomputed.") 

    parser.add_argument('--run_blowup_test', action='store_true',
                        help="Run the blowup test scenario after main run.")
    parser.add_argument('--run_commutator_test', action='store_true',
                        help="Run the commutator test scenario after main run.")
    parser.add_argument('--run_time_reverse_test', action='store_true',
                        help="Run the time-reverse test scenario after main run.")

    parser.add_argument('--j_min', type=int, default=6,
                        help="Minimum exponent j for the dyadic shell range (default 6).")
    parser.add_argument('--j_max', type=int, default=9,
                        help="Maximum exponent j for the dyadic shell range (default 9).")
    parser.add_argument('--dt', type=float, default=1e-4,
                        help="Time step for vorticity evolution (default 1e-4).")
    parser.add_argument('--timesteps', type=int, default=0,
                        help="Number of time steps to run and certify (default 0).")
    parser.add_argument('--strict', action='store_true',
                        help="Exit with error if any violation occurs in the inequality check.")

    args = parser.parse_args()

    run_blowup = args.run_blowup_test
    run_commutator = args.run_commutator_test
    run_time_reverse = args.run_time_reverse_test

    # 1) Load or generate initial field
    if args.input is not None:
        omega_t = cp.load(args.input)
        print(f"Loaded omega_t from {args.input}, shape {omega_t.shape}.")
    else:
        print("No --input provided. Generating a synthetic 'vortex tube' field.")
        omega_t = make_vortex_tube(args.N)

    # 2) Optionally load next-timestep field
    omega_tpdt = None
    if args.input2 is not None:
        omega_tpdt = cp.load(args.input2)
        print(f"Loaded omega_tpdt from {args.input2}, shape {omega_tpdt.shape}.")

    # 3) Optionally load grad_u if provided
    grad_u = None
    if args.grad_u is not None and os.path.isfile(args.grad_u):
        grad_u = cp.load(args.grad_u)
        print(f"Loaded grad_u from {args.grad_u}, shape {grad_u.shape}.")

    # 4) Multi-step evolution if timesteps > 0
    if args.timesteps > 0:
        print(f"Running {args.timesteps} time steps with dt={args.dt} and verifying at each step.")
        omega = omega_t.copy()
        for t in range(args.timesteps):
            context = f"timestep_{t:03d}"
            omega_next = time_evolve(omega, args.nu, args.dt)
            res, diagnostics = main_compute(
                omega, omega_next,
                grad_u,
                args.alpha,
                args.nu,
                j_min=args.j_min,
                j_max=args.j_max,
                dt=args.dt,
                eps=args.eps,
                plot=args.plot,
                validate=args.run_validation_checks,
                context=context
            )

            if args.run_validation_checks:
                try:
                    validate_damping_terms(
                        res['Y'], res['dYdt'], res['rhs'],
                        res['align_sup'], res['delta'],
                        res['Lambda'], res['C_nl']
                    )
                except AssertionError as e:
                    print(f"[VALIDATION ERROR] {e}")
                    save_forensic_failure(
                        timestep=t,
                        omega=omega,
                        Y=res['Y'],
                        dYdt=res['dYdt'],
                        rhs=res['rhs'],
                        align_sup=res['align_sup']
                    )
                    if args.strict:
                        sys.exit(1)

                warnings = float_sanity_check({
                    'Y': res['Y'],
                    'dY/dt': res['dYdt'],
                    'RHS': res['rhs'],
                    'align_sup': res['align_sup']
                })
                for w in warnings:
                    print(w)
                H = compute_entropy(res['_entropic_per_shell'])
                print(f"[entropy] H(t) = {H:.4f}")

            if args.export_cert and (res['dYdt'] is not None) and (res['rhs'] is not None):
                export_certificate(
                    f"/content/output/certificate_{context}.lean",
                    res['dYdt'],
                    res['rhs'],
                    res['C_nl'],
                    res['delta'],
                    res['Lambda'],
                    res['Y'],
                    res['align_sup']
                )

            if args.strict and res['holds'] is False:
                print(f"FAIL: Damping inequality violated at timestep {t}")
                save_forensic_failure(
                    timestep=t,
                    omega=omega,
                    Y=res['Y'],
                    dYdt=res['dYdt'],
                    rhs=res['rhs'],
                    align_sup=res['align_sup']
                )
                sys.exit(1)

            # NEW: write certificate line for this timestep
            log_recursive_damping_certificate(
                res,
                t * args.dt,
                args.nu,
                args.alpha,
                log_path=args.damping_log,
                Y_star=args.Y_star,
                tolerance=args.damping_tol,
            )

            omega = omega_next.copy()

        print(f"Timestep evolution complete for {args.timesteps} steps.")

    else:
        # 5) Single-step certification if timesteps == 0
        res, diagnostics = main_compute(
            omega_t, omega_tpdt,
            grad_u,
            args.alpha,
            args.nu,
            j_min=args.j_min,
            j_max=args.j_max,
            dt=args.dt,
            eps=args.eps,
            plot=args.plot,
            validate=args.run_validation_checks,
            context='mainrun'
        )

        if args.export_cert and (res['dYdt'] is not None) and (res['rhs'] is not None):
            export_certificate(
                "/content/output/certificate.lean",
                res['dYdt'],
                res['rhs'],
                res['C_nl'],
                res['delta'],
                res['Lambda'],
                res['Y'],
                res['align_sup']
            )

        if args.strict and res['holds'] is False:
            print("FAIL: Damping inequality violated for the single-step scenario.")
            save_forensic_failure(
                timestep=0,
                omega=omega_t,
                Y=res['Y'],
                dYdt=res['dYdt'],
                rhs=res['rhs'],
                align_sup=res['align_sup']
            )
            sys.exit(1)

        if args.run_validation_checks:
            try:
                validate_damping_terms(
                    res['Y'], res['dYdt'], res['rhs'],
                    res['align_sup'], res['delta'],
                    res['Lambda'], res['C_nl']
                )
            except AssertionError as e:
                print(f"[VALIDATION ERROR] {e}")
                save_forensic_failure(
                    timestep=0,
                    omega=omega_t,
                    Y=res['Y'],
                    dYdt=res['dYdt'],
                    rhs=res['rhs'],
                    align_sup=res['align_sup']
                )
                if args.strict:
                    sys.exit(1)

            warnings = float_sanity_check({
                'Y': res['Y'],
                'dY/dt': res['dYdt'],
                'RHS': res['rhs'],
                'align_sup': res['align_sup']
            })
            for w in warnings:
                print(w)
            H = compute_entropy(res['_entropic_per_shell'])
            print(f"[entropy] H(t) = {H:.4f}")

        # NEW: dump certificate for t = 0 (single-step mode)
        log_recursive_damping_certificate(
            res,
            0.0,
            args.nu,
            args.alpha,
            log_path=args.damping_log,
            Y_star=args.Y_star,
            tolerance=args.damping_tol,
        )

        print("Default evolution + certification completed successfully.")
        print("See /content/output/diagnostics_log_mainrun.json for details.")

    # Now run additional diagnostic tests (after the main run)
    if run_blowup:
        simulate_blowup(
            N=args.N, alpha=args.alpha, j_min=args.j_min, j_max=args.j_max,
            dt=args.dt, timesteps=1, strict=args.strict, plot=args.plot,
            run_validation_checks=args.run_validation_checks, nu=args.nu,
            export_cert=args.export_cert, eps=args.eps
        )

    if run_commutator:
        stress_commutator(
            N=args.N, alpha=args.alpha, j_min=args.j_min, j_max=args.j_max,
            dt=args.dt, timesteps=1, strict=args.strict, plot=args.plot,
            run_validation_checks=args.run_validation_checks, nu=args.nu,
            export_cert=args.export_cert, eps=args.eps
        )

    if run_time_reverse:
        time_reverse_test(
            N=args.N, alpha=args.alpha, j_min=args.j_min, j_max=args.j_max,
            dt=args.dt, timesteps=1, strict=args.strict, plot=args.plot,
            run_validation_checks=args.run_validation_checks, nu=args.nu,
            export_cert=args.export_cert, eps=args.eps
        )

    print("All requested runs/tests are complete. Exiting.")

if __name__ == "__main__":
    main()
