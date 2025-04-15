import os
import sys
import shutil
import subprocess
import time
import hashlib
import json
import numpy as np
import zipfile
import datetime
from pathlib import Path

try:
    import cupy as cp
except ImportError:
    cp = None

####### --- Synthetic Field Generators --- #######

def make_shell_localized(j=3, N=32, seed=42):
    # Shell-localized vorticity (Fourier bump at shell j)
    np.random.seed(seed)
    if cp: cp.random.seed(seed)
    omega = np.zeros((3, N, N, N), dtype=np.float64)
    grid = np.fft.fftfreq(N) * N
    Kx, Ky, Kz = np.meshgrid(grid, grid, grid, indexing='ij')
    K2 = Kx**2 + Ky**2 + Kz**2
    mask = (K2 >= 2**(2*j)) & (K2 < 2**(2*(j+1)))
    for c in range(3):
        noise = np.random.randn(N,N,N)
        fft = np.fft.fftn(noise)
        fft *= mask
        loc = np.fft.ifftn(fft).real
        omega[c] = loc / np.sqrt(np.mean(loc**2)+1e-12)
    return omega

def make_vortex_tube(N=32, amp=1.0):
    # Simple vortex tube along e1
    omega = np.zeros((3, N, N, N), dtype=np.float64)
    x = np.linspace(-1,1,N,endpoint=False)
    X,Y,Z = np.meshgrid(x,x,x,indexing='ij')
    r = np.sqrt(Y**2+Z**2)
    profile = amp * np.exp(-20*r**2)
    omega[0] = profile  # Aligned with e1
    return omega

def align_to_e1(omega):
    # Rotate max-energy direction to e1
    v = omega.reshape(3,-1).mean(axis=1)
    if np.linalg.norm(v) < 1e-8:
        return omega
    e1 = np.array([1,0,0.])
    axis = np.cross(v, e1)
    if np.linalg.norm(axis) < 1e-8:
        return omega
    axis = axis / np.linalg.norm(axis)
    theta = np.arccos(np.dot(v, e1)/(np.linalg.norm(v)*np.linalg.norm(e1)))
    from scipy.spatial.transform import Rotation as R
    Rmat = R.from_rotvec(axis*theta).as_matrix()
    return np.tensordot(Rmat, omega, axes=([1],[0]))

def combine_shells(j1=2, j2=6, N=32):
    # Blend two shell-localized fields
    w1 = make_shell_localized(j=j1,N=N,seed=10*j1)
    w2 = make_shell_localized(j=j2,N=N,seed=10*j2+1)
    return 0.5*w1 + 1.5*w2

def random_gaussian(scale=1e-2, N=32, seed=123):
    np.random.seed(seed)
    omega = scale * np.random.randn(3,N,N,N)
    return omega

def proof_evolve(omega_t, dt=1e-4):
    # Very simple Euler step for demonstration (add some noise)
    omega_tpdt = omega_t + dt * np.random.randn(*omega_t.shape) * 0.01
    return omega_tpdt

######## ---- Field Registry ---- ########

fields = [
  {"name": "shell_j3", "generator": make_shell_localized, "params": {"j":3, "seed":42}},
  {"name": "shell_j5", "generator": make_shell_localized, "params": {"j":5, "seed":43}},
  {"name": "aligned_blowup", "generator": make_vortex_tube, "post": "align_to_e1"},
  {"name": "adversarial_combo", "generator": combine_shells, "params": {"j1":2, "j2":6}},
  {"name": "random_noise", "generator": random_gaussian, "params": {"scale":1e-2, "seed":99}},
]

gen_post_map = {
    "align_to_e1": align_to_e1,
}

######## ---- Utility Functions ---- ########
def get_hash(arr):
    return hashlib.md5(arr.astype(np.float32).tobytes()).hexdigest()[:10]

def save_field(arr, fname):
    np.save(fname, arr)

def load_field(fname):
    return np.load(fname)

def get_git_hash():
    try:
        import subprocess
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/content/", text=True).strip()
        return git_hash
    except:
        return None

def copy_if_exists(src, dst):
    try:
        if os.path.exists(src):
            shutil.copy(src, dst)
            return True
    except Exception as e:
        print(f"Failed to copy {src} to {dst}: {e}")
    return False

def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)

def bundle_counterexamples(counterexample_paths, zip_out="/content/output/counterexample_bundle.zip"):
    with zipfile.ZipFile(zip_out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for cdir in counterexample_paths:
            for fname in Path(cdir).glob('*'):
                zf.write(fname, arcname=f"{Path(cdir).name}/{fname.name}")

######## ---- Metrics & Report ---- ########

summary = {
    "total_runs": 0,
    "proof_colab_failures": 0,
    "ode_falsifier_failures": 0,
    "field_failure_types": [],
    "counterexample_paths": [],
    "entropy_curves": [],
    "run_details": [],
}

######## ---- Main Adversarial Loop ---- ########

def main():
    N = 512
    output_dir = "/content/output/"
    cx_dir = os.path.join(output_dir, "counterexamples/")
    safe_makedirs(output_dir)
    safe_makedirs(cx_dir)
    git_hash = get_git_hash()
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    global_run = 1

    for run_idx, field in enumerate(fields, 1):
        name = field['name']
        params = field.get('params', {})
        gen = field['generator']
        context = f"test_{name}_run{run_idx}"
        seed = params.get('seed', 1000+run_idx) if 'seed' in params else 1000+run_idx
        np.random.seed(seed)
        if cp: cp.random.seed(seed)
        run_info = {
            "index": run_idx,
            "context": context,
            "name": name,
            "params": params,
            "timestamp": timestamp,
            "git_hash": git_hash,
            "seed": seed,
        }
        # --- Generate vorticity field ---
        if 'params' in field:
            omega = gen(**params, N=N) if 'N' in gen.__code__.co_varnames else gen(**params)
        else:
            omega = gen(N=N)
        # Apply any postproc:
        if field.get('post'):
            omega = gen_post_map[field['post']](omega)
        omega_t = omega.astype(np.float32)
        save_field(omega_t, '/content/omega_t.npy')
        omega_tpdt = proof_evolve(omega_t)
        save_field(omega_tpdt, '/content/omega_tpdt.npy')
        cx_hash = get_hash(omega_t)
        context_full = f"{context}_{cx_hash}"
        run_info["context_full"] = context_full
        run_info["cx_hash"] = cx_hash
        # Save config:
        config_json = f"/content/output/{context_full}_config.json"
        with open(config_json,"w") as f:
            json.dump(run_info, f, indent=2)
        proof_log = os.path.join(output_dir, 'diagnostics_log.json')
        proof_cert = os.path.join(output_dir, 'proof_certificate.lean')
        falsifier_log = os.path.join(output_dir, 'falsifier_log.txt')
        falsifier_cert = os.path.join(output_dir, 'falsifier_certificate.lean')
        # --- RUN proof_colab.py ---
        proof_args = [
            sys.executable, "proof_colab.py",
            "--input", "omega_t.npy",
            "--input2", "omega_tpdt.npy",
            "--alpha", "2.5", "--plot", "--export_cert",
            "--run_validation_checks", "--strict",
            "--context", context_full
        ]
        summary["total_runs"] += 1
        proc1 = subprocess.run(proof_args, capture_output=True, cwd="/content/")
        proof_failed = proc1.returncode != 0
        ode_failed = False
        # --- RUN recursive_ode_falsifier.py ---
        if not proof_failed:
            ode_args = [sys.executable, "recursive_ode_falsifier.py", "--strict", "--lean_cert"]
            proc2 = subprocess.run(ode_args, capture_output=True, cwd="/content/")
            ode_failed = proc2.returncode != 0
        # --- Handle any failures ---
        failed = False
        cx_this_run_dir = ""
        if proof_failed or ode_failed:
            failed = True
            fail_reason = []
            cxdir = os.path.join(cx_dir, context_full)
            safe_makedirs(cxdir)
            cx_this_run_dir = cxdir
            safe_makedirs(cxdir)
            shutil.copy('/content/omega_t.npy', os.path.join(cxdir, f"{context_full}_omega_t.npy"))
            shutil.copy('/content/omega_tpdt.npy', os.path.join(cxdir, f"{context_full}_omega_tpdt.npy"))
            copy_if_exists(proof_log, os.path.join(cxdir, f"{context_full}_diagnostics_log.json"))
            copy_if_exists(proof_cert, os.path.join(cxdir, f"{context_full}_proof_certificate.lean"))
            copy_if_exists(falsifier_log, os.path.join(cxdir, f"{context_full}_falsifier_log.txt"))
            copy_if_exists(falsifier_cert, os.path.join(cxdir, f"{context_full}_falsifier_certificate.lean"))
            shutil.copy(config_json, os.path.join(cxdir, f"{context_full}_config.json"))
            with open(os.path.join(cxdir, f"{context_full}_proof_stdout.txt"),"w") as f: f.write(proc1.stdout.decode(errors='ignore'))
            with open(os.path.join(cxdir, f"{context_full}_proof_stderr.txt"),"w") as f: f.write(proc1.stderr.decode(errors='ignore'))
            if not proof_failed:
                with open(os.path.join(cxdir, f"{context_full}_ode_stdout.txt"),"w") as f: f.write(proc2.stdout.decode(errors='ignore'))
                with open(os.path.join(cxdir, f"{context_full}_ode_stderr.txt"),"w") as f: f.write(proc2.stderr.decode(errors='ignore'))
            summary["counterexample_paths"].append(cxdir)
            if proof_failed:
                summary["proof_colab_failures"] += 1
                fail_reason.append('proof_colab')
            if ode_failed:
                summary["ode_falsifier_failures"] += 1
                fail_reason.append('recursive_ode_falsifier')
            summary["field_failure_types"].append(f"{name} ({'/'.join(fail_reason)})")
        # Optionally collect entropy curve:
        try:
            diag_path = os.path.join(output_dir, 'diagnostics_log_timestep_000.json')
            if os.path.exists(diag_path):
                with open(diag_path) as f:
                    diag = json.load(f)
                if '_entropic_per_shell' in diag:
                    ent_arr = [float(s.get('norm',0)) for s in diag['_entropic_per_shell']]
                    summary["entropy_curves"].append({"context":context, "entropy": ent_arr})
        except Exception: pass
        # Save run details
        run_info.update({
            "proof_failed": proof_failed,
            "ode_failed": ode_failed,
            "counterexample_dir": cx_this_run_dir
        })
        summary["run_details"].append(run_info)
        result_tag = "[FAIL]" if failed else "[PASS]"
        print(f"{result_tag} {context_full} | proof_colab: {proc1.returncode} | ode: {('n/a' if proof_failed else proc2.returncode)}")
        sys.stdout.flush()

    # Final bundle zip
    if summary["counterexample_paths"]:
        bundle_counterexamples(summary["counterexample_paths"])

    # Print report
    print("------ Adversarial Falsification Test Report ------")
    print(f"Total runs: {summary['total_runs']}")
    print(f"proof_colab.py failures: {summary['proof_colab_failures']}")
    print(f"recursive_ode_falsifier.py failures: {summary['ode_falsifier_failures']}")
    print(f"Field types causing failure:\n  " + "\n  ".join(sorted(set(summary['field_failure_types']))))
    if summary["entropy_curves"]:
        print("Entropy curves available for analysis (first context shown):")
        print(summary["entropy_curves"][0])
    print(f"Counterexamples saved in: {cx_dir}")
    print(f"Bundle zip (if any): /content/output/counterexample_bundle.zip")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
