#!/usr/bin/env python3
import os
import sys
import hashlib
import json
import time
import socket
import argparse
import numpy as np
from datetime import datetime
from functools import wraps
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ================= Registry & Decorators ====================

class FieldRegistry:
    def __init__(self):
        self.generators = {}
    def register(self, name):
        def wrapper(func):
            self.generators[name] = func
            return func
        return wrapper
    def generate(self, name, **params):
        if name not in self.generators:
            raise ValueError(f"Unknown field type: {name}")
        return self.generators[name](**params)

field_registry = FieldRegistry()

def validate_field(fn):
    """Decorator for generator functions to check output shape, finiteness, divergence."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        omega = fn(*args, **kwargs)
        if omega.shape[0] != 3 or len(omega.shape) != 4:
            raise ValueError(f"Field shape invalid: expected (3,N,N,N), got {omega.shape}")
        if not np.isfinite(omega).all():
            raise ValueError("Field contains inf/nan values")
        if np.linalg.norm(omega) < 1e-8:
            raise ValueError("Field norm too small")
        # Divergence-free test (optional unless forced)
        if kwargs.get('enforce_divergence', True):
            N = omega.shape[1]
            dx = 1.0/N
            div = (
                np.gradient(omega[0], dx, axis=0) +
                np.gradient(omega[1], dx, axis=1) +
                np.gradient(omega[2], dx, axis=2)
            )
            if np.abs(div).mean() >= 1e-3:
                raise ValueError("Field failed divergence test: mean(abs(div)) >= 1e-3")
        return omega
    return wrapper

# ============= Field Generators =========================

@field_registry.register("shell_random")
@validate_field
def shell_random(j=8, N=512, seed=42, amplitude=1.0, **_):
    """Divergence-free field with energy in shell |k| ~ 2^j."""
    np.random.seed(seed)
    F = np.zeros((3, N, N, N), dtype=np.complex64)
    freq = np.fft.fftfreq(N)*N
    Kx, Ky, Kz = np.meshgrid(freq, freq, freq, indexing='ij')
    K2 = Kx**2 + Ky**2 + Kz**2
    mask = (K2 >= 2**(2*j)) & (K2 < 2**(2*(j+1)))
    if not np.any(mask):
        max_freq = np.max(np.abs(freq))
        j_adj = int(np.floor(np.log2(max_freq)))
        mask = (K2 >= 2**(2*j_adj)) & (K2 < 2**(2*(j_adj+1)))
    idx = np.nonzero(mask)
    if idx[0].size > 0:
        kvec = np.vstack([Kx[idx], Ky[idx], Kz[idx]])  # shape (3, M)
        norm_k = np.linalg.norm(kvec, axis=0)
        valid = norm_k >= 1e-7
        if np.any(valid):
            kvec_valid = kvec[:, valid]
            norm_k_valid = norm_k[valid]
            k_unit = kvec_valid / norm_k_valid  # shape (3, M_valid)
            M_valid = k_unit.shape[1]
            u = np.random.randn(3, M_valid) + 1j * np.random.randn(3, M_valid)
            dot = np.sum(u * k_unit, axis=0)
            u = u - k_unit * dot
            indices = np.array(np.nonzero(mask)).T  # shape (M, 3)
            indices_valid = indices[valid]
            F[:, indices_valid[:,0], indices_valid[:,1], indices_valid[:,2]] = u
    norm_F = np.sqrt((np.abs(F)**2).sum())
    if norm_F == 0:
        raise ValueError("No energy in selected Fourier modes; adjust 'j' or 'N'.")
    F /= norm_F
    F *= amplitude
    omega = np.fft.ifftn(F, axes=(1,2,3)).real.astype(np.float32)
    return omega

@field_registry.register("multi_shell")
@validate_field
def multi_shell(j1=6, j2=9, N=512, seed=888, amplitude=1.0, **_):
    """Superpose two shells."""
    a = shell_random(j=j1, N=N, seed=seed, amplitude=amplitude/2, enforce_divergence=False)
    b = shell_random(j=j2, N=N, seed=seed+1, amplitude=amplitude/2, enforce_divergence=False)
    omega = a + b
    return omega

@field_registry.register("aligned_noise")
@validate_field
def aligned_noise(N=512, seed=123, amplitude=1.0, **_):
    np.random.seed(seed)
    omega = np.random.randn(3, N, N, N)
    v = omega.reshape(3,-1).mean(axis=1)
    if np.linalg.norm(v) > 1e-8:
        theta = np.arccos(np.dot(v, [1,0,0])/np.linalg.norm(v))
        if theta > 1e-5:
            axis = np.cross(v, [1,0,0])
            axis = axis/np.linalg.norm(axis)
            from scipy.spatial.transform import Rotation as R
            rot = R.from_rotvec(axis*theta)
            omega = np.tensordot(rot.as_matrix(), omega, axes=([1],[0]))
    omega *= amplitude / np.sqrt(np.mean(omega**2)+1e-12)
    return omega.astype(np.float32)

@field_registry.register("white_noise")
@validate_field
def white_noise(N=512, seed=77, amplitude=1.0, **_):
    np.random.seed(seed)
    omega = amplitude * np.random.randn(3, N, N, N).astype(np.float32)
    return omega

@field_registry.register("vortex_tube")
@validate_field
def vortex_tube(N=512, amplitude=1.0, **_):
    x = np.linspace(-1,1,N,endpoint=False)
    X,Y,Z = np.meshgrid(x,x,x,indexing='ij')
    r = np.sqrt(Y**2+Z**2)
    tube = amplitude * np.exp(-30*r**2)
    omega = np.zeros((3,N,N,N), dtype=np.float32)
    omega[0] = tube
    return omega

@field_registry.register("boundary_layer")
@validate_field
def boundary_layer(N=512, amplitude=1.0, **_):
    x = np.linspace(-1,1,N,endpoint=False)
    Z = np.meshgrid(x,x,x,indexing='ij')[2]
    layer = amplitude * np.exp(-200*Z**2)
    omega = np.zeros((3,N,N,N), dtype=np.float32)
    omega[1] = layer
    return omega

@field_registry.register("adversarial_mix")
@validate_field
def adversarial_mix(N=512, j=3, j2=6, seed=444, amplitude=1.0, **_):
    field1 = shell_random(j=j, N=N, seed=seed, amplitude=amplitude/2, enforce_divergence=False)
    field2 = white_noise(N=N, seed=seed+993, amplitude=amplitude/4, enforce_divergence=False)
    field3 = vortex_tube(N=N, amplitude=amplitude/4, enforce_divergence=False)
    omega = field1 + field2 + field3
    return omega

# ============ Evolution Stepper =======================

def proof_evolve(omega_t, dt=1e-4, magnitude=0.01, seed=2023):
    np.random.seed(hash(float(np.sum(omega_t))+dt+seed) % (2**32-1))
    noise = magnitude * np.random.randn(*omega_t.shape)
    return omega_t + dt*noise

# ========== Hashing & Metadata ========================

def get_sha256(arr):
    m = hashlib.sha256()
    m.update(arr.tobytes())
    return m.hexdigest()

def get_md5(arr):
    m = hashlib.md5()
    m.update(arr.astype(np.float32).tobytes())
    return m.hexdigest()

def get_src_hash(fn):
    co = fn.__code__
    code_bytes = co.co_code
    h = hashlib.sha1(code_bytes).hexdigest()
    return h

def get_git_commit():
    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        return commit
    except Exception:
        return "unknown"

# ========== CLI & Main ===============================

def visualize_field(omega, out_fn="preview_slice.png"):
    import matplotlib.pyplot as plt
    N = omega.shape[1]
    mid = N//2
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    for i, comp in enumerate(['x', 'y', 'z']):
        im = axs[i].imshow(omega[i, mid], cmap='RdBu', vmax=np.abs(omega[i, mid]).max())
        axs[i].set_title(f"$\\omega_{comp}$ (z={mid})")
        plt.colorbar(im, ax=axs[i])
    plt.tight_layout()
    plt.savefig(out_fn)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate reproducible random vorticity fields for proof engine.")
    parser.add_argument('--type', type=str, required=True, help="Field type")
    parser.add_argument('--j', type=int, help="Shell/Multi")
    parser.add_argument('--j1', type=int)
    parser.add_argument('--j2', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amplitude', type=float, default=1.0)
    parser.add_argument('--N', type=int, default=512)
    parser.add_argument('--out', type=str, default="/content/omega_t.npy")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evolve', action='store_true')
    parser.add_argument('--export_config', action='store_true')
    args = parser.parse_args()

    params = {}
    argsvars = vars(args)
    for k in argsvars:
        v = argsvars[k]
        if k in {'visualize', 'evolve', 'out', 'export_config', 'type'} or v is None:
            continue
        params[k] = v

    ftype = args.type
    if ftype not in field_registry.generators:
        raise ValueError(f'Field type not recognized: {ftype}')
    fn = field_registry.generators[ftype]
    import inspect
    valid_args = inspect.getfullargspec(fn).args
    fn_params = {k: params[k] for k in params if k in valid_args}

    omega = field_registry.generate(ftype, **fn_params)
    assert omega.shape == (3, args.N, args.N, args.N)
    save_path = args.out
    np.save(save_path, omega)

    sha256 = get_sha256(omega)
    md5 = get_md5(omega)
    src_hash = get_src_hash(fn)
    git_commit = get_git_commit()
    ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    hostname = socket.gethostname()

    if args.visualize and plt is not None:
        out_img = os.path.splitext(save_path)[0] + "_preview.png"
        visualize_field(omega, out_img)

    if args.evolve:
        omega_tpdt = proof_evolve(omega)
        out_tpdt = os.path.splitext(save_path)[0] + "_tpdt.npy"
        np.save(out_tpdt, omega_tpdt)

    if args.export_config:
        config = {
            "type": ftype,
            "params": fn_params,
            "shape": list(omega.shape),
            "hash_sha256": sha256,
            "float_hash_md5": md5,
            "generator_source_hash": src_hash,
            "timestamp": ts,
            "git_commit": git_commit,
            "hostname": hostname
        }
        config_path = os.path.splitext(save_path)[0] + "_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_path}")

    print(f"Field '{ftype}' written to {save_path}")
    print(f"SHA256: {sha256}")
    print(f"MD5(float32): {md5}")

if __name__ == "__main__":
    main()
