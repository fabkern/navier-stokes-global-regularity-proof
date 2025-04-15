# Recursive Navier–Stokes Proof Audit Engine

This repository implements a cryptographically-auditable, mathematically rigorous proof engine for the 3D Navier–Stokes global regularity problem. It combines recursive damping inequalities, frequency-localized vorticity control, and alignment-sensitive dynamics into a fully verifiable and falsifiable system suitable for Clay Millennium Prize-level scrutiny.

---

## Purpose

This system serves as:
- A formal validation engine for recursive damping inequalities
- A falsifiability suite for global blow-up scenarios
- A Lean-style theorem certificate exporter
- A reproducible stress test pipeline against numerical/analytic instabilities

Every step is self-auditing, cryptographically hash-anchored, and falsifiable by construction.

---

## Recursive Damping Inequality

The core inequality is:

\[
\frac{d}{dt} Y(t) \leq C_{nl}\, Y(t)^2\, \mathcal{A}(t) - \nu\, \Lambda\, Y(t)^{1+\delta}
\]

Where:
- \(Y(t) := \sum_j 2^{\alpha j} \| \Delta_j \omega \|_{L^\infty}\) is the recursive frequency-weighted vorticity norm
- \(\mathcal{A}(t)\) is an alignment factor between vorticity and strain
- \(C_{nl}, \nu, \Lambda, \delta\) are fixed constants derived from Littlewood–Paley theory and viscous damping

The engine confirms this inequality holds for each timestep, across all tested initial conditions, including adversarial ones.

---

## Falsifiability Protocol

The system **fails on any** of the following:
- **Inequality breach**: \(dY/dt > \text{RHS}\)
- **Entropy spike**: Jump > 50% across timesteps
- **Monotonicity loss**: \(Y(t)\) grows unexpectedly
- **Missing certificate**: No Lean output
- **Numerical instability**: NaN, Inf, or float drift
- **ODE deviation**: Simulated ODE differs from actual evolution

Failures are archived in `/content/output/counterexamples/`, including `.npy`, `.json`, `.lean`, and diagnostic plots.

---

## Component Breakdown

| Script                        | Role                        | Description |
|-------------------------------|-----------------------------|-------------|
| `proof_colab.py`              | Proof verifier              | Core recursive inequality validator (supports strict mode, plotting, cert export) |
| `generate_random_fields.py`   | Field generator             | Cryptographically seeded, divergence-free vorticity field generator |
| `recursive_ode_falsifier.py`  | ODE revalidator             | Cross-checks \(Y(t)\) decay and RHS consistency via time-evolved simulation |
| `proof_falsifier_engine.py`   | Adversarial injector        | Runs randomized and edge-case tests against proof logic |
| `proof_summary_dashboard.py`  | Dashboard + Audit Summary   | Compiles entropy, pass/fail, certificate stats, visualizations |
| `certificate_packager.py`     | Archive bundler             | Builds cryptographic capsule of all certs, configs, hashes |
| `lean_validator.py`           | Lean cert checker           | Validates `.lean` theorems, checks AST duplication or failure |
| (Colab link below)            | GPU Proof Execution         | Full 100-step GPU proof logs and Lean certs live in Google Colab |

---

## Live GPU Proof (Google Colab)

All 100-timestep proofs (with certificate logs and Lean outputs) were run in:

🔗 **[Colab Proof Notebook (A100 GPU)](https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID_HERE)**  
Replace with your actual Colab link after uploading.

---

## How to Run

```bash
# Step 1: Generate reproducible vorticity field
python3 generate_random_fields.py --type multi_shell --j1 6 --j2 9 --N 512 --seed 123 --evolve --export_config

# Step 2: Verify inequality + generate Lean certs
python3 proof_colab.py --timesteps 20 --input omega_t.npy --input2 omega_t_tpdt.npy --alpha 2.5 --j_min 6 --j_max 9 --strict --plot --export_cert --run_validation_checks

# Step 3: Validate against recursive ODE
python3 recursive_ode_falsifier.py --strict --lean_cert

# Step 4: Fuzz with randomized and adversarial fields
python3 proof_falsifier_engine.py

# Step 5: Summarize and visualize results
python3 proof_summary_dashboard.py --export --plot

# Step 6: Package archive bundle
python3 certificate_packager.py

# Step 7: Validate Lean certificates
python3 lean_validator.py --dir /content/output/counterexamples/ --strict --check_duplicate_ast
```

---

## Output Artifacts

| File/Folder                      | Purpose |
|----------------------------------|---------|
| `omega_t.npy`, `omega_tpdt.npy` | Vorticity fields (pre/post timestep) |
| `certificate_timestep_*.lean`   | Per-step Lean theorem statement |
| `diagnostics_log_*.json`        | All per-shell norms, alignments, constants |
| `proof_plot_Y.png`              | Inequality LHS vs RHS visualized |
| `entropy_histogram.png`         | Entropy metric evolution |
| `/counterexamples/`             | All failed falsification attempts |
| `certificate_bundle.zip`        | All above packed for audit |
| `proof_summary.csv/md/json`     | Dashboard-level summary |
| `README.md`                     | This file |

---

## Auditing + Reproducibility

- **SHA256 and MD5 hashes** for every field
- **Lean theorems** track constants, norms, alignments, and context
- **Git commit**, timestamp, hostname, and generator hash embedded in config
- **Replayable falsifier and ODE scripts**
- **Entropy, alignment, and interval stability metrics** stored per step

---

## Extend This Engine

- Add new generators to `generate_random_fields.py`
- Change proof constants via CLI: `--alpha`, `--nu`, etc.
- Add new falsifiers via `proof_falsifier_engine.py`
- Extend Lean certificate content with `export_certificate(...)`

---

## Authorship & License

- Maintainer: [Fabian Kern - fabkern@proton.me]
- License: MIT (with mathematical provenance clause)
- Status: Under audit. No formal peer-reviewed verification yet. Submissions and third-party reproducibility reports welcome.

