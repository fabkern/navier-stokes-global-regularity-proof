%%writefile lean_validator.py
#!/usr/bin/env python3

import os
import sys
import re
import csv
import json
import glob
import hashlib
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# =================== Helper Functions ===================

def sha256file(fn):
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def md5file(fn):
    h = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def get_lean_version():
    try:
        lean_version = subprocess.getoutput("lean --version").strip()
        if lean_version:
            return lean_version
    except Exception:
        pass
    return "Lean unavailable (syntax-only mode)"

def run_lean_check(lean_path):
    try:
        completed = subprocess.run(['lean', '--check', lean_path], capture_output=True, text=True, timeout=15)
        return completed.returncode, completed.stdout, completed.stderr
    except Exception as ex:
        return -99, "", f"EXCEPTION: {ex}"

def parse_theorem_header(contents):
    """
    Extract theorem name, type ("false" or "true"), and statement location.
    """
    header_pattern = re.compile(r'theorem\s+([A-Za-z0-9_]+)\s*:\s*([^:]+):=\s*by', re.S)
    m = header_pattern.search(contents)
    if not m:
        # fallback: more flexible for Lean 4 theorem syntax
        header_pattern2 = re.compile(r'theorem\s+([A-Za-z0-9_]+)\s*:\s*(.*?)\s*:=\s*by', re.S)
        m = header_pattern2.search(contents)
    if m:
        name = m.group(1).strip()
        thm_type = m.group(2).strip()
        is_false = "false" in thm_type
        is_true = "true" in thm_type
        start = m.end()
        return name, (is_false, is_true), start
    return None, (None, None), -1

def extract_proof_body(contents):
    """
    Everything after ':= by'
    """
    sp = contents.split(":= by",1)
    if len(sp)<2:
        return ""
    return sp[1].strip()

def count_proof_steps(body):
    return len([line for line in body.splitlines() if line.strip() and not line.strip().startswith("--")])

def classify_lean_error(stderr):
    if not stderr:
        return None
    if "unknown identifier" in stderr.lower():
        return "Unknown Identifier"
    if "syntax error" in stderr.lower():
        return "Syntax Error"
    if "contradiction" in stderr.lower():
        return "Contradiction"
    if "timeout" in stderr.lower():
        return "Timeout/Tactic Fail"
    if "error" in stderr.lower():
        return "General Error"
    return "Unknown"

def load_json_if_exists(path):
    if not os.path.exists(path): return None
    with open(path,"r") as f:
        return json.load(f)

def smart_str(x):
    if isinstance(x, str): return x
    if isinstance(x, (list, dict)): return json.dumps(x)
    return str(x)

# =================== Main Validation ====================

def main():
    parser = argparse.ArgumentParser(description="Lean certificate validator and proof auditor")
    parser.add_argument("--dir", type=str, required=True, help="Directory with .lean files")
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--check_duplicate_ast", action="store_true", default=False)
    args = parser.parse_args()

    root = args.dir
    lean_files = sorted(glob.glob(os.path.join(root, "**/*.lean"), recursive=True))

    lean_version = get_lean_version()
    print(f"Lean version: {lean_version}")

    results = []
    context_proofbody_map = dict()
    ast_hash_map = defaultdict(list)
    duplicates = []
    vacuous = []
    config_mismatches = []
    fail_compiles = []
    lean_hash_set = set()

    for lf in lean_files:
        d = {}
        d['lean_path'] = lf
        d['lean_sha256'] = sha256file(lf)
        d['lean_md5'] = md5file(lf)
        lean_hash_set.add(d['lean_sha256'])
        with open(lf,"r") as f:
            contents = f.read()
        d['lean_size_bytes'] = len(contents.encode())
        theorem_name, (is_false, is_true), thmbody_start = parse_theorem_header(contents)
        d['theorem_name'] = theorem_name
        d['proves_false'] = bool(is_false)
        d['proves_true'] = bool(is_true)
        proof_body = extract_proof_body(contents)
        d['proof_lines'] = count_proof_steps(proof_body)
        d['proof_body_ast_hash'] = hashlib.sha256(proof_body.encode()).hexdigest()
        if args.check_duplicate_ast:
            ast_hash_map[d['proof_body_ast_hash']].append((lf,theorem_name))
        d['lean_version'] = lean_version
        # Trivial/vacuous proof check:
        tauto_match = re.match(r'^\s*(contradiction|trivial)\s*$', proof_body,re.I)
        d['is_trivial'] = bool(tauto_match) or \
            (d['proves_true'] and d['proof_lines']<=1)
        # Associated config
        cfg_path = lf.replace(".lean", "_config.json")
        config = load_json_if_exists(cfg_path)
        d['config_path'] = cfg_path if config else None
        d['has_config'] = bool(config)
        context_hash = config.get("context_full") or config.get("context") if config else None
        d["context_config"] = context_hash
        d['config_field_type'] = config.get("type") if config else "unknown"
        d['config_seed'] = smart_str(config.get("params",{}).get("seed")) if config else ""
        d['config_hash'] = config.get("hash_sha256") or config.get("cx_hash") if config else None
        # RIS scoring
        RIS = 0
        # 1. Compiles under Lean
        rc, out, err = run_lean_check(lf)
        d['lean_compile_returncode'] = rc
        d['lean_stdout'] = out
        d['lean_stderr'] = err
        d['lean_stderr_hash'] = hashlib.sha256((err or "").encode()).hexdigest()
        d['lean_check_status'] = ("PASS" if rc==0 else "FAIL")
        if rc==0:
            RIS += 1
        else:
            fail_compiles.append(lf)
        # 2. Config exists and matches SHA (if possible)
        if config and (
            d['lean_sha256'] == d['config_hash'] or not d['config_hash']
        ):
            RIS += 1
        else:
            config_mismatches.append(lf)
        # 3. Theorem name includes context hash
        if theorem_name and context_hash and context_hash in theorem_name:
            RIS += 1
        else:
            config_mismatches.append(lf)
        # 4. Proof is not trivial/vacuous
        if not d['is_trivial'] and d['proves_false']:
            RIS += 1
        else:
            vacuous.append(lf)
        # 5. Unique proof body
        is_unique = (ast_hash_map[d['proof_body_ast_hash']]==[(lf, theorem_name)]) if args.check_duplicate_ast else True
        if is_unique:
            RIS += 1
        else:
            duplicates.append(lf)
        d['RIS'] = RIS

        # Redundancy check (populate later)
        context_proofbody_map[(context_hash or theorem_name)] = d['proof_body_ast_hash']
        # Collect theorem duplication
        results.append(d)

    # After scan: detect duplicate ASTs (ignore self)
    if args.check_duplicate_ast:
        for proof_hash, lst in ast_hash_map.items():
            if len(lst)>1:
                for lf, thm in lst:
                    for d in results:
                        if d['lean_path']==lf:
                            d['is_duplicate_ast'] = True
                duplicates.extend([x[0] for x in lst if x[0] not in duplicates])

    # ==== Export .csv ====
    summary_csv = os.path.join(root, "lean_validation_summary.csv")
    fieldnames = [
        "lean_path","lean_sha256","lean_md5","theorem_name",
        "proves_false","proves_true","proof_lines","proof_body_ast_hash",
        "lean_version","has_config","context_config","config_field_type",
        "config_seed","config_hash","lean_check_status","RIS"
    ]
    with open(summary_csv,"w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=fieldnames)
        w.writeheader()
        for d in results:
            w.writerow({k:smart_str(d.get(k,"")) for k in fieldnames})

    # ==== Export .md report ====
    summary_md = os.path.join(root, "lean_validation_report.md")
    with open(summary_md,"w") as f:
        f.write(f"# Lean Certificate Validation Report\n")
        f.write(f"Audit time: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Target directory: `{root}`\n")
        f.write(f"Lean version: {lean_version}\n\n")
        f.write(f"Total certificates: {len(results)}\n\n")
        f.write("## Failures to compile:\n")
        for fn in fail_compiles:
            f.write(f"- {fn}\n")
        f.write("\n## Config mismatches or theorem/context inconsistency:\n")
        for fn in config_mismatches:
            f.write(f"- {fn}\n")
        f.write("\n## Vacuous/trivial certificates:\n")
        for fn in vacuous:
            f.write(f"- {fn}\n")
        if args.check_duplicate_ast:
            f.write("\n## Duplicate AST bodies detected (possible redundancy):\n")
            for fn in duplicates:
                f.write(f"- {fn}\n")
        f.write("\n## RIS Score Histogram\n")
        ris_hist = defaultdict(int)
        for d in results:
            ris_hist[d['RIS']] += 1
        for k in sorted(ris_hist):
            f.write(f"- RIS={k}: {ris_hist[k]}\n")
        f.write("\n## Certificates with RIS < 5\n")
        for d in results:
            if d['RIS']<5:
                f.write(f"- {d['lean_path']} (RIS={d['RIS']})\n")
        f.write("\n\nDetail per certificate is available in the CSV.")

    # ==== Export .json metadata ====
    summary_json = os.path.join(root, "lean_validator_metadata.json")
    for d in results:
        # Truncate large keys to keep JSON readable
        if 'lean_stdout' in d and len(d['lean_stdout'])>500: d['lean_stdout'] = d['lean_stdout'][:500] + "···"
        if 'lean_stderr' in d and len(d['lean_stderr'])>500: d['lean_stderr'] = d['lean_stderr'][:500] + "···"
    with open(summary_json, "w") as f:
        json.dump({"certificates": results, "lean_version": lean_version, "timestamp": datetime.utcnow().isoformat()}, f, indent=2)

    print(f"\nSummary:")
    print(f"- CSV: {summary_csv}")
    print(f"- Markdown report: {summary_md}")
    print(f"- Metadata: {summary_json}")
    if args.strict and (fail_compiles or duplicates or vacuous or config_mismatches):
        print("Strict mode: at least one error detected. Failing.")
        sys.exit(99)
    print("Done.")

if __name__ == "__main__":
    main()
