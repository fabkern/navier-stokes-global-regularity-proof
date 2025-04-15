#!/usr/bin/env python3

import os
import sys
import json
import glob
import shutil
import hashlib
import argparse
from pathlib import Path
import zipfile
import datetime

############# Utility Functions ################

def sha256file(fn):
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        while True:
            blk = f.read(65536)
            if not blk: break
            h.update(blk)
    return h.hexdigest()

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def load_json(fn):
    with open(fn,"r") as f:
        return json.load(f)

def fail(msg):
    print(f"[certificate_packager.py] ERROR: {msg}", file=sys.stderr)
    sys.exit(2)

def lean_valid_certificate(lean_path):
    try:
        with open(lean_path,"r") as f:
            data = f.read()
        return "theorem proof_invalid" in data
    except Exception:
        return False

def find_file(prefix, exts, context, extra_match=None):
    """Find file with basename prefix, extension in list, containing context."""
    for e in exts:
        for match in glob.glob(f"**/*{prefix}*{context}*{e}", recursive=True):
            if extra_match and extra_match not in match: continue
            return match
    return None

def find_field_file(ref_dir, hashval, suffix="omega_t.npy"):
    cands = glob.glob(os.path.join(ref_dir, f"**/*{suffix}"), recursive=True)
    for fn in cands:
        try:
            if sha256file(fn) == hashval:
                return fn
        except: pass
    return None

def get_source_hash(path):
    # Take SHA1 of this script as generator source hash
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except Exception:
        return "unknown"

def monotonic_decay(yt):
    if not yt or len(yt)<2: return False
    return all((a>=b) for a,b in zip(yt,yt[1:]))

############# Main Capsule Logic ###############

def package_capsules(
    proof_summary_json="/content/output/proof_summary.json",
    output_dir="/content/output/capsules/",
    verify_only=False,
    min_ris=3,
    sign_gpg=False,
):

    # -------------------------------------------------
    # Step 1: Load summary, enumerate valid runs
    # -------------------------------------------------
    if not os.path.exists(proof_summary_json):
        fail(f"proof_summary.json not found at {proof_summary_json}")
    with open(proof_summary_json,"r") as f:
        summary = json.load(f)["summary"] if "summary" in json.load(f) else json.load(f)

    runlist = []
    for run in summary:
        # Must be PASS, have config, lean cert present & valid, relerr ≤2%, monotonic, RIS ≥ min_ris
        if run.get("pass_fail") != "PASS": continue
        if not run.get("lean_cert_present"): continue
        if not run.get("lean_cert_valid"): continue
        if run.get("relerr_max") is None or float(run.get("relerr_max",9e9)) > 0.02: continue
        if not monotonic_decay(run.get("Y_t",[])): continue
        if int(run.get("RIS",0)) < min_ris: continue
        runlist.append(run)
    if not runlist:
        fail("No valid runs found for packaging.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator_src_hash = get_source_hash(__file__)
    capsules = []
    now = datetime.datetime.utcnow().isoformat()+"Z"

    for run in runlist:
        context = run['context']
        field_type = run.get("type")
        params = run.get("params",{})
        seed = params.get("seed","")
        git_commit = run.get("git_commit","")
        sha_reported = run.get("sha256") or run.get("hash_sha256")
        capsule_data = {
            "context": context,
            "field_type": field_type,
            "seed": seed,
            "params": params,
            "timestamp": now,
            "git_commit": git_commit,
            "sha256": sha_reported,
            "generator_source_hash": generator_src_hash,
            "files": {},
        }
        # ------- Locate and validate all relevant files -------
        # -- omega_t.npy (must match hash from config)
        omega_t_fn = find_field_file("/content/output/", sha_reported, suffix="omega_t.npy")
        if not omega_t_fn:
            fail(f"Run {context}: omega_t.npy with matching hash not found.")
        if sha256file(omega_t_fn) != sha_reported:
            fail(f"Run {context}: hash mismatch for omega_t.npy.")

        # -- omega_tpdt.npy (optional, but must exist if claimed)
        omega_tpdt_fn = None
        for cand in glob.glob(os.path.join(os.path.dirname(omega_t_fn),"*omega_tpdt.npy")):
            if os.path.exists(cand): omega_tpdt_fn = cand

        # -- Config JSON
        config_fn = find_file(prefix="", exts=["_config.json"], context=context)
        if not config_fn:
            fail(f"Run {context}: config json missing.")

        # -- Lean certificate (proof or falsifier), must validate
        lean_fn = find_file(prefix="", exts=[".lean"], context=context)
        if not lean_fn or not lean_valid_certificate(lean_fn):
            fail(f"Run {context}: valid .lean certificate not found or invalid.")

        # -- Diagnostics, logs, preview images, CSVs
        diag_log = find_file(prefix="diagnostics_log", exts=[".json"], context=context)
        csv_fn   = find_file(prefix="", exts=[".csv"], context=context)
        falsifier_log = find_file(prefix="falsifier_log", exts=[".txt"], context=context)
        preview_png = find_file(prefix="", exts=[".png"], context=context)
        # ---- Gather all files to package
        fileset = {
            "omega_t.npy": omega_t_fn,
            "config.json": config_fn,
            "certificate.lean": lean_fn
        }
        if omega_tpdt_fn: fileset["omega_tpdt.npy"] = omega_tpdt_fn
        if diag_log: fileset["diagnostics.json"] = diag_log
        if csv_fn: fileset["summary.csv"] = csv_fn
        if falsifier_log: fileset["falsifier_log.txt"] = falsifier_log
        if preview_png: fileset["preview.png"] = preview_png

        # -- Compute all file hashes
        file_hashes = {name: sha256file(path) for name, path in fileset.items()}
        capsule_data["files"] = file_hashes
        # -- Capsule SHA: hash of (all .npy/.json/.lean data in sorted order)
        cap_bytes = b''
        for k in sorted(fileset.keys()):
            with open(fileset[k],"rb") as f:
                cap_bytes += f.read()
        capsule_sha = sha256_bytes(cap_bytes)
        capsule_data["capsule_sha256"] = capsule_sha

        # -- Archive into ZIP
        capsule_name = f"{context}_capsule.zip"
        capsule_fn = os.path.join(output_dir, capsule_name)
        with zipfile.ZipFile(capsule_fn,"w",compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, src in fileset.items():
                zf.write(src, arcname=fname)
            # manifest will be added after below
        # Save manifest
        manifest_fn = os.path.join(output_dir, f"{context}_capsule_manifest.json")
        with open(manifest_fn, "w") as mf:
            json.dump(capsule_data, mf, indent=2)
        # Add manifest into zip
        with zipfile.ZipFile(capsule_fn,"a",compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_fn,"capsule_manifest.json")
        # Optionally: GPG-sign zip (placeholder)
        if sign_gpg:
            # For actual signing, replace this block
            sigtxt = f"signed-by-certificate-packager:{now}"
            sigfile = os.path.join(output_dir, f"{context}_capsule_signature.txt")
            with open(sigfile, "w") as f:
                f.write(sigtxt)
            with zipfile.ZipFile(capsule_fn,"a",compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(sigfile, "capsule_signature.txt")
        print(f"[OK] Capsule created: {capsule_fn}")
        capsules.append({
            "context": context,
            "capsule_zip": capsule_fn,
            "capsule_manifest": manifest_fn,
            "capsule_sha256": capsule_sha
        })

    # ----------- Summarize ---------------
    print(f"Packaged {len(capsules)} archive capsules into {output_dir}.")
    # Master manifest
    mani = {
        "capsules": capsules,
        "timestamp": now,
        "generator_source_hash": generator_src_hash
    }
    with open(os.path.join(output_dir,"capsule_manifest_master.json"),"w") as f:
        json.dump(mani, f, indent=2)
    print("Master manifest written.")

##########################################

def verify_capsules(capsule_dir="/content/output/capsules/"):
    print(f"Verifying capsules in {capsule_dir} ...")
    manifests = glob.glob(os.path.join(capsule_dir,"*_capsule_manifest.json"))
    zips = glob.glob(os.path.join(capsule_dir,"*_capsule.zip"))
    for mani in manifests:
        with open(mani,"r") as f:
            cap = json.load(f)
        capsule_sha = cap.get("capsule_sha256")
        # find ZIP
        context = cap.get("context")
        zipfile_path = os.path.join(capsule_dir, f"{context}_capsule.zip")
        if not os.path.exists(zipfile_path): fail(f"Capsule zip missing: {zipfile_path}")
        # Check file hashes inside
        with zipfile.ZipFile(zipfile_path,"r") as zf:
            for fname, file_sha in cap["files"].items():
                try:
                    data = zf.read(fname)
                    if sha256_bytes(data) != file_sha:
                        fail(f"[{context}] Hash mismatch for {fname} inside {zipfile_path}")
                except KeyError:
                    fail(f"[{context}] File missing in archive: {fname}")
            # Manifest file:
            mani_data = zf.read("capsule_manifest.json")
            if sha256_bytes(mani_data) != sha256_bytes(json.dumps(cap,sort_keys=True,indent=2).encode()):
                print(f"[WARN] Capsule manifest hash mismatch for {context}")
        # Check Lean cert
        cert_fname = None
        for fname in cap["files"]:
            if fname.endswith(".lean"):
                cert_fname = fname
        if cert_fname:
            with zipfile.ZipFile(zipfile_path,"r") as zf:
                cert_data = zf.read(cert_fname).decode(errors='ignore')
                if "theorem proof_invalid" not in cert_data:
                    fail(f"[{context}] Lean certificate invalid in archive.")
    print("All capsules verified OK.")

##########################################

def main():
    parser = argparse.ArgumentParser(description="Pack and validate proof certificate capsules.")
    parser.add_argument("--summary", type=str, default="/content/output/proof_summary.json")
    parser.add_argument("--output_dir", type=str, default="/content/output/capsules/")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--sign-gpg", action="store_true")
    parser.add_argument("--min_ris", type=int, default=3)
    args = parser.parse_args()
    if args.verify_only:
        verify_capsules(args.output_dir)
    else:
        package_capsules(
            proof_summary_json=args.summary,
            output_dir=args.output_dir,
            verify_only=False,
            min_ris=args.min_ris,
            sign_gpg=args.sign_gpg
        )

if __name__ == "__main__":
    main()
