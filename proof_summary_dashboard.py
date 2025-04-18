%%writefile proof_summary_dashboard.py
#!/usr/bin/env python3
import os
import sys
import re
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize proof certificates into JSON"
    )
    parser.add_argument(
        "--input_dir", "-i",
        required=True,
        help="Path to folder containing certificate_timestep_*.lean"
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        required=True,
        help="Number of timesteps to process (e.g. 100)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to write proof_summary.json"
    )
    parser.add_argument(
        "--log_failed", action="store_true",
        help="Also write a JSON of failed/skipped files alongside the summary"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    input_dir = args.input_dir
    if not input_dir.endswith(os.sep):
        input_dir += os.sep

    # regex patterns
    re_Y       = re.compile(r"--\s*Y\s*=\s*\(\s*([0-9eE.+-]+)")
    re_align   = re.compile(r"--\s*align_sup\s*=\s*\(\s*([0-9eE.+-]+)")
    re_dYdt    = re.compile(r"--\s*Computed\s*\(LHS\s*=\s*dY/dt\)\s*:\s*([0-9eE.+-]+)")
    re_rhs     = re.compile(r"--\s*Computed\s*\(RHS\)\s*:\s*([0-9eE.+-]+)")
    re_verified = re.compile(r"^theorem\s+\w+\s*:\s*dYdt\s*≤\s*RHS", re.MULTILINE)

    summary = []
    failed = []

    for i in range(args.timesteps):
        fname = f"certificate_timestep_{i:03}.lean"
        path = os.path.join(input_dir, fname)
        entry = {
            "file": path,
            "Y": None,
            "alignment": None,
            "dY_dt": None,
            "RHS": None,
            "verified": False
        }
        try:
            with open(path, "r") as f:
                contents = f.read()
            if not contents.strip():
                print(f"Warning: empty file {path}", file=sys.stderr)
                failed.append(fname)
            else:
                # parse Y
                m = re_Y.search(contents)
                if m:
                    try:
                        entry["Y"] = float(m.group(1))
                    except ValueError:
                        print(f"Warning: could not parse Y in {path}", file=sys.stderr)
                else:
                    print(f"Warning: Y not found in {path}", file=sys.stderr)

                # parse alignment
                m = re_align.search(contents)
                if m:
                    try:
                        entry["alignment"] = float(m.group(1))
                    except ValueError:
                        print(f"Warning: could not parse alignment in {path}", file=sys.stderr)
                else:
                    print(f"Warning: alignment not found in {path}", file=sys.stderr)

                # parse dY/dt
                m = re_dYdt.search(contents)
                if m:
                    try:
                        entry["dY_dt"] = float(m.group(1))
                    except ValueError:
                        print(f"Warning: could not parse dY/dt in {path}", file=sys.stderr)
                else:
                    print(f"Warning: dY/dt not found in {path}", file=sys.stderr)

                # parse RHS
                m = re_rhs.search(contents)
                if m:
                    try:
                        entry["RHS"] = float(m.group(1))
                    except ValueError:
                        print(f"Warning: could not parse RHS in {path}", file=sys.stderr)
                else:
                    print(f"Warning: RHS not found in {path}", file=sys.stderr)

                # parse verified theorem
                if re_verified.search(contents):
                    entry["verified"] = True

                # record failures if any metric missing
                if any(entry[k] is None for k in ["Y","alignment","dY_dt","RHS"]):
                    failed.append(fname)

        except FileNotFoundError:
            print(f"Warning: file not found {path}", file=sys.stderr)
            failed.append(fname)
        except Exception as e:
            print(f"Warning: error reading {path}: {e}", file=sys.stderr)
            failed.append(fname)

        summary.append(entry)

    # write summary JSON
    out_data = {"summary": summary}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Wrote summary to {args.output}")

    # optionally write failed log
    if args.log_failed:
        failed_path = args.output.rstrip(".json") + "_failed.json"
        with open(failed_path, "w") as f:
            json.dump({"failed": failed}, f, indent=2)
        print(f"Wrote failed log to {failed_path}")

if __name__ == "__main__":
    main()
