"""Launch WignerFlow Studio: Interactive Web Workbench & 10-Gate Equivariant Physics Compliance Lab.

Usage:
    python scripts/wignerflow_studio.py --port 8765
    python scripts/wignerflow_studio.py --audit-only --target aspirin_hetero_o3 --precision float64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from equitraj.compliance import AUDIT_TARGETS, FAULT_CATALOG, run_compliance_audit  # noqa: E402
from equitraj.studio import create_studio_server  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="WignerFlow Studio: Interactive Web UI & 10-Gate Equivariant MD Compliance Lab"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--audit-only", action="store_true", help="Run CLI 10-gate compliance audit and exit")
    parser.add_argument("--target", default="argon_flow_v1", choices=list(AUDIT_TARGETS))
    parser.add_argument("--fault", default="none", choices=[f["id"] for f in FAULT_CATALOG])
    parser.add_argument("--precision", default="float32", choices=("float32", "float64"))
    parser.add_argument("--json-out", default=None, help="Optional JSON output path for --audit-only")
    args = parser.parse_args(argv)

    if args.audit_only:
        rep = run_compliance_audit(target_id=args.target, fault=args.fault, precision=args.precision)
        print("=" * 88)
        print(f" WignerFlow 10-Gate Equivariant & Physical Compliance Audit | {rep['title']}")
        print(f" Precision: {rep['precision']} | Fault: {rep['fault']} | Verdict: {rep['verdict']} ({rep['pass_count']}/{rep['total_gates']} PASS in {rep['wall_s']}s)")
        print("-" * 88)
        for g in rep["gates"]:
            print(f" [{g['status']:<4s}] {g['id']:<3s} {g['name']:<38s} | val={str(g['value']):<10s} ({g['threshold']})")
        print("=" * 88)
        if args.json_out:
            out_p = Path(args.json_out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        return 0

    srv = create_studio_server(args.host, args.port)
    print(f"WignerFlow Studio running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
