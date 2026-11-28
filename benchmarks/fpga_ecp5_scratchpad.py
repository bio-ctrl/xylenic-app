"""Reproducible ECP5 DP16KD inference and placement check for ORCUS scratchpads."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "orcus" / "rtl" / "orcus_operand_scratchpad.sv"
GEOMETRY = {
    4: {"width": 16, "depth": 1024, "dp16kd": 2},
    8: {"width": 32, "depth": 512, "dp16kd": 2},
    16: {"width": 64, "depth": 256, "dp16kd": 4},
    32: {"width": 128, "depth": 128, "dp16kd": 8},
}


def run(command: list[str], log: Path, timeout: int) -> float:
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                            timeout=timeout, check=False)
    elapsed = time.perf_counter() - start
    log.write_text(result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n"
                           + (result.stdout + result.stderr)[-8000:])
    return elapsed


def cell_inventory(netlist: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for module in netlist.get("modules", {}).values():
        for cell in module.get("cells", {}).values():
            name = cell.get("type", "UNKNOWN")
            counts[name] = counts.get(name, 0) + 1
    return counts


def check_mapping(netlist_path: Path, lanes: int) -> dict:
    netlist = json.loads(netlist_path.read_text())
    counts = cell_inventory(netlist)
    expected = GEOMETRY[lanes]["dp16kd"]
    actual = counts.get("DP16KD", 0)
    distributed = {
        name: count for name, count in counts.items()
        if any(token in name.upper() for token in ("DPR16", "RAM16", "RAM32", "SPRAM"))
    }
    if actual != expected:
        raise AssertionError(
            f"{lanes}-lane mapping has {actual} DP16KD cells; expected {expected}; "
            f"inventory={counts}"
        )
    if distributed:
        raise AssertionError(f"distributed-RAM fallback detected: {distributed}")
    if any(name.startswith("$mem") for name in counts):
        raise AssertionError("generic memory cells remain after ECP5 mapping")
    return {
        "dp16kd_blocks_total": actual,
        "dp16kd_blocks_per_operand": actual // 2,
        "distributed_ram_cells": distributed,
        "lut4_cells": counts.get("LUT4", 0),
        "cell_inventory": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yosys", default="yosys")
    parser.add_argument("--nextpnr", default="nextpnr-ecp5")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/orcus-scratchpad-ecp5"))
    parser.add_argument("--freq", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--require-tools", action="store_true")
    args = parser.parse_args()

    yosys, nextpnr = shutil.which(args.yosys), shutil.which(args.nextpnr)
    missing = [name for name, path in ((args.yosys, yosys), (args.nextpnr, nextpnr)) if not path]
    if missing:
        if args.require_tools:
            raise SystemExit("missing required ECP5 tools: " + ", ".join(missing))
        print(json.dumps({"status": "skipped", "missing_tools": missing}, indent=2))
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    nextpnr_version_result = subprocess.run([nextpnr, "--version"], text=True, capture_output=True, check=False)
    nextpnr_version_text = (nextpnr_version_result.stdout + nextpnr_version_result.stderr).strip()
    if not nextpnr_version_text:
        raise RuntimeError("could not read nextpnr-ecp5 version")
    summary = {
        "experiment": "ORCUS M4.2 A/B operand scratchpad DP16KD mapping",
        "yosys_version": subprocess.run([yosys, "-V"], text=True, capture_output=True).stdout.strip(),
        "nextpnr_ecp5_version": nextpnr_version_text.splitlines()[0],
        "target": {"family": "Lattice ECP5", "device_option": "--85k",
                   "package": "CABGA381", "speed_grade": 6},
        "configurations": [],
        "placement": None,
    }

    # Exact block-RAM inference proof for every supported v1 geometry.
    for lanes, geometry in GEOMETRY.items():
        netlist_path = args.outdir / f"scratchpads-{lanes}lane.json"
        log_path = args.outdir / f"scratchpads-{lanes}lane-yosys.log"
        script = (
            f'read_verilog -sv "{RTL}"; '
            f'chparam -set LANES {lanes} orcus_operand_scratchpads; '
            'hierarchy -check -top orcus_operand_scratchpads; '
            f'synth_ecp5 -top orcus_operand_scratchpads -json "{netlist_path}"; '
            'stat'
        )
        elapsed = run([yosys, "-p", script], log_path, args.timeout)
        mapping = check_mapping(netlist_path, lanes)
        summary["configurations"].append({
            "lanes": lanes, **geometry, **mapping,
            "yosys_synthesis_seconds": elapsed,
            "netlist": netlist_path.name, "yosys_log": log_path.name,
        })

    # A low-I/O 32-lane harness keeps both memories observable for nextpnr.
    # The raw scratchpad top exposes too many pins for the selected package.
    lanes = 32
    netlist_path = args.outdir / "scratchpad-32lane-pnr.json"
    log_path = args.outdir / "scratchpad-32lane-pnr-yosys.log"
    script = (
        f'read_verilog -sv "{RTL}"; '
        f'chparam -set LANES {lanes} orcus_scratchpad_ecp5_top; '
        'hierarchy -check -top orcus_scratchpad_ecp5_top; '
        f'synth_ecp5 -top orcus_scratchpad_ecp5_top -json "{netlist_path}"; '
        'stat'
    )
    synth_seconds = run([yosys, "-p", script], log_path, args.timeout)
    mapping = check_mapping(netlist_path, lanes)
    report_path = args.outdir / "scratchpad-32lane-nextpnr.json"
    config_path = args.outdir / "scratchpad-32lane-routed.config"
    pnr_log = args.outdir / "scratchpad-32lane-nextpnr.log"
    command = [
        nextpnr, "--85k", "--package", "CABGA381", "--speed", "6",
        "--freq", str(args.freq), "--seed", str(args.seed),
        "--json", str(netlist_path), "--textcfg", str(config_path),
        "--report", str(report_path), "--detailed-timing-report",
        "--lpf-allow-unconstrained", "--timing-allow-fail",
    ]
    route_seconds = run(command, pnr_log, args.timeout)
    report = json.loads(report_path.read_text())
    fmax = report.get("fmax", {})
    clock = fmax.get("clk") or (next(iter(fmax.values())) if len(fmax) == 1 else {})
    achieved = clock.get("achieved") if isinstance(clock, dict) else None
    constraint = clock.get("constraint") if isinstance(clock, dict) else None
    if not isinstance(achieved, (int, float)) or not isinstance(constraint, (int, float)):
        raise AssertionError(f"nextpnr report missing clock metrics: {report}")
    summary["placement"] = {
        "lanes": lanes, **mapping,
        "yosys_synthesis_seconds": synth_seconds,
        "nextpnr_place_and_route_seconds": route_seconds,
        "achieved_fmax_mhz": achieved, "constraint_mhz": constraint,
        "timing_met": achieved >= constraint,
        "external_io_constraints": "unconstrained",
        "report": report_path.name, "routed_config": config_path.name,
        "nextpnr_log": pnr_log.name,
    }
    output = args.outdir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
