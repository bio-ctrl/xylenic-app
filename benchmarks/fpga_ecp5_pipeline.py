"""Exact-target ECP5 place-and-route for the separately pipelined Dadda RTL."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.fpga_ecp5 import (
    DEVICE, PACKAGE, SPEED_GRADE, parse_nextpnr_report, parse_yosys_netlist,
)
from benchmarks.dadda_pipeline import LANES, run_formal, run_streaming
from orcus.dadda_pipeline import generate_dadda_pipelined


def _run(command: list[str], log: Path, timeout: int) -> float:
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        log.write_text(stdout + "\n" + stderr)
        raise TimeoutError(f"timeout after {timeout}s: {' '.join(command)}") from exc
    elapsed = time.perf_counter() - started
    log.write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}\n{(proc.stdout+proc.stderr)[-6000:]}")
    return elapsed


def _version(binary: str) -> str:
    proc = subprocess.run([binary, "--version"], text=True, capture_output=True, check=False)
    value = (proc.stdout + proc.stderr).strip()
    if not value:
        raise RuntimeError(f"could not read version for {binary}")
    return value.splitlines()[0]


def run_one(lanes: int, outdir: Path, yosys: str, nextpnr: str,
            iverilog: str, vvp: str, freq_mhz: float, seed: int,
            timeout: int) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    design = generate_dadda_pipelined(lanes)
    rtl = outdir / "design.v"
    netlist = outdir / "synth.json"
    report_path = outdir / "nextpnr-report.json"
    config = outdir / "routed.config"
    synth_log, route_log = outdir / "yosys-synth.log", outdir / "nextpnr.log"
    rtl.write_text(design.source)
    manifest_path = outdir / "pipeline-manifest.json"
    manifest_path.write_text(json.dumps(design.manifest, indent=2, sort_keys=True) + "\n")
    validation_dir = outdir / "validation"
    formal = run_formal(lanes, yosys, timeout=min(timeout, 180), log_dir=validation_dir)
    streaming = run_streaming(lanes, iverilog, vvp, timeout=min(timeout, 60), log_dir=validation_dir)

    top = f"orcus_dadda_pipelined_{lanes}"
    script = f'read_verilog "{rtl}"; synth_ecp5 -top {top} -json "{netlist}"; stat'
    synth_seconds = _run([yosys, "-p", script], synth_log, timeout)
    resources = parse_yosys_netlist(json.loads(netlist.read_text()), top)

    route_command = [
        nextpnr, "--85k", "--package", PACKAGE, "--speed", str(SPEED_GRADE),
        "--freq", str(freq_mhz), "--seed", str(seed),
        "--json", str(netlist), "--textcfg", str(config),
        "--report", str(report_path), "--detailed-timing-report",
        "--lpf-allow-unconstrained", "--timing-allow-fail",
    ]
    route_seconds = _run(route_command, route_log, timeout)
    pnr = parse_nextpnr_report(json.loads(report_path.read_text()))
    return {
        "lanes": lanes,
        "accumulator_width": design.width,
        "variant": "fused_dadda_pipelined",
        "pipeline_latency_cycles": design.latency,
        "stage_targets": list(design.stage_targets),
        "registered_heap_bits_per_stage": [
            stage["registered_bit_count"] for stage in design.manifest["stage_boundaries"]
        ],
        "formal": formal,
        "streaming": streaming,
        "resources": resources,
        "post_route": pnr,
        "target": {
            "part": DEVICE, "package": PACKAGE, "speed_grade": SPEED_GRADE,
            "clock_target_mhz": freq_mhz, "external_io": "unconstrained",
        },
        "timings_seconds": {
            "yosys_synth_ecp5": synth_seconds,
            "nextpnr_place_and_route": route_seconds,
        },
        "files": {
            "rtl": rtl.name, "pipeline_manifest": manifest_path.name, "yosys_netlist": netlist.name,
            "yosys_log": synth_log.name, "nextpnr_report": report_path.name,
            "nextpnr_log": route_log.name, "routed_config": config.name, "validation_dir": validation_dir.name,
        },
    }


def run_matrix(outdir: Path, yosys: str = "yosys", nextpnr: str = "nextpnr-ecp5",
               iverilog: str = "iverilog", vvp: str = "vvp",
               freq_mhz: float = 100.0, seed: int = 1, timeout: int = 900) -> dict:
    tools = (yosys, nextpnr, iverilog, vvp)
    missing = [tool for tool in tools if not shutil.which(tool)]
    if missing:
        raise RuntimeError("required tools missing: " + ", ".join(missing))
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "orcus-fpga-ecp5-pipelined-dadda-v1",
        "status": "running",
        "target": {
            "part": DEVICE, "package": PACKAGE, "speed_grade": SPEED_GRADE,
            "clock_target_mhz": freq_mhz, "seed": seed,
            "external_io": "unconstrained",
        },
        "tools": {
            "yosys": _version(yosys), "nextpnr_ecp5": _version(nextpnr),
            "iverilog": _version(iverilog), "python": sys.version,
            "platform": platform.platform(),
        },
        "matrix": [],
    }
    output = outdir / "results.json"
    try:
        for lanes in LANES:
            row = run_one(lanes, outdir / f"{lanes}-lane-pipelined-dadda",
                          yosys, nextpnr, iverilog, vvp, freq_mhz, seed, timeout)
            report["matrix"].append(row)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/orcus-fpga-ecp5-pipelined-dadda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--yosys", default="yosys")
    parser.add_argument("--nextpnr", default="nextpnr-ecp5")
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    parser.add_argument("--freq", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    report = run_matrix(args.outdir, args.yosys, args.nextpnr, args.iverilog,
                        args.vvp, args.freq, args.seed, args.timeout)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
