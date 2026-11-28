"""Sequential formal and 512-transaction streaming validation for pipelined Dadda RTL."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from orcus.dadda_pipeline import generate_dadda_pipelined
from orcus.lane_scaling import module_source, accumulator_width

LANES = (4, 8, 16, 32)
VECTOR_COUNT = 512
SEED = 0x0C0FFEE


def _ports(lanes: int) -> tuple[list[str], list[str]]:
    decls, conns = [], []
    for lane in range(lanes):
        decls.extend((f"input wire signed [3:0] a{lane}", f"input wire signed [3:0] b{lane}"))
        conns.extend((f".a{lane}(a{lane})", f".b{lane}(b{lane})"))
    return decls, conns


def _weighted_sum(heap: list[list[str]], width: int, prefix: str) -> tuple[list[str], str]:
    terms = []
    declarations = []
    for column, tokens in enumerate(heap):
        for index, token in enumerate(tokens):
            term = f"{prefix}_term_{column}_{index}"
            declarations.append(
                f"  wire [{width - 1}:0] {term} = "
                f"{{{{{width - 1}{{1'b0}}}}, {token}}} << {column};"
            )
            terms.append(term)
    expression = " + ".join(terms) if terms else f"{width}'b0"
    return declarations, expression


def build_initial_heap_miter(lanes: int, lane_index: int = 0) -> str:
    """Prove one lane's Baugh-Wooley heap equals its signed 4x4 product."""
    if lane_index < 0 or lane_index >= lanes:
        raise ValueError(f"invalid lane {lane_index} for {lanes} lanes")
    design = generate_dadda_pipelined(lanes)
    width = design.width
    all_heap = design.manifest["stage_boundaries"][0]["input_heap"]
    prefix = f"_{lane_index}_"
    lane_heap = [
        [token for token in column if token.startswith(("pp" + prefix, "bias" + prefix))]
        for column in all_heap
    ]
    ports = [
        f"input wire signed [3:0] a{lane_index}",
        f"input wire signed [3:0] b{lane_index}",
        "output wire bad",
    ]
    lines = [f"module orcus_dadda_initial_heap_miter_{lanes}_{lane_index}({', '.join(ports)});"]

    for i in range(4):
        for j in range(4):
            name = f"pp_{lane_index}_{i}_{j}"
            product = f"(a{lane_index}[{i}] & b{lane_index}[{j}])"
            inverted = (i == 3) != (j == 3)
            lines.append(f"  wire {name} = " + (f"~{product};" if inverted else f"{product};"))
    lane_tokens = {token for column in lane_heap for token in column}
    for bit in range(width):
        bias = f"bias_{lane_index}_{bit}"
        if bias in lane_tokens:
            lines.append(f"  wire {bias} = 1'b1;")

    lines.append(
        f"  wire signed [7:0] signed_a = {{{{4{{a{lane_index}[3]}}}}, a{lane_index}}};"
    )
    lines.append(
        f"  wire signed [7:0] signed_b = {{{{4{{b{lane_index}[3]}}}}, b{lane_index}}};"
    )
    lines.append("  wire signed [7:0] signed_product = signed_a * signed_b;")
    lines.append(
        f"  wire signed [{width - 1}:0] extended_product = "
        f"{{{{{width - 8}{{signed_product[7]}}}}, signed_product}};"
    )
    heap_decls, heap_expr = _weighted_sum(lane_heap, width, "initial")
    lines.extend(heap_decls)
    lines.append(f"  wire [{width - 1}:0] initial_heap_sum = {heap_expr};")
    lines.append("  assign bad = (initial_heap_sum != extended_product);")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"





def validate_stage_bit_partition(input_heap: list[list[str]], output_heap: list[list[str]],
                                 compressors: list[dict], width: int) -> dict:
    """Validate ownership of primary inputs, intra-stage carries, and outputs."""
    if width <= 0 or len(input_heap) != width or len(output_heap) != width:
        raise ValueError("heap column count must equal positive accumulator width")
    def locate(heap, label):
        found = {}
        for col, tokens in enumerate(heap):
            for token in tokens:
                if token in found:
                    raise ValueError(f"{label} bit {token} appears more than once")
                found[token] = col
        return found
    primary = locate(input_heap, "input")
    sums, carries, consumed = {}, {}, {}
    by_col = [[] for _ in range(width)]
    for rec in compressors:
        col, kind, ins = rec.get("column"), rec.get("kind"), rec.get("inputs")
        if not isinstance(col, int) or not 0 <= col < width:
            raise ValueError(f"invalid compressor column: {col!r}")
        arity = 3 if kind == "full_adder" else 2 if kind == "half_adder" else None
        if arity is None or not isinstance(ins, list) or len(ins) != arity:
            raise ValueError(f"invalid compressor kind/arity: {rec!r}")
        st = rec.get("sum")
        if not isinstance(st, str) or not st or st in sums or st in carries or st in primary:
            raise ValueError(f"duplicate or aliased compressor sum output {st}")
        sums[st] = col
        ct, cc = rec.get("carry"), rec.get("carry_column")
        if col == width - 1:
            if ct is not None or cc is not None or not rec.get("carry_truncated_modulo_2_W"):
                raise ValueError("top-column carry must be explicitly truncated modulo 2**width")
        else:
            if not isinstance(ct, str) or not ct:
                raise ValueError(f"compressor in column {col} is missing its carry output")
            if cc != col + 1:
                raise ValueError(f"misweighted carry {ct}: expected column {col + 1}, found {cc}")
            if ct in sums or ct in carries or ct in primary:
                raise ValueError(f"duplicate or aliased compressor carry output {ct}")
            carries[ct] = cc
        by_col[col].append(rec)
    for rec in compressors:
        col = rec["column"]
        for token in rec["inputs"]:
            if token in primary:
                expected_col, source_kind = primary[token], "primary input"
            elif token in carries:
                expected_col, source_kind = carries[token], "intra-stage carry"
            else:
                raise ValueError(f"compressor consumes unknown stage bit {token}")
            if expected_col != col:
                raise ValueError(f"misweighted {source_kind} bit {token}: expected column {col}, found {expected_col}")
            if token in consumed:
                raise ValueError(f"stage bit {token} consumed more than once")
            consumed[token] = rec.get("id", "unnamed")
    primary_pass = [token for token in primary if token not in consumed]
    carry_pass = [token for token in carries if token not in consumed]
    passthrough = [[] for _ in range(width)]
    for token in primary_pass:
        passthrough[primary[token]].append(token)
    for token in carry_pass:
        passthrough[carries[token]].append(token)
    for col in range(width):
        expected = passthrough[col] + [token for token, weight in sums.items() if weight == col]
        actual = output_heap[col]
        if len(actual) != len(set(actual)):
            raise ValueError(f"duplicate output bit in column {col}")
        if set(actual) != set(expected) or len(actual) != len(expected):
            missing, extra = sorted(set(expected)-set(actual)), sorted(set(actual)-set(expected))
            raise ValueError(f"output partition mismatch in column {col}: missing={missing}, unexpected={extra}")
    consumed_primary = sum(token in primary for token in consumed)
    consumed_carries = sum(token in carries for token in consumed)
    if consumed_primary + len(primary_pass) != len(primary):
        raise ValueError("primary input bit accounting is incomplete")
    if consumed_carries + len(carry_pass) != len(carries):
        raise ValueError("intra-stage carry accounting is incomplete")
    if sum(map(len, output_heap)) != len(sums) + sum(map(len, passthrough)):
        raise ValueError("output bit accounting is incomplete")
    return {"columns":[{"column":c,"compressors":by_col[c],"pass_through":passthrough[c]} for c in range(width)],
            "input_bit_count":len(primary),"consumed_input_bit_count":consumed_primary,
            "primary_pass_through_bit_count":len(primary_pass),
            "generated_carry_bit_count":len(carries),"consumed_carry_bit_count":consumed_carries,
            "pass_through_bit_count":sum(map(len,passthrough)),
            "output_bit_count":sum(map(len,output_heap)),"compressor_count":len(compressors)}


def build_stage_partition(lanes: int, stage_index: int) -> dict:
    design = generate_dadda_pipelined(lanes)
    boundaries = design.manifest["stage_boundaries"]
    if stage_index < 0 or stage_index >= len(boundaries):
        raise ValueError(f"invalid stage {stage_index} for {lanes} lanes")
    boundary = boundaries[stage_index]
    compressors = [r for r in design.manifest["compressors"] if r["stage"] == stage_index]
    part = validate_stage_bit_partition(boundary["input_heap"], boundary["output_heap"], compressors, design.width)
    part["stage"], part["target_height"] = stage_index, boundary["target_height"]
    return part


def build_stage_column_miter(lanes: int, stage_index: int, column_index: int) -> str:
    """Prove local compressor conservation; no carry ripples between stage columns."""
    design = generate_dadda_pipelined(lanes)
    part = build_stage_partition(lanes, stage_index)
    if column_index < 0 or column_index >= design.width:
        raise ValueError(f"invalid column {column_index} for {lanes} lanes")
    records = part["columns"][column_index]["compressors"]
    if not records:
        raise ValueError(f"column {column_index} has no compressors to prove")
    ins = [x for r in records for x in r["inputs"]]
    sums = [r["sum"] for r in records]
    carries = [r["carry"] for r in records if r["carry"] is not None]
    if len(ins) != len(set(ins)):
        raise ValueError(f"column {column_index} has overlapping compressor inputs")
    lines = [f"module orcus_dadda_stage_column_miter_{lanes}_{stage_index}_{column_index}(",
             ", ".join([f"input wire {x}" for x in ins]+["output wire bad"]), ");"]
    for r in records:
        x,y=r["inputs"][:2]
        if r["kind"]=="full_adder":
            z=r["inputs"][2]
            lines.append(f"  wire {r['sum']} = {x} ^ {y} ^ {z};")
            if r["carry"] is not None:
                lines.append(f"  wire {r['carry']} = ({x} & {y}) | ({x} & {z}) | ({y} & {z});")
        else:
            lines.append(f"  wire {r['sum']} = {x} ^ {y};")
            if r["carry"] is not None:
                lines.append(f"  wire {r['carry']} = {x} & {y};")
    if column_index == design.width-1:
        lines.append(f"  assign bad = (({' ^ '.join(ins)}) != ({' ^ '.join(sums)}));")
    else:
        w=max(2,(len(ins)+2*len(records)+1).bit_length())
        def ext(x): return f"{{{{{w-1}{{1'b0}}}}, {x}}}"
        lhs=" + ".join(ext(x) for x in ins)
        rhs_terms=[ext(x) for x in sums]+[f"({ext(x)} << 1)" for x in carries]
        lines.append(f"  wire [{w-1}:0] conservation_inputs = {lhs};")
        lines.append(f"  wire [{w-1}:0] conservation_outputs = {' + '.join(rhs_terms)};")
        lines.append("  assign bad = (conservation_inputs != conservation_outputs);")
    lines.append("endmodule")
    return "\n".join(lines)+"\n"




def build_atomic_compressor_miter(kind: str) -> tuple[str, str]:
    """Build a tiny, reusable SAT lemma for one 1-bit FA or HA."""
    if kind == "full_adder":
        top = "orcus_atomic_fa_conservation"
        ports = "input wire a, input wire b, input wire cin, output wire bad"
        body = [
            "  wire sum_bit = a ^ b ^ cin;",
            "  wire carry_bit = (a & b) | (a & cin) | (b & cin);",
            "  wire [2:0] lhs = {2'b0, a} + {2'b0, b} + {2'b0, cin};",
        ]
    elif kind == "half_adder":
        top = "orcus_atomic_ha_conservation"
        ports = "input wire a, input wire b, output wire bad"
        body = [
            "  wire sum_bit = a ^ b;",
            "  wire carry_bit = a & b;",
            "  wire [2:0] lhs = {2'b0, a} + {2'b0, b};",
        ]
    else:
        raise ValueError(f"unsupported compressor kind: {kind}")
    body.extend([
        "  wire [2:0] rhs = {2'b0, sum_bit} + ({2'b0, carry_bit} << 1);",
        "  assign bad = (lhs != rhs);",
    ])
    return top, "\n".join([f"module {top}({ports});", *body, "endmodule", ""])


def validate_stage_netlist_wiring(source: str, compressors: list[dict],
                                  stage_index: int,
                                  output_heap: list[list[str]] | None = None) -> dict:
    """Match every scheduled compressor to its exact emitted Verilog equations."""
    source_lines = source.splitlines()
    checked = 0
    ids = set()
    for rec in compressors:
        ident = rec.get("id")
        if not isinstance(ident, str) or ident in ids:
            raise ValueError(f"duplicate or invalid compressor id: {ident!r}")
        ids.add(ident)
        kind, ins = rec.get("kind"), rec.get("inputs")
        if kind == "full_adder" and isinstance(ins, list) and len(ins) == 3:
            x, y, z = ins
            sum_expr = f"{x} ^ {y} ^ {z}"
            carry_expr = f"({x} & {y}) | ({x} & {z}) | ({y} & {z})"
        elif kind == "half_adder" and isinstance(ins, list) and len(ins) == 2:
            x, y = ins
            sum_expr = f"{x} ^ {y}"
            carry_expr = f"{x} & {y}"
        else:
            raise ValueError(f"invalid compressor record {ident}: {rec!r}")
        sum_line = f"  wire {rec['sum']} = {sum_expr};"
        if source_lines.count(sum_line) != 1:
            raise ValueError(f"stage {stage_index} compressor {ident} sum pin/netlist mismatch")
        carry = rec.get("carry")
        if carry is None:
            if rec.get("carry_column") is not None or not rec.get("carry_truncated_modulo_2_W"):
                raise ValueError(f"stage {stage_index} compressor {ident} has invalid MSB truncation metadata")
            prefix, suffix = ident.split("_", 1)
            unexpected_carry = f"{prefix}_c_{suffix}"
            if any(line.startswith(f"  wire {unexpected_carry} =") for line in source_lines):
                raise ValueError(f"stage {stage_index} compressor {ident} unexpectedly emits an MSB carry")
        else:
            carry_line = f"  wire {carry} = {carry_expr};"
            if source_lines.count(carry_line) != 1:
                raise ValueError(f"stage {stage_index} compressor {ident} carry pin/netlist mismatch")
        checked += 1
    registered_bits_checked = 0
    if output_heap is not None:
        for column in output_heap:
            for token in column:
                qname = f"st{stage_index}_q_{token}"
                register_line = f"      {qname} <= {token};"
                if source_lines.count(register_line) != 1:
                    raise ValueError(
                        f"stage {stage_index} register/pass-through wiring mismatch for {token}"
                    )
                registered_bits_checked += 1
    return {"status": "PASS", "stage": stage_index,
            "compressors_checked": checked,
            "registered_bits_checked": registered_bits_checked,
            "full_adders": sum(r["kind"] == "full_adder" for r in compressors),
            "half_adders": sum(r["kind"] == "half_adder" for r in compressors),
            "method": "exact emitted-Verilog compressor pin/equation matching"}

def build_final_cpa_miter(lanes: int) -> str:
    """Retain the final two-row CPA equivalence proof as a separate SAT job."""
    design=generate_dadda_pipelined(lanes)
    heap=design.manifest["stage_boundaries"][-1]["output_heap"]
    tokens=list(dict.fromkeys(x for col in heap for x in col))
    lines=[f"module orcus_dadda_final_cpa_miter_{lanes}(",
           ", ".join([f"input wire {x}" for x in tokens]+["output wire bad"]), ");"]
    row0=[col[0] if col else "1'b0" for col in heap]
    row1=[col[1] if len(col)>1 else "1'b0" for col in heap]
    lines.append(f"  wire [{design.width-1}:0] final_row0 = {{{', '.join(reversed(row0))}}};")
    lines.append(f"  wire [{design.width-1}:0] final_row1 = {{{', '.join(reversed(row1))}}};")
    lines.append(f"  wire [{design.width-1}:0] final_cpa = final_row0 + final_row1;")
    decls,expr=_weighted_sum(heap,design.width,"final_cpa")
    lines.extend(decls)
    lines.append(f"  wire [{design.width-1}:0] weighted_output = {expr};")
    lines.append("  assign bad = (final_cpa != weighted_output);")
    lines.append("endmodule")
    return "\n".join(lines)+"\n"


def build_control_miter(lanes: int) -> tuple[str, int]:
    """Compare post-edge DUT valid against a reference including out_valid.

    The DUT contains LATENCY reduction-stage valid registers plus the
    separately registered out_valid boundary.  The reference therefore needs
    LATENCY + 1 bits for a post-edge comparison, while transaction latency
    remains LATENCY rising-edge intervals from input sampling to output.
    """
    design = generate_dadda_pipelined(lanes)
    latency = design.latency
    reference_depth = latency + 1
    ports = ["input wire clk", "input wire reset", "input wire in_valid", "output wire bad"]
    lines = [f"module orcus_pipeline_control_miter_{lanes}({', '.join(ports)});",
             "  wire dut_valid;",
             f"  wire signed [{design.width - 1}:0] dut_data;"]
    dut_ports = [".clk(clk)", ".reset(reset)", ".in_valid(in_valid)",
                 ".out_valid(dut_valid)", ".out_data(dut_data)"]
    for lane in range(lanes):
        lines.append(f"  wire signed [3:0] a{lane} = 4'sd0;")
        lines.append(f"  wire signed [3:0] b{lane} = 4'sd0;")
        dut_ports.extend((f".a{lane}(a{lane})", f".b{lane}(b{lane})"))
    lines.append(f"  orcus_dadda_pipelined_{lanes} dut ({', '.join(dut_ports)});")
    lines.append(f"  reg [{reference_depth - 1}:0] ref_valid_pipe;")
    lines.append("  always @(posedge clk) begin")
    lines.append("    if (reset) ref_valid_pipe <= 0;")
    lines.append("    else begin")
    lines.append("      ref_valid_pipe[0] <= in_valid;")
    for index in range(1, reference_depth):
        lines.append(f"      ref_valid_pipe[{index}] <= ref_valid_pipe[{index - 1}];")
    lines.extend(["    end", "  end"])
    lines.append(f"  assign bad = (dut_valid != ref_valid_pipe[{reference_depth - 1}]);")
    lines.append("endmodule")
    return design.source + "\n" + "\n".join(lines) + "\n", latency


def build_sequential_miter(lanes: int) -> tuple[str, int]:
    """Compatibility alias: return the control-only sequential miter."""
    return build_control_miter(lanes)

def _signed4(value: int) -> int:
    value &= 15
    return value - 16 if value & 8 else value


def _sv_signed(value: int, width: int) -> str:
    return f"-{width}'sd{-value}" if value < 0 else f"{width}'sd{value}"


def _vectors(lanes: int, count: int = VECTOR_COUNT) -> list[tuple[list[int], list[int], int]]:
    rng = random.Random(SEED + lanes)
    width = accumulator_width(lanes)
    mask = (1 << width) - 1
    result = []
    for _ in range(count):
        a = [rng.randrange(16) for _ in range(lanes)]
        b = [rng.randrange(16) for _ in range(lanes)]
        total = sum(_signed4(x) * _signed4(y) for x, y in zip(a, b)) & mask
        if total & (1 << (width - 1)):
            total -= 1 << width
        result.append((a, b, total))
    return result


def build_streaming_testbench(lanes: int) -> tuple[str, int]:
    design = generate_dadda_pipelined(lanes)
    width, latency = design.width, design.latency
    vectors = _vectors(lanes)
    ports = [".clk(clk)", ".reset(reset)", ".in_valid(in_valid)", ".out_valid(out_valid)", ".out_data(out_data)"]
    for lane in range(lanes):
        ports.extend((f".a{lane}(a{lane})", f".b{lane}(b{lane})"))
    lines = [design.source, "module tb;", "  reg clk=0;", "  reg reset=1;", "  reg in_valid=0;",
             "  wire out_valid;", f"  wire signed [{width - 1}:0] out_data;"]
    for lane in range(lanes):
        lines.extend((f"  reg signed [3:0] a{lane}=0;", f"  reg signed [3:0] b{lane}=0;"))
    lines.append(f"  orcus_dadda_pipelined_{lanes} dut ({', '.join(ports)});")
    lines.append("  always #5 clk = ~clk;")
    lines.append("  integer idx;")
    lines.append("  initial begin")
    lines.append("    repeat (2) @(negedge clk); reset = 0;")
    for idx, (a, b, expected) in enumerate(vectors):
        lines.append("    @(negedge clk); in_valid = 1'b1;")
        for lane in range(lanes):
            lines.append(f"    a{lane} = {_sv_signed(_signed4(a[lane]), 4)}; b{lane} = {_sv_signed(_signed4(b[lane]), 4)};")
        lines.append("    @(posedge clk); #1;")
        if idx >= latency:
            exp = vectors[idx - latency][2]
            lines.append(f"    if (out_valid !== 1'b1 || out_data !== {_sv_signed(exp, width)}) $fatal(1, \"stream mismatch at input {idx}, expected transaction {idx-latency}\");")
        else:
            lines.append("    if (out_valid !== 1'b0) $fatal(1, \"out_valid asserted before pipeline latency\");")
    for flush in range(latency):
        lines.append("    @(negedge clk); in_valid = 1'b0;")
        lines.append("    @(posedge clk); #1;")
        exp = vectors[len(vectors) - latency + flush][2]
        lines.append(f"    if (out_valid !== 1'b1 || out_data !== {_sv_signed(exp, width)}) $fatal(1, \"flush mismatch {flush}\");")
    lines.extend(["    @(negedge clk); in_valid = 0;", "    @(posedge clk); #1;",
                  "    if (out_valid !== 1'b0) $fatal(1, \"out_valid did not drain\");",
                  "    @(negedge clk); in_valid = 1'b1;"])
    for lane in range(lanes):
        lines.append(f"    a{lane} = 4'sd0; b{lane} = 4'sd0;")
    lines.extend(["    @(posedge clk); #1;",
                  "    @(negedge clk); reset = 1'b1; in_valid = 1'b0;",
                  "    @(posedge clk); #1;",
                  "    if (out_valid !== 1'b0) $fatal(1, \"reset failed to clear valid\");",
                  "    @(negedge clk); reset = 1'b0;"])
    for _ in range(latency + 1):
        lines.extend(["    @(posedge clk); #1;",
                      "    if (out_valid !== 1'b0) $fatal(1, \"stale transaction leaked after reset\");"])
    lines.extend([f"    $display(\"PASS lanes={lanes} vectors={VECTOR_COUNT} latency={latency}\");",
                  "    $finish;", "  end", "endmodule"])
    return "\n".join(lines) + "\n", latency


def run_formal(lanes: int, yosys: str, timeout: int = 180, log_dir: Path | None = None) -> dict:
    """Prove each compressor stage independently, then prove valid/reset control.

    No sequential multiplier datapath is unrolled. The only sequential SAT
    instance observes the DUT's valid output and is optimized to its control
    cone before SAT.
    """
    if not shutil.which(yosys):
        raise RuntimeError(f"required validation tool not found: {yosys}")
    design = generate_dadda_pipelined(lanes)
    results = []
    with tempfile.TemporaryDirectory(prefix=f"orcus-pipe-compositional-{lanes}-") as tmp:
        root = Path(tmp)

        def execute(label: str, source: str, top: str, sequential: bool) -> dict:
            verilog = root / f"{label}.v"
            log = root / f"{label}.log"
            verilog.write_text(source)
            prep = f"read_verilog {verilog}; prep -top {top}; flatten; opt_clean -purge;"
            if sequential:
                depth = 2 * design.latency + 2
                sat = (f"sat -seq {depth} -tempinduct -verify -prove bad 0 "
                       "-set-init-zero")
            else:
                depth = None
                sat = "sat -verify -prove bad 0"
            started = time.perf_counter()
            try:
                proc = subprocess.run([yosys, "-p", prep + " " + sat],
                                      text=True, capture_output=True, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                log_text = stdout + "\n" + stderr
                if log_dir is not None:
                    log_dir.mkdir(parents=True, exist_ok=True)
                    (log_dir / f"{label}.v").write_text(source)
                    (log_dir / f"{label}-timeout.log").write_text(log_text)
                raise TimeoutError(f"{label} formal timeout for {lanes} lanes after {timeout}s") from exc
            elapsed = time.perf_counter() - started
            log_text = proc.stdout + "\n" + proc.stderr
            log.write_text(log_text)
            if log_dir is not None:
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / f"{label}.v").write_text(source)
                (log_dir / f"{label}.log").write_text(log_text)
            if proc.returncode:
                raise RuntimeError(
                    f"{label} formal failed for {lanes} lanes: {log_text[-5000:]}"
                )
            result = {"status": "PASS", "runtime_seconds": elapsed, "log": f"{label}.log"}
            if depth is not None:
                result["depth_cycles"] = depth
                result["method"] = "Yosys sequential temporal induction on valid/reset control cone"
            else:
                result["method"] = "Yosys combinational SAT weighted-heap equivalence"
            return result

        initial_proofs = []
        for lane_index in range(lanes):
            initial_source = build_initial_heap_miter(lanes, lane_index)
            initial_top = f"orcus_dadda_initial_heap_miter_{lanes}_{lane_index}"
            proof = execute(f"initial-baugh-wooley-lane-{lane_index}",
                            initial_source, initial_top, sequential=False)
            proof["lane"] = lane_index
            initial_proofs.append(proof)

        atomic_proofs = {}
        for kind in ("full_adder", "half_adder"):
            atomic_top, atomic_source = build_atomic_compressor_miter(kind)
            proof = execute(f"atomic-{kind}-conservation", atomic_source,
                            atomic_top, sequential=False)
            proof["method"] = "Yosys SAT atomic bit-weight conservation lemma"
            atomic_proofs[kind] = proof

        for stage_index in range(design.latency):
            partition = build_stage_partition(lanes, stage_index)
            stage_compressors = [r for r in design.manifest["compressors"]
                                 if r["stage"] == stage_index]
            wiring = validate_stage_netlist_wiring(
                design.source, stage_compressors, stage_index,
                design.manifest["stage_boundaries"][stage_index]["output_heap"])
            # The ownership/weight partition composes the atomic FA/HA lemmas:
            # every source bit is consumed once or passed through once; every
            # sum remains at 2**column; every carry advances one column; and
            # an MSB carry is explicitly discarded modulo 2**W.
            results.append({
                "status":"PASS", "stage":stage_index, "target_height":design.stage_targets[stage_index],
                "method":"atomic FA/HA SAT lemmas composed over exact emitted-Verilog token graph",
                "atomic_lemmas": ["full_adder", "half_adder"],
                "netlist_wiring": wiring,
                "bit_accounting":{k:partition[k] for k in (
                    "input_bit_count","consumed_input_bit_count","primary_pass_through_bit_count",
                    "generated_carry_bit_count","consumed_carry_bit_count",
                    "pass_through_bit_count","output_bit_count","compressor_count")},
                "columns_composed":len(partition["columns"]),
                "final_cpa_included":False,
            })

        final_cpa_source = build_final_cpa_miter(lanes)
        final_cpa = execute("final-cpa-equivalence", final_cpa_source,
                            f"orcus_dadda_final_cpa_miter_{lanes}", sequential=False)

        control_source, latency = build_control_miter(lanes)
        control_top = f"orcus_pipeline_control_miter_{lanes}"
        control = execute("control-valid-reset-induction",
                          control_source, control_top, sequential=True)

    return {
        "status": "PASS",
        "method": "atomic FA/HA SAT conservation composed over exact compressor-token wiring, separate final CPA SAT, and sequential control induction",
        "latency_cycles": latency,
        "initial_heap_proofs": initial_proofs,
        "atomic_compressor_proofs": atomic_proofs,
        "stage_proofs": results,
        "final_cpa_proof": final_cpa,
        "control_proof": control,
    }

def run_streaming(lanes: int, iverilog: str, vvp: str, timeout: int = 60, log_dir: Path | None = None) -> dict:
    source, latency = build_streaming_testbench(lanes)
    with tempfile.TemporaryDirectory(prefix=f"orcus-pipe-stream-{lanes}-") as tmp:
        root = Path(tmp)
        tb, image = root / "stream.v", root / "stream.out"
        tb.write_text(source)
        compile_run = subprocess.run([iverilog, "-g2012", "-s", "tb", "-o", str(image), str(tb)],
                                     text=True, capture_output=True, timeout=timeout)
        compile_text = compile_run.stdout + "\\n" + compile_run.stderr
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{lanes}-lane-stream-testbench.v").write_text(source)
            (log_dir / f"{lanes}-lane-iverilog.log").write_text(compile_text)
        if compile_run.returncode:
            raise RuntimeError(f"iverilog compile failed for {lanes} lanes: {compile_run.stderr[-4000:]}")
        try:
            sim = subprocess.run([vvp, str(image)], text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            if log_dir is not None:
                (log_dir / f"{lanes}-lane-vvp-timeout.log").write_text(stdout + "\\n" + stderr)
            raise TimeoutError(f"streaming simulation timeout for {lanes} lanes after {timeout}s") from exc
        sim_text = sim.stdout + "\n" + sim.stderr
        if log_dir is not None:
            (log_dir / f"{lanes}-lane-vvp.log").write_text(sim_text)
        if sim.returncode or "PASS" not in sim.stdout:
            raise RuntimeError(f"streaming simulation failed for {lanes} lanes: {sim.stdout[-4000:]} {sim.stderr[-2000:]}")
        return {"status": "PASS", "vectors": VECTOR_COUNT, "latency_cycles": latency,
                "stdout": sim.stdout.strip(),
                "compile_log": f"{lanes}-lane-iverilog.log",
                "simulation_log": f"{lanes}-lane-vvp.log"}


def run_all(yosys: str = "yosys", iverilog: str = "iverilog", vvp: str = "vvp") -> dict:
    for binary in (yosys, iverilog, vvp):
        if not shutil.which(binary):
            raise RuntimeError(f"required validation tool not found: {binary}")
    results = []
    for lanes in LANES:
        design = generate_dadda_pipelined(lanes)
        formal = run_formal(lanes, yosys)
        streaming = run_streaming(lanes, iverilog, vvp)
        results.append({"lanes": lanes, "width": design.width,
                        "latency_cycles": design.latency,
                        "stage_targets": list(design.stage_targets),
                        "registered_heap_bits_per_stage": [
                            stage["registered_bit_count"] for stage in design.manifest["stage_boundaries"]],
                        "formal": formal, "streaming": streaming})
    return {"schema": "orcus-dadda-pipeline-validation-v1",
            "status": "complete", "seed": SEED, "vectors_per_configuration": VECTOR_COUNT,
            "results": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yosys", default="yosys")
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    parser.add_argument("--output", type=Path, default=Path("artifacts/orcus-dadda-pipeline-validation.json"))
    args = parser.parse_args()
    report = run_all(args.yosys, args.iverilog, args.vvp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
