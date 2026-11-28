"""Separately generated, cycle-accurate pipelined Dadda compressor RTL.

The legacy combinational generator is intentionally not modified.  This module
reconstructs the Dadda schedule and registers the complete live heap after
each reduction level, including pass-through bits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .lane_scaling import accumulator_width, baugh_wooley_bias


@dataclass(frozen=True)
class PipelineDesign:
    lanes: int
    width: int
    latency: int
    stage_targets: tuple[int, ...]
    source: str
    manifest: dict[str, Any]


def _bias_columns(width: int) -> list[int]:
    bias = baugh_wooley_bias(width)
    return [bit for bit in range(width) if (bias >> bit) & 1]


def generate_dadda_pipelined(lanes: int) -> PipelineDesign:
    """Generate a synchronous Dadda pipeline with one register boundary/level.

    Inputs are sampled by the first reduction-level registers.  Every
    subsequent level captures every live bit in the heap.  The final CPA is
    evaluated from the last registered heap and captured in the output
    register.  Thus latency is exactly the number of Dadda reduction levels.
    """
    width = accumulator_width(lanes)
    columns: list[list[str]] = [[] for _ in range(width)]
    lines: list[str] = []
    initial: list[str] = []

    for lane in range(lanes):
        for i in range(4):
            for j in range(4):
                name = f"pp_{lane}_{i}_{j}"
                inverted = (i == 3) != (j == 3)
                expr = f"(a{lane}[{i}] & b{lane}[{j}])"
                lines.append(f"  wire {name} = " + (f"~{expr};" if inverted else f"{expr};"))
                columns[i + j].append(name)
                initial.append(name)
        for bit in _bias_columns(width):
            name = f"bias_{lane}_{bit}"
            lines.append(f"  wire {name} = 1'b1;")
            columns[bit].append(name)
            initial.append(name)

    maximum = max(len(col) for col in columns)
    targets = [2]
    while targets[-1] < maximum:
        targets.append((3 * targets[-1]) // 2)
    stage_targets = list(reversed(targets[:-1]))
    compressor_records: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []
    stage_registers: list[list[tuple[str, str]]] = []

    for stage, target in enumerate(stage_targets):
        next_columns: list[list[str]] = [[] for _ in range(width)]
        stage_carries: list[list[str]] = [[] for _ in range(width)]
        stage_logic: list[str] = []
        for column in range(width):
            work = list(columns[column]) + stage_carries[column]
            while len(work) + len(next_columns[column]) > target:
                excess = len(work) + len(next_columns[column]) - target
                kind = "full_adder" if excess >= 2 and len(work) >= 3 else "half_adder"
                arity = 3 if kind == "full_adder" else 2
                inputs = work[:arity]
                del work[:arity]
                idx = len(compressor_records)
                prefix = "fa" if kind == "full_adder" else "ha"
                sum_name = f"{prefix}_s_{idx}"
                carry_name = f"{prefix}_c_{idx}" if column < width - 1 else None
                if kind == "full_adder":
                    x, y, z = inputs
                    sum_expr = f"{x} ^ {y} ^ {z}"
                    carry_expr = f"({x} & {y}) | ({x} & {z}) | ({y} & {z})"
                else:
                    x, y = inputs
                    sum_expr = f"{x} ^ {y}"
                    carry_expr = f"{x} & {y}"
                stage_logic.append(f"  wire {sum_name} = {sum_expr};")
                next_columns[column].append(sum_name)
                if carry_name:
                    stage_logic.append(f"  wire {carry_name} = {carry_expr};")
                    stage_carries[column + 1].append(carry_name)
                compressor_records.append({
                    "id": f"{prefix}_{idx}", "kind": kind, "column": column,
                    "stage": stage, "inputs": inputs, "sum": sum_name,
                    "carry": carry_name,
                    "carry_column": column + 1 if carry_name else None,
                    "carry_truncated_modulo_2_W": carry_name is None,
                })
            next_columns[column].extend(work)

        # Register every bit in the complete next heap.  This deliberately
        # includes untouched pass-through bits, not only compressor outputs.
        register_pairs: list[tuple[str, str]] = []
        registered_columns: list[list[str]] = [[] for _ in range(width)]
        for col, tokens in enumerate(next_columns):
            for token in tokens:
                qname = f"st{stage}_q_{token}"
                lines.append(f"  reg {qname};")
                register_pairs.append((qname, token))
                registered_columns[col].append(qname)
        lines.extend(stage_logic)
        stage_registers.append(register_pairs)
        boundary_records.append({
            "stage": stage,
            "target_height": target,
            "input_heap": [list(col) for col in columns],
            "output_heap": [list(col) for col in next_columns],
            "registered_heap": [list(col) for col in registered_columns],
            "registered_bit_count": len(register_pairs),
        })
        columns = registered_columns

    if any(len(col) > 2 for col in columns):
        raise RuntimeError("pipelined Dadda schedule did not reduce heap to two rows")

    row0 = [col[0] if col else "1'b0" for col in columns]
    row1 = [col[1] if len(col) > 1 else "1'b0" for col in columns]
    lines.append(f"  wire [{width - 1}:0] final_row0 = {{{', '.join(reversed(row0))}}};")
    lines.append(f"  wire [{width - 1}:0] final_row1 = {{{', '.join(reversed(row1))}}};")
    lines.append(f"  wire signed [{width - 1}:0] final_sum = final_row0 + final_row1;")

    latency = len(stage_targets)
    top = f"orcus_dadda_pipelined_{lanes}"
    ports = ["input wire clk", "input wire reset", "input wire in_valid",
             f"output reg out_valid", f"output reg signed [{width - 1}:0] out_data"]
    for lane in range(lanes):
        ports.extend((f"input wire signed [3:0] a{lane}",
                      f"input wire signed [3:0] b{lane}"))
    header = [f"module {top}({', '.join(ports)});",
              f"  localparam integer LANES = {lanes};",
              f"  localparam integer ACC_WIDTH = {width};",
              f"  localparam integer LATENCY = {latency};",
              f"  reg [{latency - 1}:0] valid_pipe;"]
    always = ["  always @(posedge clk) begin", "    if (reset) begin",
              "      valid_pipe <= {"+str(latency)+"{1'b0}};",
              "      out_valid <= 1'b0;", "      out_data <= {"+str(width)+"{1'b0}};"]
    for pairs in stage_registers:
        for qname, _ in pairs:
            always.append(f"      {qname} <= 1'b0;")
    always.extend(["    end else begin", "      valid_pipe[0] <= in_valid;"])
    for index in range(1, latency):
        always.append(f"      valid_pipe[{index}] <= valid_pipe[{index - 1}];")
    for pairs in stage_registers:
        for qname, dname in pairs:
            always.append(f"      {qname} <= {dname};")
    always.extend(["      out_data <= final_sum;",
                   f"      out_valid <= valid_pipe[{latency - 1}];",
                   "    end", "  end", "endmodule"])
    source = "\n".join(header + lines + always) + "\n"
    manifest = {
        "schema": "orcus-dadda-pipeline-v1",
        "lanes": lanes, "width": width, "latency_cycles": latency,
        "stage_targets": stage_targets,
        "initial_heap_bits": initial,
        "compressors": compressor_records,
        "stage_boundaries": boundary_records,
        "final_rows": [row0, row1],
        "reset": "synchronous-active-high",
        "valid_protocol": "in_valid delayed exactly LATENCY rising edges to out_valid",
        "throughput": "one input transaction per cycle",
    }
    return PipelineDesign(lanes, width, latency, tuple(stage_targets), source, manifest)
