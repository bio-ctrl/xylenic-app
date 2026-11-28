"""Cycle-exact Icarus tests for the ORCUS M4.3 K-loop controller."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from orcus.inference_tile import TileConfig

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "orcus" / "rtl" / "orcus_kloop_controller.sv"
SCRATCHPAD = ROOT / "orcus" / "rtl" / "orcus_operand_scratchpad.sv"


def _testbench(lanes: int) -> str:
    cfg = TileConfig(lanes)
    template = r"""`timescale 1ns/1ps
module tb;
    localparam integer LANES = @LANES@;
    localparam integer WIDTH = LANES * 4;
    localparam integer DEPTH = 16384 / WIDTH;
    localparam integer AW = $clog2(DEPTH);
    localparam integer KW = $clog2(DEPTH + 1);
    localparam integer L = @LATENCY@;

    reg clk = 0;
    always #5 clk = ~clk;
    reg reset = 1, start = 0, a_we = 0, b_we = 0, pipeline_done = 0;
    reg [AW-1:0] base_addr = 0, wr_addr = 0;
    reg [KW-1:0] k_length = 0;
    reg [WIDTH-1:0] a_wdata = 0, b_wdata = 0;
    wire busy, rd_en, first, last, valid;
    wire [AW-1:0] rd_addr;
    wire [WIDTH-1:0] a_rd_data, b_rd_data;

    orcus_kloop_controller #(.LANES(LANES)) dut (
        .clk(clk), .reset(reset), .start(start),
        .base_addr(base_addr), .k_length(k_length),
        .wr_addr(wr_addr), .a_we(a_we), .b_we(b_we),
        .pipeline_done(pipeline_done), .busy(busy),
        .rd_en(rd_en), .rd_addr(rd_addr),
        .first(first), .last(last), .valid(valid)
    );

    orcus_operand_scratchpads #(.LANES(LANES)) memories (
        .clk(clk), .wr_addr(wr_addr), .a_we(a_we), .b_we(b_we),
        .a_wdata(a_wdata), .b_wdata(b_wdata),
        .rd_en(rd_en), .rd_addr(rd_addr),
        .a_rd_data(a_rd_data), .b_rd_data(b_rd_data)
    );

    function [WIDTH-1:0] pattern_a(input integer address);
        integer lane;
        begin
            pattern_a = 0;
            for (lane = 0; lane < LANES; lane = lane + 1)
                pattern_a[lane*4 +: 4] = (address*3 + lane*5 + 8) & 15;
        end
    endfunction

    function [WIDTH-1:0] pattern_b(input integer address);
        integer lane;
        begin
            pattern_b = 0;
            for (lane = 0; lane < LANES; lane = lane + 1)
                pattern_b[lane*4 +: 4] = (address*7 + lane*3 + 2) & 15;
        end
    endfunction

    task write_row(input integer address);
        begin
            @(negedge clk);
            wr_addr = address;
            a_wdata = pattern_a(address);
            b_wdata = pattern_b(address);
            a_we = 1; b_we = 1;
            @(posedge clk); #1;
            a_we = 0; b_we = 0;
        end
    endtask

    task check_beat(input integer address, input integer index,
                    input integer count);
        begin
            if (valid !== 1'b1)
                $fatal(1, "valid missing LANES=%0d addr=%0d", LANES, address);
            if (first !== (index == 0))
                $fatal(1, "first misaligned LANES=%0d index=%0d", LANES, index);
            if (last !== (index == count-1))
                $fatal(1, "last misaligned LANES=%0d index=%0d", LANES, index);
            if (a_rd_data !== pattern_a(address))
                $fatal(1, "A data/tag misalignment LANES=%0d addr=%0d", LANES, address);
            if (b_rd_data !== pattern_b(address))
                $fatal(1, "B data/tag misalignment LANES=%0d addr=%0d", LANES, address);
        end
    endtask

    task next_beat(input integer address, input integer index,
                   input integer count);
        begin
            @(negedge clk);
            if (rd_en !== 1'b1 || rd_addr !== address[AW-1:0])
                $fatal(1, "non-consecutive read issue LANES=%0d expected=%0d got_en=%b got_addr=%0d",
                       LANES, address, rd_en, rd_addr);
            @(posedge clk); #1;
            check_beat(address, index, count);
        end
    endtask

    task drain_to_done(input integer count, input integer next_base,
                       input integer next_count);
        integer cycle;
        begin
            // From the final read edge, the result register is due L+2
            // edges later. pipeline_done is asserted on that exact edge.
            for (cycle = 0; cycle < L+1; cycle = cycle + 1) begin
                @(posedge clk); #1;
                if (valid || first || last)
                    $fatal(1, "stale tag during drain LANES=%0d", LANES);
            end
            @(negedge clk);
            pipeline_done = 1;
            if (next_count > 0) begin
                start = 1;
                base_addr = next_base;
                k_length = next_count;
            end
            @(posedge clk); #1;
            if (next_count > 0) begin
                check_beat(next_base, 0, next_count);
                if (!busy)
                    $fatal(1, "back-to-back request lost busy state");
            end else begin
                if (busy)
                    $fatal(1, "busy did not clear on pipeline_done");
                if (valid || first || last)
                    $fatal(1, "tag unexpectedly asserted at completion");
            end
            start = 0;
            pipeline_done = 0;
        end
    endtask

    integer address;
    initial begin
        // Establish defined controller state before any request or memory write.
        @(posedge clk); #1;
        @(negedge clk); reset = 0;

        // Initialize all addresses used by the directed tests, including
        // the final legal word, with independent A/B deterministic patterns.
        for (address = 0; address < 8; address = address + 1)
            write_row(address);
        write_row(DEPTH-1);

        // K=1: first and last must be asserted on the same synchronous beat.
        @(negedge clk);
        base_addr = 2; k_length = 1; start = 1;
        @(posedge clk); #1;
        check_beat(2, 0, 1);
        if (!busy) $fatal(1, "busy not asserted after accepted K=1");
        start = 0;
        drain_to_done(1, 4, 3);

        // K=3: three consecutive addresses, with no issue bubble or tag drift.
        // The previous transaction was replaced on its exact completion edge.
        next_beat(5, 1, 3);
        next_beat(6, 2, 3);
        drain_to_done(3, 0, 1);

        // The chained K=1 transaction at address zero is active now.
        drain_to_done(1, 0, 0);

        // Reset in the middle of a new K=3 burst suppresses all stale tags.
        @(negedge clk);
        base_addr = 1; k_length = 3; start = 1;
        @(posedge clk); #1;
        check_beat(1, 0, 3);
        @(negedge clk);
        reset = 1; start = 0; pipeline_done = 0;
        @(posedge clk); #1;
        if (busy || valid || first || last || rd_en)
            $fatal(1, "synchronous reset failed to flush active transaction");
        @(negedge clk); reset = 0;
        repeat (3) begin
            @(posedge clk); #1;
            if (valid || first || last || rd_en)
                $fatal(1, "ghost tag/read after reset");
        end

        // Invalid K=0 and end-exclusive overflow must be silently rejected.
        @(negedge clk); base_addr = 0; k_length = 0; start = 1;
        @(posedge clk); #1;
        if (busy || valid || first || last || rd_en)
            $fatal(1, "K=0 request was not rejected");
        @(negedge clk); start = 0; base_addr = DEPTH-1; k_length = 2; start = 1;
        @(posedge clk); #1;
        if (busy || valid || first || last || rd_en)
            $fatal(1, "address-wrap request was not rejected");

        // Same-cycle host write to the first requested address is rejected.
        @(negedge clk);
        start = 1; base_addr = 3; k_length = 1;
        wr_addr = 3; a_we = 1; a_wdata = pattern_a(3);
        @(posedge clk); #1;
        if (busy || valid || first || last || rd_en)
            $fatal(1, "start/write address collision was not rejected");
        @(negedge clk); start = 0; a_we = 0;

        // A start while busy is ignored; it must not perturb the live stream.
        base_addr = 2; k_length = 2; start = 1;
        @(posedge clk); #1;
        check_beat(2, 0, 2);
        @(negedge clk); base_addr = 7; k_length = 1;
        @(posedge clk); #1;
        check_beat(3, 1, 2);
        if (rd_addr !== 0 && rd_en)
            $fatal(1, "overlapping start changed active request");
        start = 0;
        // Reset is the abort mechanism for this final intentionally partial request.
        @(negedge clk); reset = 1;
        @(posedge clk); #1;
        if (busy || valid || first || last)
            $fatal(1, "reset failed to abort partial final request");

        $display("PASS K-loop LANES=%0d DEPTH=%0d L=%0d", LANES, DEPTH, L);
        $finish;
    end
endmodule
"""
    return (template
            .replace("@LANES@", str(lanes))
            .replace("@LATENCY@", str(cfg.core_latency_cycles)))


@unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"),
                     "Icarus Verilog is required")
class KLoopControllerRTLTests(unittest.TestCase):
    def test_all_supported_lane_configurations(self) -> None:
        for lanes in (4, 8, 16, 32):
            with self.subTest(lanes=lanes), tempfile.TemporaryDirectory() as temp:
                temp_path = Path(temp)
                tb_path = temp_path / "tb.sv"
                sim_path = temp_path / "sim.out"
                tb_path.write_text(_testbench(lanes))
                compile_result = subprocess.run(
                    ["iverilog", "-g2012", "-s", "tb", "-o", str(sim_path),
                     str(SCRATCHPAD), str(RTL), str(tb_path)],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(compile_result.returncode, 0,
                                 compile_result.stdout + compile_result.stderr)
                run_result = subprocess.run(
                    ["vvp", str(sim_path)], cwd=ROOT, text=True,
                    capture_output=True, check=False,
                )
                self.assertEqual(run_result.returncode, 0,
                                 run_result.stdout + run_result.stderr)
                self.assertIn(f"PASS K-loop LANES={lanes}", run_result.stdout)


if __name__ == "__main__":
    unittest.main()
